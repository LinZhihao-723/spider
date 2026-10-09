use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::SystemTime;

use spider_core::job::JobState;
use spider_core::task::TaskIndex;
use spider_core::task::TaskState;
use spider_core::types::id::ExecutionManagerId;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use spider_core::types::id::TaskId;
use spider_core::types::id::TaskInstanceId;
use spider_core::types::io::ExecutionContext;
use spider_core::types::io::SerializedTaskOutputs;
use spider_core::types::io::TaskOutput;
use tokio::sync::RwLockReadGuard;
use tokio::sync::RwLockWriteGuard;

use crate::cache::error::CacheError;
use crate::cache::error::InternalError;
use crate::cache::error::StaleStateError;
use crate::cache::task::PreparedChild;
use crate::cache::task::TaskGraph;
use crate::cache::tree::ChildSetSummary;
use crate::cache::tree::Lane;
use crate::cache::tree::TreeControlBlock;
use crate::cache::tree::TreeEvent;
use crate::cache::tree::TreeEventKind;
use crate::cache::tree::TreeOutcome;
use crate::db::RecoverableJobContext;
use crate::inbound_queue::InboundQueueSender;
use crate::job_submission::ValidatedJobSubmission;
use crate::limits::MAX_CHILDREN_PER_JOB;
use crate::limits::MAX_JOB_CHILD_BYTES;
use crate::task_instance_pool::TaskInstanceMetadata;
use crate::task_instance_pool::TaskInstancePoolConnector;
use crate::test_hooks::HookContext;
use crate::test_hooks::HookPoint;

/// A shareable control block for a job.
///
/// All mutable state, including the task graph, connectors, and inbound-queue sender, is held
/// inside the
/// underlying [`JobExecutionState`] and protected by [`JobExecutionStateHandle`]'s read-write lock.
///
/// Job-level transitions are submitted as events to the job's tree.
///
/// # Type Parameters
///
/// * `InboundQueueSenderType` - The type of the inbound queue sender.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
#[derive(Clone)]
pub struct SharedJobControlBlock<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> {
    inner: Arc<JobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>>,
}

impl<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> SharedJobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>
{
    /// Factory function.
    ///
    /// Creates the control block of a new root job in [`JobState::Ready`], whose events go to
    /// `tree`.
    ///
    /// # Returns
    ///
    /// The created [`SharedJobControlBlock`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`TaskGraph::create`]'s return values on failure.
    pub async fn create(
        id: JobId,
        owner_id: ResourceGroupId,
        job_submission: ValidatedJobSubmission,
        inbound_queue_sender: InboundQueueSenderType,
        task_instance_pool_connector: TaskInstancePoolConnectorType,
        tree: TreeControlBlock,
    ) -> Result<Self, CacheError> {
        let num_tasks = job_submission.task_graph().get_num_tasks();
        let task_graph = TaskGraph::create(job_submission).await?;
        Ok(Self::from_parts(
            id,
            owner_id,
            None,
            0,
            tree,
            JobExecutionState::new(
                JobState::Ready,
                task_graph,
                num_tasks,
                inbound_queue_sender,
                task_instance_pool_connector,
            ),
        ))
    }

    /// Factory function.
    ///
    /// Creates the control block of a child of `parent` from its prepared child, in
    /// [`JobState::Running`], in `parent`'s tree.
    ///
    /// # Returns
    ///
    /// The created [`SharedJobControlBlock`].
    #[must_use]
    pub fn from_prepared_child(
        id: JobId,
        parent: &Self,
        prepared_child: PreparedChild,
        inbound_queue_sender: InboundQueueSenderType,
        task_instance_pool_connector: TaskInstancePoolConnectorType,
    ) -> Self {
        Self::from_parts(
            id,
            parent.owner_id(),
            Some(parent.id()),
            parent.depth() + 1,
            parent.tree().clone(),
            JobExecutionState::new(
                JobState::Running,
                prepared_child.graph,
                prepared_child.num_tasks,
                inbound_queue_sender,
                task_instance_pool_connector,
            ),
        )
    }

    /// Recovers a job control block from the given recoverable job context (assumed to be returned
    /// from persistent storage, with its state repaired), as a job at `depth` in `tree`.
    ///
    /// # NOTE
    ///
    /// * This constructor does not mutate the storage states.
    /// * This constructor does not send recovered tasks to the inbound-queue.
    ///
    /// # Returns
    ///
    /// The recovered [`SharedJobControlBlock`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::UnexpectedJobState`] if `state` is not recoverable.
    /// * [`InternalError::InvalidRecoverableJobContext`] if:
    ///   * A commit-ready job doesn't have a commit task specified.
    ///   * A commit-ready job doesn't have outputs ready.
    ///   * A blocking-on-children job doesn't have outputs ready.
    ///   * A cleanup-ready job doesn't have a cleanup task specified.
    /// * Forwards [`TaskGraph::create`]'s return values on failure.
    pub async fn recover(
        recoverable_job_context: RecoverableJobContext,
        depth: u32,
        inbound_queue_sender: InboundQueueSenderType,
        task_instance_pool_connector: TaskInstancePoolConnectorType,
        tree: TreeControlBlock,
    ) -> Result<Self, CacheError> {
        let RecoverableJobContext {
            id,
            resource_group_id,
            state,
            parent_id,
            root_id: _,
            submission,
            outputs,
        } = recoverable_job_context;

        let (num_incomplete_tasks, task_graph) = match state {
            JobState::Ready | JobState::Running => {
                let num_tasks = submission.task_graph().get_num_tasks();
                let task_graph = TaskGraph::create(submission).await?;
                (num_tasks, task_graph)
            }
            JobState::CommitReady => {
                let Some(commit_task_descriptor) =
                    submission.task_graph().get_commit_task_descriptor()
                else {
                    return Err(InternalError::InvalidRecoverableJobContext(String::from(
                        "commit-ready job doesn't have a commit task specified",
                    ))
                    .into());
                };
                let Some(outputs) = outputs else {
                    return Err(InternalError::InvalidRecoverableJobContext(String::from(
                        "commit-ready job doesn't have outputs ready",
                    ))
                    .into());
                };
                let task_graph = TaskGraph::recover_from_commit_ready(
                    commit_task_descriptor,
                    submission.task_graph().get_cleanup_task_descriptor(),
                    outputs,
                );
                (0, task_graph)
            }
            JobState::BlockingOnChildren => {
                let Some(outputs) = outputs else {
                    return Err(InternalError::InvalidRecoverableJobContext(String::from(
                        "blocking-on-children job doesn't have outputs ready",
                    ))
                    .into());
                };
                let task_graph = TaskGraph::recover_without_regular_tasks(
                    outputs,
                    submission.task_graph().get_commit_task_descriptor(),
                    submission.task_graph().get_cleanup_task_descriptor(),
                );
                (0, task_graph)
            }
            JobState::CleanupReady => {
                let Some(cleanup_task_descriptor) =
                    submission.task_graph().get_cleanup_task_descriptor()
                else {
                    return Err(InternalError::InvalidRecoverableJobContext(String::from(
                        "cleanup-ready job doesn't have a cleanup task specified",
                    ))
                    .into());
                };
                let task_graph = TaskGraph::recover_from_cleanup_ready(cleanup_task_descriptor);
                (0, task_graph)
            }
            other => {
                return Err(InternalError::UnexpectedJobState {
                    current: other,
                    expected: JobState::Running,
                }
                .into());
            }
        };

        Ok(Self::from_parts(
            id,
            resource_group_id,
            parent_id,
            depth,
            tree,
            JobExecutionState::new(
                state,
                task_graph,
                num_incomplete_tasks,
                inbound_queue_sender,
                task_instance_pool_connector,
            ),
        ))
    }

    /// Returns the job ID.
    #[must_use]
    pub fn id(&self) -> JobId {
        self.inner.id
    }

    #[must_use]
    pub fn owner_id(&self) -> ResourceGroupId {
        self.inner.owner_id
    }

    /// # Returns
    ///
    /// The ID of the job's parent, or [`None`] for a root.
    #[must_use]
    pub fn parent_id(&self) -> Option<JobId> {
        self.inner.parent_id
    }

    /// # Returns
    ///
    /// The job's depth in its tree, 0 for a root.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.inner.depth
    }

    /// # Returns
    ///
    /// The control block of the job's tree.
    #[must_use]
    pub fn tree(&self) -> &TreeControlBlock {
        &self.inner.tree
    }

    #[must_use]
    pub fn has_commit_task(&self) -> bool {
        self.inner.has_commit_task
    }

    #[must_use]
    pub fn has_cleanup_task(&self) -> bool {
        self.inner.has_cleanup_task
    }

    /// Reserves `children` against [`MAX_CHILDREN_PER_JOB`] and [`MAX_JOB_CHILD_BYTES`], on top of
    /// the children that the job's earlier reports reserved.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`ChildSetSummary`] with the job's totals including `children`, if they exceed a per-job
    ///   limit. Nothing is reserved then.
    pub fn reserve_children(&self, children: ChildSetSummary) -> Result<(), ChildSetSummary> {
        let mut reserved = self
            .inner
            .reserved_children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let totals = ChildSetSummary {
            count: reserved.count + children.count,
            bytes: reserved.bytes + children.bytes,
        };
        if totals.count > MAX_CHILDREN_PER_JOB || totals.bytes > MAX_JOB_CHILD_BYTES {
            return Err(totals);
        }
        *reserved = totals;
        drop(reserved);
        Ok(())
    }

    /// Releases a reservation of [`Self::reserve_children`] whose children were not stored.
    pub fn release_children(&self, children: ChildSetSummary) {
        let mut reserved = self
            .inner
            .reserved_children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reserved.count -= children.count;
        reserved.bytes -= children.bytes;
    }

    /// # Returns
    ///
    /// The current job state.
    pub async fn state(&self) -> JobState {
        self.inner.job_execution_state.read_state().await
    }

    /// Gets the outputs of the job from the in-memory task graph.
    ///
    /// # Returns
    ///
    /// The outputs of the job on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`JobExecutionStateHandle::read_succeeded`]'s return values on failure.
    /// * Forwards [`TaskGraph::read_output_payloads`]'s return values on failure.
    pub async fn get_outputs(&self) -> Result<Vec<TaskOutput>, CacheError> {
        let jcb = &self.inner;
        let job = jcb.job_execution_state.read_succeeded().await?;
        let outputs = job.task_graph.read_output_payloads().await?;
        drop(job);
        Ok(outputs)
    }

    /// Starts the job.
    ///
    /// Any tasks in [`TaskState::Ready`] will be enqueued to the inbound queue on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`Self::submit_event`]'s return values on failure.
    pub async fn start(&self) -> Result<(), CacheError> {
        self.submit_event(TreeEventKind::Start).await.map(|_| ())
    }

    /// Resends all ready tasks to the inbound queue.
    ///
    /// The method handles the following job states:
    ///
    /// * [`JobState::Running`] — all tasks in [`TaskState::Ready`] via
    ///   [`InboundQueueSender::send_task_ready`].
    /// * [`JobState::CommitReady`] — the commit task via [`InboundQueueSender::send_commit_ready`].
    /// * [`JobState::CleanupReady`] — the cleanup task via
    ///   [`InboundQueueSender::send_cleanup_ready`].
    ///
    /// For other job states, this method is a no-op and returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`InboundQueueSender::send_task_ready`]'s return values on failure.
    /// * Forwards [`InboundQueueSender::send_commit_ready`]'s return values on failure.
    /// * Forwards [`InboundQueueSender::send_cleanup_ready`]'s return values on failure.
    pub async fn resend_ready_tasks(&self) -> Result<(), CacheError> {
        let jcb = &self.inner;
        let job = jcb.job_execution_state.read_state_guard().await;
        let state = job.state;
        let sender = job.inbound_queue_sender.clone();
        let ready_task_indices = if state == JobState::Running {
            job.task_graph.get_all_ready_task_indices().await
        } else {
            Vec::new()
        };
        drop(job);
        match state {
            JobState::Running if !ready_task_indices.is_empty() => {
                sender
                    .send_task_ready(jcb.owner_id, jcb.id, ready_task_indices)
                    .await?;
            }
            JobState::CommitReady => sender.send_commit_ready(jcb.owner_id, jcb.id).await?,
            JobState::CleanupReady => sender.send_cleanup_ready(jcb.owner_id, jcb.id).await?,
            _ => {}
        }
        Ok(())
    }

    /// Creates a task instance for the given task and registers it in the task instance pool.
    ///
    /// # Returns
    ///
    /// The execution context for the created task instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`Self::create_regular_task_instance`]'s return values on failure.
    /// * Forwards [`Self::create_commit_task_instance`]'s return values on failure.
    /// * Forwards [`Self::create_cleanup_task_instance`]'s return values on failure.
    pub async fn create_task_instance(
        &self,
        task_id: TaskId,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<ExecutionContext, CacheError> {
        let jcb = &self.inner;
        match task_id {
            TaskId::Index(task_index) => {
                Self::create_regular_task_instance(jcb, task_index, execution_manager_id).await
            }
            TaskId::Commit => Self::create_commit_task_instance(jcb, execution_manager_id).await,
            TaskId::Cleanup => Self::create_cleanup_task_instance(jcb, execution_manager_id).await,
        }
    }

    /// Marks the task instance as succeeded, with the children that its report prepared.
    ///
    /// If all tasks have succeeded, submits the job's [`TreeEventKind::RegularTasksDone`] event and
    /// waits until the tree's committer has persisted the job outputs and transitioned the job:
    /// to [`JobState::BlockingOnChildren`] with the children of all its tasks registered, if any
    /// task prepared children. Otherwise, if the completed task unblocks any child tasks, those
    /// child tasks are enqueued to the inbound queue.
    ///
    /// # Returns
    ///
    /// The current [`JobState`] after the operation on success. Must be one of:
    ///
    /// * [`JobState::Running`]
    /// * [`JobState::BlockingOnChildren`]
    /// * [`JobState::CommitReady`]
    /// * [`JobState::Succeeded`]
    /// * [`JobState::Failed`], if the job's outputs or children break a per-job limit.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::TaskIndexOutOfBound`] if the task index is out of range.
    /// * [`InternalError::TaskGraphCorrupted`] if no incomplete tasks remain while new ready tasks
    ///   are generated.
    /// * Forwards [`JobExecutionStateHandle::read_running`]'s return values on failure.
    /// * Forwards [`SharedTaskControlBlock::succeed_task_instance`]'s return values on failure.
    /// * Forwards [`InboundQueueSender::send_task_ready`]'s return values on failure.
    /// * Forwards [`Self::submit_event`]'s return values on failure.
    pub async fn succeed_task_instance(
        &self,
        task_instance_id: TaskInstanceId,
        task_index: TaskIndex,
        task_outputs: Vec<TaskOutput>,
        prepared_children: Vec<PreparedChild>,
    ) -> Result<JobState, CacheError> {
        let jcb = &self.inner;
        let job = jcb.job_execution_state.read_running().await?;
        let tcb = job
            .task_graph
            .get_task_control_block(task_index)
            .ok_or(InternalError::TaskIndexOutOfBound)?;
        let reports_children = !prepared_children.is_empty();
        let ready_task_indices = tcb
            .succeed_task_instance(
                task_instance_id,
                task_outputs,
                prepared_children,
                jcb.tree.hooks(),
                || jcb.hook_context(Some(task_instance_id)),
            )
            .await?;
        if reports_children {
            job.has_prepared_children.store(true, Ordering::Relaxed);
        }
        jcb.hit_hook(HookPoint::PhaseAAfterTcbFlip, Some(task_instance_id))
            .await?;
        // NOTE: `fetch_sub` returns the previous value, so the new count is the returned value
        // minus 1.
        let num_incomplete_tasks = job.num_incomplete_tasks.fetch_sub(1, Ordering::AcqRel) - 1;

        if !ready_task_indices.is_empty() {
            if num_incomplete_tasks == 0 {
                return Err(InternalError::TaskGraphCorrupted(
                    "no incomplete tasks while new ready task indices are generated".to_owned(),
                )
                .into());
            }
            let sender = job.inbound_queue_sender.clone();
            let state = job.state;
            drop(job);
            sender
                .send_task_ready(jcb.owner_id, jcb.id, ready_task_indices)
                .await?;
            return Ok(state);
        }

        if num_incomplete_tasks != 0 {
            return Ok(job.state);
        }

        let registers = job.has_prepared_children.load(Ordering::Relaxed);
        drop(job);
        jcb.hit_hook(HookPoint::PhaseABeforeSubmit, Some(task_instance_id))
            .await?;
        self.submit_event(TreeEventKind::RegularTasksDone { registers })
            .await
    }

    /// Marks the commit task instance as succeeded and transitions the job to
    /// [`JobState::Succeeded`].
    ///
    /// # Returns
    ///
    /// [`JobState::Succeeded`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::UndefinedCommitTask`] if the job has no commit task.
    /// * Forwards [`JobExecutionStateHandle::read_commit_ready`]'s return values on failure.
    /// * Forwards [`SharedTerminationTaskControlBlock::succeed_task_instance`]'s return values on
    ///   failure.
    /// * Forwards [`Self::submit_event`]'s return values on failure.
    pub async fn succeed_commit_task_instance(
        &self,
        task_instance_id: TaskInstanceId,
    ) -> Result<JobState, CacheError> {
        let jcb = &self.inner;
        let job = jcb.job_execution_state.read_commit_ready().await?;
        job.task_graph
            .get_commit_task_control_block()
            .ok_or(InternalError::UndefinedCommitTask)?
            .succeed_task_instance(task_instance_id)
            .await?;
        drop(job);
        self.submit_event(TreeEventKind::CommitDone).await
    }

    /// Marks the cleanup task instance as succeeded and transitions the job to
    /// [`JobState::Cancelled`].
    ///
    /// # Returns
    ///
    /// [`JobState::Cancelled`] on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::UndefinedCleanupTask`] if the job has no cleanup task.
    /// * Forwards [`JobExecutionStateHandle::read_cleanup_ready`]'s return values on failure.
    /// * Forwards [`SharedTerminationTaskControlBlock::succeed_task_instance`]'s return values on
    ///   failure.
    /// * Forwards [`Self::submit_event`]'s return values on failure.
    pub async fn succeed_cleanup_task_instance(
        &self,
        task_instance_id: TaskInstanceId,
    ) -> Result<JobState, CacheError> {
        let jcb = &self.inner;
        let job = jcb.job_execution_state.read_cleanup_ready().await?;
        job.task_graph
            .get_cleanup_task_control_block()
            .ok_or(InternalError::UndefinedCleanupTask)?
            .succeed_task_instance(task_instance_id)
            .await?;
        drop(job);
        self.submit_event(TreeEventKind::CleanupDone).await
    }

    /// Marks a task instance as failed.
    ///
    /// If the task has remaining retries, it is re-enqueued to the inbound queue. Otherwise, the
    /// job's [`TreeEventKind::TerminalFailure`] event is submitted, and the job transitions to
    /// [`JobState::Failed`].
    ///
    /// # Returns
    ///
    /// The current [`JobState`] after the operation on success. Must be one of:
    ///
    /// * [`JobState::Running`]
    /// * [`JobState::CommitReady`]
    /// * [`JobState::CleanupReady`]
    /// * [`JobState::Failed`]
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::TaskIndexOutOfBound`] if the task index is out of range.
    /// * [`InternalError::UndefinedCommitTask`] if the job has no commit task.
    /// * [`InternalError::UndefinedCleanupTask`] if the job has no cleanup task.
    /// * Forwards [`JobExecutionStateHandle::read_running`]'s return values on failure.
    /// * Forwards [`JobExecutionStateHandle::read_commit_ready`]'s return values on failure.
    /// * Forwards [`JobExecutionStateHandle::read_cleanup_ready`]'s return values on failure.
    /// * Forwards [`SharedTaskControlBlock::fail_task_instance`]'s return values on failure.
    /// * Forwards [`SharedTerminationTaskControlBlock::fail_task_instance`]'s return values on
    ///   failure.
    /// * Forwards [`InboundQueueSender::send_task_ready`]'s return values on failure.
    /// * Forwards [`InboundQueueSender::send_commit_ready`]'s return values on failure.
    /// * Forwards [`InboundQueueSender::send_cleanup_ready`]'s return values on failure.
    /// * Forwards [`Self::submit_event`]'s return values on failure.
    pub async fn fail_task_instance(
        &self,
        task_instance_id: TaskInstanceId,
        task_id: TaskId,
        error_message: String,
    ) -> Result<JobState, CacheError> {
        let jcb = &self.inner;
        let lane = match task_id {
            TaskId::Index(task_index) => {
                let job = jcb.job_execution_state.read_running().await?;
                let task_state = job
                    .task_graph
                    .get_task_control_block(task_index)
                    .ok_or(InternalError::TaskIndexOutOfBound)?
                    .fail_task_instance(task_instance_id, error_message.clone())
                    .await?;
                if matches!(task_state, TaskState::Ready | TaskState::Running) {
                    let sender = job.inbound_queue_sender.clone();
                    let state = job.state;
                    drop(job);
                    sender
                        .send_task_ready(jcb.owner_id, jcb.id, vec![task_index])
                        .await?;
                    return Ok(state);
                }
                Lane::Regular
            }
            TaskId::Commit => {
                let job = jcb.job_execution_state.read_commit_ready().await?;
                let task_state = job
                    .task_graph
                    .get_commit_task_control_block()
                    .ok_or(InternalError::UndefinedCommitTask)?
                    .fail_task_instance(task_instance_id, error_message.clone())
                    .await?;
                if matches!(task_state, TaskState::Ready | TaskState::Running) {
                    let sender = job.inbound_queue_sender.clone();
                    let state = job.state;
                    drop(job);
                    sender.send_commit_ready(jcb.owner_id, jcb.id).await?;
                    return Ok(state);
                }
                Lane::Commit
            }
            TaskId::Cleanup => {
                let job = jcb.job_execution_state.read_cleanup_ready().await?;
                let task_state = job
                    .task_graph
                    .get_cleanup_task_control_block()
                    .ok_or(InternalError::UndefinedCleanupTask)?
                    .fail_task_instance(task_instance_id, error_message.clone())
                    .await?;
                if matches!(task_state, TaskState::Ready | TaskState::Running) {
                    let sender = job.inbound_queue_sender.clone();
                    let state = job.state;
                    drop(job);
                    sender.send_cleanup_ready(jcb.owner_id, jcb.id).await?;
                    return Ok(state);
                }
                Lane::Cleanup
            }
        };

        self.submit_event(TreeEventKind::TerminalFailure {
            lane,
            message: error_message,
        })
        .await
    }

    /// Cancels the job, which must be the root of its tree, with every other abortable job of its
    /// tree, and enqueues their cleanup tasks (if any).
    ///
    /// # Returns
    ///
    /// The current [`JobState`] after the cancellation operation on success. Must be one of:
    ///
    /// * [`JobState::CleanupReady`]
    /// * [`JobState::Cancelled`]
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`Self::submit_event`]'s return values on failure.
    pub async fn cancel(&self) -> Result<JobState, CacheError> {
        self.submit_event(TreeEventKind::CancelRoot).await
    }

    /// Applies a transition that the tree's committer has persisted to the cached job state.
    ///
    /// A job that a transition deactivates gets its non-terminal tasks cancelled and the children
    /// its tasks prepared dropped, and a job that leaves [`JobState::Running`] or is deactivated
    /// drops its regular tasks.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::CacheDbDivergence`] if the cached state is not `from`.
    pub(crate) async fn apply_tree_state(
        &self,
        from: JobState,
        to: JobState,
    ) -> Result<(), CacheError> {
        let jcb = &self.inner;
        let mut job = jcb.job_execution_state.write().await;
        if job.state != from {
            return Err(InternalError::CacheDbDivergence {
                job_id: jcb.id,
                expected: from,
                actual: job.state,
            }
            .into());
        }
        job.state = to;
        let is_deactivated = matches!(
            to,
            JobState::Failed | JobState::Cancelled | JobState::CleanupReady
        ) && matches!(
            from,
            JobState::Ready
                | JobState::Running
                | JobState::BlockingOnChildren
                | JobState::CommitReady
        );
        if is_deactivated {
            job.task_graph.cancel_non_terminal().await;
            job.task_graph.drop_prepared_children().await;
        }
        if from == JobState::Running || is_deactivated {
            job.task_graph.finalize();
        }
        drop(job);
        Ok(())
    }

    /// Collects what the job's [`TreeEventKind::RegularTasksDone`] event needs, taking the
    /// prepared children out of the job's tasks.
    ///
    /// # Returns
    ///
    /// The job outputs and the children that its tasks prepared, in task-index order, if the job is
    /// [`JobState::Running`], [`None`] otherwise, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`TaskGraph::read_output_payloads`]'s return values on failure.
    pub(crate) async fn collect_regular_done_inputs(
        &self,
    ) -> Result<Option<RegularDoneInputs>, CacheError> {
        let job = self.inner.job_execution_state.read_state_guard().await;
        if job.state != JobState::Running {
            return Ok(None);
        }
        let outputs = job.task_graph.read_output_payloads().await?;
        let children = job.task_graph.take_prepared_children().await;
        drop(job);
        Ok(Some(RegularDoneInputs { outputs, children }))
    }

    /// # Returns
    ///
    /// The indices of the job's ready tasks if the job is [`JobState::Ready`], [`None`] otherwise,
    /// on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::TaskGraphCorrupted`] if a ready job has no ready task.
    pub(crate) async fn collect_ready_task_indices(
        &self,
    ) -> Result<Option<Vec<TaskIndex>>, CacheError> {
        let job = self.inner.job_execution_state.read_state_guard().await;
        if job.state != JobState::Ready {
            return Ok(None);
        }
        let ready_task_indices = job.task_graph.get_all_ready_task_indices().await;
        drop(job);
        if ready_task_indices.is_empty() {
            return Err(InternalError::TaskGraphCorrupted(
                "initial task graph has no ready tasks".to_owned(),
            )
            .into());
        }
        Ok(Some(ready_task_indices))
    }

    /// Wraps the parts of a job at `depth` in its tree, whose parent is `parent_id` ([`None`] for
    /// a root), into a control block.
    fn from_parts(
        id: JobId,
        owner_id: ResourceGroupId,
        parent_id: Option<JobId>,
        depth: u32,
        tree: TreeControlBlock,
        job_execution_state: JobExecutionState<
            InboundQueueSenderType,
            TaskInstancePoolConnectorType,
        >,
    ) -> Self {
        Self {
            inner: Arc::new(JobControlBlock {
                id,
                owner_id,
                parent_id,
                depth,
                has_commit_task: job_execution_state.task_graph.has_commit_task(),
                has_cleanup_task: job_execution_state.task_graph.has_cleanup_task(),
                tree,
                job_execution_state: JobExecutionStateHandle {
                    inner: tokio::sync::RwLock::new(job_execution_state),
                },
                reserved_children: std::sync::Mutex::new(ChildSetSummary { count: 0, bytes: 0 }),
            }),
        }
    }

    /// Submits a job-level event of this job to its tree and waits for the committer's reply.
    ///
    /// # Returns
    ///
    /// The job's state after the event is applied on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`CacheError::StaleState`] if the event lost a race, if the tree's committer has exited,
    ///   or if the job is no longer in the tree, in which case the error is the one the event's job
    ///   state calls for.
    /// * [`CacheError::ServiceStopping`] if the service is shutting down.
    /// * Forwards [`TreeControlBlock::submit_and_wait`]'s return values on failure.
    async fn submit_event(&self, kind: TreeEventKind) -> Result<JobState, CacheError> {
        let tree = &self.inner.tree;
        let event = TreeEvent {
            job_id: self.inner.id,
            kind: kind.clone(),
        };
        match tree.submit_and_wait(event).await? {
            TreeOutcome::Applied(state) => Ok(state),
            TreeOutcome::Stale(error) => Err(error.into()),
            TreeOutcome::Untracked => {
                if tree.is_stopping() {
                    return Err(CacheError::ServiceStopping);
                }
                Err(kind.stale_error(self.state().await).into())
            }
        }
    }

    /// Creates a task instance for a regular task in the job and registers it in the task instance
    /// pool.
    ///
    /// # Returns
    ///
    /// The execution context for the created task instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::TaskIndexOutOfBound`] if the task index is out of range.
    /// * Forwards [`JobExecutionStateHandle::read_running`]'s return values on failure.
    /// * Forwards [`SharedTaskControlBlock::register_task_instance`]'s return values on failure.
    /// * Forwards [`TaskInstancePoolConnector::register_task_instance`]'s return values on failure.
    async fn create_regular_task_instance(
        jcb: &JobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>,
        task_index: TaskIndex,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<ExecutionContext, CacheError> {
        let job = jcb.job_execution_state.read_running().await?;
        let tcb = job
            .task_graph
            .get_task_control_block(task_index)
            .ok_or(InternalError::TaskIndexOutOfBound)?;
        let task_instance_id = job
            .task_instance_pool_connector
            .get_next_available_task_instance_id();
        let execution_context = tcb.register_task_instance(task_instance_id).await?;
        jcb.hit_hook(
            HookPoint::CreateInstanceBeforePoolSend,
            Some(task_instance_id),
        )
        .await?;
        let registration = TaskInstanceMetadata {
            resource_group_id: jcb.owner_id,
            job_id: jcb.id,
            task_id: TaskId::Index(task_index),
            task_instance_id,
            execution_manager_id,
            soft_timeout_ddl: SystemTime::now().checked_add(Duration::from_millis(
                execution_context.timeout_policy.soft_timeout_ms,
            )),
        };
        job.task_instance_pool_connector
            .register_task_instance(tcb.clone(), registration)
            .await?;

        // The lock is intentionally held until just before return so all TCB accesses
        // observe a consistent state within the lock's scope.
        drop(job);
        Ok(execution_context)
    }

    /// Creates a task instance for the commit task and registers it in the task instance pool.
    ///
    /// # Returns
    ///
    /// The execution context for the created task instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::UndefinedCommitTask`] if the job has no commit task.
    /// * Forwards [`JobExecutionStateHandle::read_commit_ready`]'s return values on failure.
    /// * Forwards [`SharedTerminationTaskControlBlock::register_task_instance`]'s return values on
    ///   failure.
    /// * Forwards [`TaskInstancePoolConnector::register_termination_task_instance`]'s return values
    ///   on failure.
    /// * Forwards [`TaskGraph::read_output_payloads`]'s return values on failure.
    /// * Forwards [`SerializedTaskOutputs::serialize_with_size_hint`]'s return values on failure.
    async fn create_commit_task_instance(
        jcb: &JobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<ExecutionContext, CacheError> {
        let job = jcb.job_execution_state.read_commit_ready().await?;
        let commit_tcb = job
            .task_graph
            .get_commit_task_control_block()
            .ok_or(InternalError::UndefinedCommitTask)?;
        let task_instance_id = job
            .task_instance_pool_connector
            .get_next_available_task_instance_id();
        let (tdl_context, timeout_policy) =
            commit_tcb.register_task_instance(task_instance_id).await?;
        let registration = TaskInstanceMetadata {
            resource_group_id: jcb.owner_id,
            job_id: jcb.id,
            task_id: TaskId::Commit,
            task_instance_id,
            execution_manager_id,
            soft_timeout_ddl: SystemTime::now()
                .checked_add(Duration::from_millis(timeout_policy.soft_timeout_ms)),
        };
        job.task_instance_pool_connector
            .register_termination_task_instance(commit_tcb.clone(), registration)
            .await?;

        let task_graph_outputs = job.task_graph.read_output_payloads().await?;
        let serialized_task_io =
            SerializedTaskOutputs::serialize_with_size_hint(&task_graph_outputs)
                .map_err(InternalError::TaskOutputs)?
                .to_raw();

        drop(job);
        Ok(ExecutionContext {
            task_instance_id,
            tdl_context,
            timeout_policy,
            serialized_task_io,
        })
    }

    /// Creates a task instance for the cleanup task and registers it in the task instance pool.
    ///
    /// # Returns
    ///
    /// The execution context for the created task instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::UndefinedCleanupTask`] if the job has no cleanup task.
    /// * Forwards [`JobExecutionStateHandle::read_cleanup_ready`]'s return values on failure.
    /// * Forwards [`SharedTerminationTaskControlBlock::register_task_instance`]'s return values on
    ///   failure.
    /// * Forwards [`TaskInstancePoolConnector::register_termination_task_instance`]'s return values
    ///   on failure.
    async fn create_cleanup_task_instance(
        jcb: &JobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<ExecutionContext, CacheError> {
        let job = jcb.job_execution_state.read_cleanup_ready().await?;
        let cleanup_tcb = job
            .task_graph
            .get_cleanup_task_control_block()
            .ok_or(InternalError::UndefinedCleanupTask)?;
        let task_instance_id = job
            .task_instance_pool_connector
            .get_next_available_task_instance_id();
        let (tdl_context, timeout_policy) =
            cleanup_tcb.register_task_instance(task_instance_id).await?;
        let registration = TaskInstanceMetadata {
            resource_group_id: jcb.owner_id,
            job_id: jcb.id,
            task_id: TaskId::Cleanup,
            task_instance_id,
            execution_manager_id,
            soft_timeout_ddl: SystemTime::now()
                .checked_add(Duration::from_millis(timeout_policy.soft_timeout_ms)),
        };
        job.task_instance_pool_connector
            .register_termination_task_instance(cleanup_tcb.clone(), registration)
            .await?;

        drop(job);
        Ok(ExecutionContext {
            task_instance_id,
            tdl_context,
            timeout_policy,
            serialized_task_io: Vec::new(),
        })
    }
}

/// What a job's [`TreeEventKind::RegularTasksDone`] event needs, collected by the tree's
/// committer.
pub(crate) struct RegularDoneInputs {
    pub outputs: Vec<TaskOutput>,
    pub children: Vec<PreparedChild>,
}

/// The control block for a job.
///
/// This struct holds the immutable identity of a job, its place in its tree, and a handle to its
/// execution state. All mutable state and connectors live inside [`JobExecutionState`] and are
/// protected by the read-write lock in [`JobExecutionStateHandle`].
///
/// # Type Parameters
///
/// * `InboundQueueSenderType` - The type of the inbound queue sender.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
struct JobControlBlock<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> {
    id: JobId,
    owner_id: ResourceGroupId,
    parent_id: Option<JobId>,
    depth: u32,
    has_commit_task: bool,
    has_cleanup_task: bool,
    tree: TreeControlBlock,
    job_execution_state:
        JobExecutionStateHandle<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    reserved_children: std::sync::Mutex<ChildSetSummary>,
}

impl<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> JobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>
{
    /// Fires the tree's hooks at `point` for this job.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`crate::test_hooks::TestHooks::hit`]'s return values on failure.
    async fn hit_hook(
        &self,
        point: HookPoint,
        task_instance_id: Option<TaskInstanceId>,
    ) -> Result<(), CacheError> {
        Ok(self
            .tree
            .hooks()
            .hit(point, || self.hook_context(task_instance_id))
            .await?)
    }

    /// # Returns
    ///
    /// The hook context of this job, for a hit about `task_instance_id`, if any.
    fn hook_context(&self, task_instance_id: Option<TaskInstanceId>) -> HookContext {
        HookContext {
            task_instance_id,
            ..HookContext::job(self.tree.root_id(), self.id)
        }
    }
}

/// A concurrency-safe handle to a job's execution state.
///
/// This type wraps [`JobExecutionState`] in a read-write lock and provides controlled access to it.
/// All read accessors except [`Self::read_state_guard`] enforce state invariants by validating the
/// underlying job state before returning a read guard. The only writer is the tree's committer,
/// through [`Self::write`].
///
/// # Type Parameters
///
/// The type parameters are forwarded directly to [`JobExecutionState`] in the same declaration
/// order. Single-character names are used to:
///
/// * Reduce verbosity while preserving consistency with the underlying type.
/// * Avoid formatting issues, as `rustfmt` does not handle line wrapping well when using more
///   descriptive type parameter names in this particular struct.
struct JobExecutionStateHandle<R: InboundQueueSender, T: TaskInstancePoolConnector> {
    inner: tokio::sync::RwLock<JobExecutionState<R, T>>,
}

impl<R: InboundQueueSender, T: TaskInstancePoolConnector> JobExecutionStateHandle<R, T> {
    /// # Returns
    ///
    /// The current job state.
    async fn read_state(&self) -> JobState {
        self.inner.read().await.state
    }

    /// # Returns
    ///
    /// A reader guard of the underlying job execution state, whatever the job state is.
    async fn read_state_guard(&self) -> RwLockReadGuard<'_, JobExecutionState<R, T>> {
        self.inner.read().await
    }

    /// # Returns
    ///
    /// A writer guard of the underlying job execution state, whatever the job state is.
    async fn write(&self) -> RwLockWriteGuard<'_, JobExecutionState<R, T>> {
        self.inner.write().await
    }

    /// # Returns
    ///
    /// A reader guard of the underlying job execution state on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`JobExecutionState::ensure_running`]'s return values on failure.
    async fn read_running(
        &self,
    ) -> Result<RwLockReadGuard<'_, JobExecutionState<R, T>>, CacheError> {
        self.validate_and_read(JobExecutionState::ensure_running)
            .await
    }

    /// # Returns
    ///
    /// A reader guard of the underlying job execution state on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`JobExecutionState::ensure_commit_ready`]'s return values on failure.
    async fn read_commit_ready(
        &self,
    ) -> Result<RwLockReadGuard<'_, JobExecutionState<R, T>>, CacheError> {
        self.validate_and_read(JobExecutionState::ensure_commit_ready)
            .await
    }

    /// # Returns
    ///
    /// A reader guard of the underlying job execution state on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`JobExecutionState::ensure_cleanup_ready`]'s return values on failure.
    async fn read_cleanup_ready(
        &self,
    ) -> Result<RwLockReadGuard<'_, JobExecutionState<R, T>>, CacheError> {
        self.validate_and_read(JobExecutionState::ensure_cleanup_ready)
            .await
    }

    /// # Returns
    ///
    /// A reader guard of the underlying job execution state on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`JobExecutionState::ensure_succeeded`]'s return values on failure.
    async fn read_succeeded(
        &self,
    ) -> Result<RwLockReadGuard<'_, JobExecutionState<R, T>>, CacheError> {
        self.validate_and_read(JobExecutionState::ensure_succeeded)
            .await
    }

    /// # Returns
    ///
    /// A reader guard of the underlying job execution state on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards `validator`'s return values on failure.
    async fn validate_and_read(
        &self,
        validator: fn(&JobExecutionState<R, T>) -> Result<(), CacheError>,
    ) -> Result<RwLockReadGuard<'_, JobExecutionState<R, T>>, CacheError> {
        let guard = self.inner.read().await;
        validator(&guard)?;
        Ok(guard)
    }
}

/// Represents the execution state of a job.
///
/// This struct holds all mutable job state, including the task graph, connectors, and
/// inbound-queue sender,
/// so that concurrent access is synchronized through [`JobExecutionStateHandle`]'s read-write lock.
///
/// # Type Parameters
///
/// * `InboundQueueSenderType` - The type of the inbound queue sender.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
struct JobExecutionState<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> {
    state: JobState,
    task_graph: TaskGraph,
    num_incomplete_tasks: AtomicUsize,
    has_prepared_children: AtomicBool,
    inbound_queue_sender: InboundQueueSenderType,
    task_instance_pool_connector: TaskInstancePoolConnectorType,
}

impl<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> JobExecutionState<InboundQueueSenderType, TaskInstancePoolConnectorType>
{
    /// Factory function.
    ///
    /// # Returns
    ///
    /// The execution state of a job in `state` with `num_incomplete_tasks` regular tasks left and
    /// no prepared children.
    const fn new(
        state: JobState,
        task_graph: TaskGraph,
        num_incomplete_tasks: usize,
        inbound_queue_sender: InboundQueueSenderType,
        task_instance_pool_connector: TaskInstancePoolConnectorType,
    ) -> Self {
        Self {
            state,
            task_graph,
            num_incomplete_tasks: AtomicUsize::new(num_incomplete_tasks),
            has_prepared_children: AtomicBool::new(false),
            inbound_queue_sender,
            task_instance_pool_connector,
        }
    }

    /// Ensures that the job is currently in the [`JobState::Running`] state.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::JobNotStarted`] if the job hasn't been started yet.
    /// * [`StaleStateError::JobNoLongerRunning`] if the job is no longer running.
    fn ensure_running(&self) -> Result<(), CacheError> {
        if !self.state.is_running() {
            if matches!(self.state, JobState::Ready) {
                return Err(InternalError::JobNotStarted.into());
            }
            return Err(StaleStateError::JobNoLongerRunning.into());
        }
        Ok(())
    }

    /// Ensures that the job is currently in the [`JobState::CommitReady`] state.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::UnexpectedJobState`] if the job is in an unexpected state.
    /// * [`StaleStateError::JobNoLongerCommitReady`] if the job is no longer commit-ready.
    fn ensure_commit_ready(&self) -> Result<(), CacheError> {
        if !matches!(self.state, JobState::CommitReady) {
            if self.state.is_terminal()
                || matches!(
                    self.state,
                    JobState::CleanupReady | JobState::BlockingOnChildren
                )
            {
                return Err(StaleStateError::JobNoLongerCommitReady.into());
            }
            return Err(InternalError::UnexpectedJobState {
                current: self.state,
                expected: JobState::CommitReady,
            }
            .into());
        }
        Ok(())
    }

    /// Ensures that the job is currently in the [`JobState::CleanupReady`] state.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::UnexpectedJobState`] if the job is in an unexpected state.
    /// * [`StaleStateError::JobNoLongerCleanupReady`] if the job is no longer cleanup-ready.
    fn ensure_cleanup_ready(&self) -> Result<(), CacheError> {
        if !matches!(self.state, JobState::CleanupReady) {
            if self.state.is_terminal() || matches!(self.state, JobState::BlockingOnChildren) {
                return Err(StaleStateError::JobNoLongerCleanupReady.into());
            }
            return Err(InternalError::UnexpectedJobState {
                current: self.state,
                expected: JobState::CleanupReady,
            }
            .into());
        }
        Ok(())
    }

    /// Ensures that the job is currently in the [`JobState::Succeeded`] state.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StaleStateError::JobNotSucceeded`] if the job is in any other state.
    fn ensure_succeeded(&self) -> Result<(), CacheError> {
        if !matches!(self.state, JobState::Succeeded) {
            return Err(StaleStateError::JobNotSucceeded(self.state).into());
        }
        Ok(())
    }
}
