use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use spider_core::job::JobState;
use spider_core::task::TaskIndex;
use spider_core::types::id::ExecutionManagerId;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use spider_core::types::id::SchedulerId;
use spider_core::types::id::SessionId;
use spider_core::types::id::TaskId;
use spider_core::types::id::TaskInstanceId;
use spider_core::types::io::ExecutionContext;
use spider_core::types::io::TaskOutput;
use spider_core::types::io::TaskOutputsSerializer;
use spider_core::types::resource_group::ExternalResourceGroupCredentials;
use spider_core::types::scheduler::RegisteredScheduler;
use spider_tdl::error::TdlError;
use spider_utils::config::Host;
use tokio_util::task::TaskTracker;

use crate::cache::error::CacheError;
use crate::cache::error::InternalError;
use crate::cache::job::SharedJobControlBlock;
use crate::cache::task::PreparedChild;
use crate::cache::tree::ChildSetSummary;
use crate::cache::tree::RecoveredRow;
use crate::cache::tree::TreeStructure;
use crate::cache::tree::plan_repair;
use crate::db::DbStorage;
use crate::db::RecoverableJobContext;
use crate::db::TreeTransition;
use crate::inbound_queue::CleanupTaskMarker;
use crate::inbound_queue::CommitTaskMarker;
use crate::inbound_queue::InboundQueueEntry;
use crate::inbound_queue::InboundQueueReceiverHandle;
use crate::inbound_queue::InboundQueueSender;
use crate::job_submission::ChildJobSubmission;
use crate::job_submission::ValidatedJobSubmission;
use crate::limits::MAX_CHILDREN_PER_JOB;
use crate::limits::MAX_DEPTH;
use crate::limits::MAX_JOB_CHILD_BYTES;
use crate::limits::MAX_REPORT_CHILD_BYTES;
use crate::limits::MAX_REPORT_OUTPUT_BYTES;
use crate::state::JobCache;
use crate::state::JobCacheGcHandle;
use crate::state::ShutdownHandle;
use crate::state::StagingPublisher;
use crate::state::StorageServerError;
use crate::state::TreeEnv;
use crate::state::create_root_tree;
use crate::state::error::ReportRejection;
use crate::state::tree_committer::create_recovered_tree;
use crate::task_instance_pool::TaskInstancePoolConnector;
use crate::test_hooks::HookContext;
use crate::test_hooks::HookPoint;
use crate::test_hooks::TestHooks;

/// The job cache of a [`ServiceState`], whose jobs publish through the service's
/// [`StagingPublisher`].
pub type ServiceJobCache<TaskInstancePoolConnectorType> =
    JobCache<StagingPublisher, TaskInstancePoolConnectorType>;

/// Bundle of constructor parameters for [`ServiceState::new`].
///
/// This is a work-around for silencing the ` clippy::too_many_arguments ` warning.
///
/// # Type Parameters
///
/// * `InboundQueueSenderType` - The type of the inbound queue sender.
/// * `DbConnectorType` - The type of the DB-layer connector.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
pub struct ServiceStateParams<
    InboundQueueSenderType: InboundQueueSender + 'static,
    DbConnectorType: DbStorage + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
> {
    pub db: DbConnectorType,
    pub session_id: SessionId,
    pub job_cache: ServiceJobCache<TaskInstancePoolConnectorType>,
    pub inbound_queue_sender: InboundQueueSenderType,
    pub inbound_queue_receiver: InboundQueueReceiverHandle,
    pub task_instance_pool_connector: TaskInstancePoolConnectorType,
    pub job_cache_gc_handle: JobCacheGcHandle,
    pub shutdown: ShutdownHandle,
    pub hooks: TestHooks,
}

/// Per-request service state providing access to the storage layer.
///
/// Internally wraps a single [`Arc`] around [`ServiceStateInner`] so that cloning is cheap (one
/// Arc clone instead of cloning each field).
///
/// # Type Parameters
///
/// * `DbConnectorType` - The type of the DB-layer connector.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
#[derive(Clone)]
pub struct ServiceState<
    DbConnectorType: DbStorage + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
> {
    inner: Arc<ServiceStateInner<DbConnectorType, TaskInstancePoolConnectorType>>,
}

impl<
    DbConnectorType: DbStorage + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
> ServiceState<DbConnectorType, TaskInstancePoolConnectorType>
{
    /// Factory function.
    ///
    /// Spawns the publisher's lane tasks into the service's task tracker, so it must be called
    /// within a tokio runtime.
    ///
    /// # Type Parameters
    ///
    /// * `InboundQueueSenderType` - The type of the inbound queue sender that the publisher
    ///   forwards into.
    ///
    /// # Returns
    ///
    /// A newly created [`ServiceState`] whose trees notify the GC actor when cached jobs
    /// terminate.
    #[must_use]
    pub fn new<InboundQueueSenderType: InboundQueueSender + 'static>(
        params: ServiceStateParams<
            InboundQueueSenderType,
            DbConnectorType,
            TaskInstancePoolConnectorType,
        >,
    ) -> Self {
        let ServiceStateParams {
            db,
            session_id,
            job_cache,
            inbound_queue_sender,
            inbound_queue_receiver,
            task_instance_pool_connector,
            job_cache_gc_handle,
            shutdown,
            hooks,
        } = params;
        let tracker = TaskTracker::new();
        let publisher = StagingPublisher::spawn(inbound_queue_sender, shutdown.token(), &tracker);
        hooks.register_runtime(shutdown.token(), &tracker);
        Self {
            inner: Arc::new(ServiceStateInner {
                db,
                session_id,
                job_cache,
                publisher,
                inbound_queue_receiver,
                task_instance_pool_connector,
                job_cache_gc_handle,
                has_previous_scheduler_connection: tokio::sync::Mutex::new(false),
                shutdown,
                tracker,
                hooks,
            }),
        }
    }

    /// # Returns
    ///
    /// The storage session ID owned by this service state.
    #[must_use]
    pub fn session_id(&self) -> SessionId {
        self.inner.session_id
    }

    /// # Returns
    ///
    /// The handle that shuts the storage service down, gracefully or fatally.
    #[must_use]
    pub fn shutdown(&self) -> &ShutdownHandle {
        &self.inner.shutdown
    }

    /// # Returns
    ///
    /// The tracker of the service's spawned tasks: tree committers, publisher lanes and handler
    /// bodies.
    #[must_use]
    pub fn tracker(&self) -> &TaskTracker {
        &self.inner.tracker
    }

    /// # Returns
    ///
    /// The service's test hooks.
    #[must_use]
    pub fn test_hooks(&self) -> &TestHooks {
        &self.inner.hooks
    }

    /// # Returns
    ///
    /// The environment that the service's tree committers run in.
    #[must_use]
    pub fn tree_env(
        &self,
    ) -> TreeEnv<StagingPublisher, DbConnectorType, TaskInstancePoolConnectorType> {
        TreeEnv {
            db: self.inner.db.clone(),
            job_cache: self.inner.job_cache.clone(),
            gc: self.inner.job_cache_gc_handle.clone(),
            publisher: self.inner.publisher.clone(),
            task_instance_pool: self.inner.task_instance_pool_connector.clone(),
            shutdown: self.inner.shutdown.clone(),
            tracker: self.inner.tracker.clone(),
            hooks: self.inner.hooks.clone(),
        }
    }

    /// Recovers every non-terminal job persisted in the database into the cache, tree by tree in
    /// ascending root ID order: each tree is repaired, then its jobs are cached and its committer
    /// is spawned.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`InternalJobOrchestration::get_recoverable_jobs`]'s return values on failure.
    /// * Forwards [`Self::repair_tree`]'s return values on failure.
    /// * Forwards [`create_recovered_tree`]'s return values on failure.
    pub async fn recover_jobs(&self) -> Result<(), StorageServerError> {
        let env = self.tree_env();
        let mut recoverable_jobs = self.inner.db.get_recoverable_jobs().await?;
        recoverable_jobs.sort_by_key(|recoverable_job| recoverable_job.id.get());
        let mut trees: BTreeMap<u64, Vec<RecoverableJobContext>> = BTreeMap::new();
        for recoverable_job in recoverable_jobs {
            let root_id = recoverable_job.root_id.unwrap_or(recoverable_job.id);
            trees
                .entry(root_id.get())
                .or_default()
                .push(recoverable_job);
        }
        for (root_id, mut tree_jobs) in trees {
            let root_id = JobId::from(root_id);
            env.hooks
                .hit(HookPoint::RecoveryBeforeRepair, || HookContext {
                    root_id: Some(root_id),
                    job_ids: tree_jobs.iter().map(|job| job.id).collect(),
                    ..HookContext::default()
                })
                .await
                .map_err(CacheError::from)?;
            let repaired_job_ids = self.repair_tree(root_id, &mut tree_jobs).await?;
            env.hooks
                .hit(HookPoint::RecoveryAfterRepair, || HookContext {
                    root_id: Some(root_id),
                    job_ids: repaired_job_ids,
                    ..HookContext::default()
                })
                .await
                .map_err(CacheError::from)?;
            let recovered_jobs: Vec<(JobId, JobState)> = tree_jobs
                .iter()
                .filter(|job| !job.state.is_terminal())
                .map(|job| (job.id, job.state))
                .collect();
            create_recovered_tree(&env, root_id, tree_jobs).await?;
            for (job_id, job_state) in recovered_jobs {
                tracing::info!(
                    job_id = ? job_id,
                    job_state = ? job_state,
                    "Job recovered into cache.",
                );
            }
        }
        Ok(())
    }

    /// Registers a job in the database and inserts its control block into the cache.
    ///
    /// # Returns
    ///
    /// The ID of the registered job on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ValidatedJobSubmission::create`]'s return values as
    ///   [`StorageServerError::BadRequest`] on failure.
    /// * Forwards [`ExternalJobOrchestration::register`]'s return values on failure.
    /// * Forwards [`create_root_tree`]'s return values on failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn register_job(
        &self,
        resource_group_id: ResourceGroupId,
        compressed_serialized_task_graph: Vec<u8>,
        compressed_serialized_task_graph_input: Vec<u8>,
    ) -> Result<JobId, StorageServerError> {
        let job_submission = ValidatedJobSubmission::create(
            compressed_serialized_task_graph,
            compressed_serialized_task_graph_input,
        )
        .map_err(|e| StorageServerError::BadRequest(e.to_string()))?;

        let this = self.clone();
        self.spawn_body(async move {
            let job_id = this
                .inner
                .db
                .register(resource_group_id, &job_submission)
                .await?;
            tracing::info!(
                job_id = ? job_id,
                rg_id = ? resource_group_id,
                "Job registered in DB."
            );

            create_root_tree(&this.tree_env(), job_id, resource_group_id, job_submission).await?;
            tracing::info!(
                job_id = ? job_id,
                rg_id = ? resource_group_id,
                "Job inserted in cache.",
            );

            Ok(job_id)
        })
        .await
    }

    /// Starts a job for execution.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageServerError::JobNotFound`] if the job is not in the cache.
    /// * Forwards [`ensure_root`]'s return values on failure.
    /// * Forwards [`SharedJobControlBlock::start`]'s return values on failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn start_job(&self, job_id: JobId) -> Result<(), StorageServerError> {
        let this = self.clone();
        self.spawn_body(async move {
            let jcb = this
                .inner
                .job_cache
                .get(job_id)
                .await
                .ok_or(StorageServerError::JobNotFound(job_id))?;
            ensure_root(job_id, jcb.parent_id())?;
            jcb.start().await?;
            tracing::info!(
                job_id = ? job_id,
                "Job started.",
            );
            Ok(())
        })
        .await
    }

    /// Cancels a root job and the rest of its tree.
    ///
    /// # Returns
    ///
    /// The job state after the cancellation operation on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::JobNotFound`] if the job is a root that is not in a terminal state, but
    ///   it is not in the cache.
    /// * [`StorageServerError::JobNotFound`] if the job is not in the cache and not in a terminal
    ///   state in the database.
    /// * Forwards [`ensure_root`]'s return values on failure.
    /// * Forwards [`SharedJobControlBlock::cancel`]'s return values on failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn cancel_job(&self, job_id: JobId) -> Result<JobState, StorageServerError> {
        let this = self.clone();
        self.spawn_body(async move {
            if let Some(jcb) = this.inner.job_cache.get(job_id).await {
                ensure_root(job_id, jcb.parent_id())?;
                let state = jcb.cancel().await?;
                tracing::info!(
                    job_id = ? job_id,
                    "Job cancelled.",
                );
                return Ok(state);
            }
            match this.inner.db.get_state_and_parent(job_id).await {
                Ok((state, parent_id)) => {
                    ensure_root(job_id, parent_id)?;
                    if !state.is_terminal() {
                        // If the job is not terminated, it should always exist in the cache.
                        return Err(StorageServerError::Cache(CacheError::Internal(
                            InternalError::JobNotFound(job_id),
                        )));
                    }
                    Ok(state)
                }
                _ => Err(StorageServerError::JobNotFound(job_id)),
            }
        })
        .await
    }

    /// Gets the state of a job.
    ///
    /// # Returns
    ///
    /// The state of the job on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ensure_root`]'s return values on failure (cache path).
    /// * Forwards [`Self::uncached_root_state`]'s return values on failure (DB fallback).
    pub async fn get_job_state(&self, job_id: JobId) -> Result<JobState, StorageServerError> {
        if let Some(jcb) = self.inner.job_cache.get(job_id).await {
            ensure_root(job_id, jcb.parent_id())?;
            return Ok(jcb.state().await);
        }
        self.uncached_root_state(job_id).await
    }

    /// Gets the outputs of a job.
    ///
    /// # Returns
    ///
    /// The outputs of the job on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ensure_root`]'s return values on failure (cache path).
    /// * Forwards [`SharedJobControlBlock::get_outputs`]'s return values on failure (cache path).
    /// * Forwards [`Self::uncached_root_state`]'s return values on failure (DB fallback).
    /// * Forwards [`ExternalJobOrchestration::get_outputs`]'s return values on failure (DB
    ///   fallback).
    pub async fn get_job_outputs(
        &self,
        job_id: JobId,
    ) -> Result<Vec<TaskOutput>, StorageServerError> {
        if let Some(jcb) = self.inner.job_cache.get(job_id).await {
            ensure_root(job_id, jcb.parent_id())?;
            return Ok(jcb.get_outputs().await?);
        }
        self.uncached_root_state(job_id).await?;
        Ok(self.inner.db.get_outputs(job_id).await?)
    }

    /// Gets the error message of a job.
    ///
    /// # Returns
    ///
    /// The error message of the job on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ensure_root`]'s return values on failure (cache path).
    /// * Forwards [`Self::uncached_root_state`]'s return values on failure (DB fallback).
    /// * Forwards [`ExternalJobOrchestration::get_error`]'s return values on failure.
    pub async fn get_job_error(&self, job_id: JobId) -> Result<String, StorageServerError> {
        match self.inner.job_cache.get(job_id).await {
            Some(jcb) => ensure_root(job_id, jcb.parent_id())?,
            None => {
                self.uncached_root_state(job_id).await?;
            }
        }
        Ok(self.inner.db.get_error(job_id).await?)
    }

    /// Resends ready tasks for all jobs in the cache to the inbound queue.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`JobCache::resend_ready_tasks`]'s return values on failure.
    pub async fn resend_ready_tasks(&self) -> Result<(), StorageServerError> {
        self.inner.job_cache.resend_ready_tasks().await
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
    /// * [`StorageServerError::StaleSession`] if the session has changed.
    /// * [`StorageServerError::JobNotFound`] if the job is not in the cache.
    /// * Forwards [`SharedJobControlBlock::create_task_instance`]'s return values on failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn create_task_instance(
        &self,
        session_id: SessionId,
        job_id: JobId,
        task_id: TaskId,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<ExecutionContext, StorageServerError> {
        self.validate_session(session_id)?;
        let this = self.clone();
        self.spawn_body(async move {
            let jcb = this
                .inner
                .job_cache
                .get(job_id)
                .await
                .ok_or(StorageServerError::JobNotFound(job_id))?;
            let exe_context = jcb
                .create_task_instance(task_id, execution_manager_id)
                .await?;
            tracing::info!(
                job_id = ? job_id,
                task_id = ? task_id,
                execution_manager_id = ? execution_manager_id,
                task_instance_id = ? exe_context.task_instance_id,
                "Task instance created.",
            );
            Ok(exe_context)
        })
        .await
    }

    /// Marks a task instance as succeeded, with the child jobs that its report creates.
    ///
    /// The outputs and the children are checked against the limits, the children are reserved
    /// against the job's per-job child limits, and they are decoded and built, before the job is
    /// touched. A report that breaks a limit, or whose children fail to decode or build, fails the
    /// task instance instead.
    ///
    /// # Returns
    ///
    /// The current job state after the operation on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageServerError::StaleSession`] if the session has changed.
    /// * [`StorageServerError::JobNotFound`] if the job is not in the cache.
    /// * Forwards [`spider_core::types::io::TaskOutputsSerializer::deserialize`]'s return values on
    ///   failure.
    /// * Forwards [`SharedJobControlBlock::succeed_task_instance`]'s return values on failure.
    /// * Forwards [`SharedJobControlBlock::fail_task_instance`]'s return values on failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn succeed_task_instance(
        &self,
        session_id: SessionId,
        job_id: JobId,
        task_instance_id: TaskInstanceId,
        task_index: TaskIndex,
        serialized_outputs: Vec<u8>,
        children: Vec<ChildJobSubmission>,
    ) -> Result<JobState, StorageServerError> {
        self.validate_session(session_id)?;
        let task_outputs = TaskOutputsSerializer::deserialize(&serialized_outputs)
            .map_err(|e| StorageServerError::Tdl(TdlError::DeserializationError(e.to_string())))?;
        let this = self.clone();
        self.spawn_body(async move {
            let jcb = this
                .inner
                .job_cache
                .get(job_id)
                .await
                .ok_or(StorageServerError::JobNotFound(job_id))?;
            let (prepared_children, reserved_children) =
                match this.prepare_report(&jcb, &task_outputs, children).await {
                    Ok(prepared) => prepared,
                    Err(rejection) => {
                        tracing::warn!(
                            job_id = ? job_id,
                            task_id = ? TaskId::Index(task_index),
                            task_instance_id = ? task_instance_id,
                            error = % rejection,
                            "Task success report rejected. Failing the task instance."
                        );
                        let state = jcb
                            .fail_task_instance(
                                task_instance_id,
                                TaskId::Index(task_index),
                                rejection.to_string(),
                            )
                            .await?;
                        tracing::info!(
                            job_id = ? job_id,
                            task_id = ? TaskId::Index(task_index),
                            task_instance_id = ? task_instance_id,
                            job_state = ? state,
                            "Task instance failed for its rejected success report.",
                        );
                        return Ok(state);
                    }
                };
            let state = jcb
                .succeed_task_instance(
                    task_instance_id,
                    task_index,
                    task_outputs,
                    prepared_children,
                )
                .await
                .inspect_err(|_| jcb.release_children(reserved_children))?;
            tracing::info!(
                job_id = ? job_id,
                task_id = ? TaskId::Index(task_index),
                task_instance_id = ? task_instance_id,
                job_state = ? state,
                "Task instance succeeded.",
            );
            Ok(state)
        })
        .await
    }

    /// Marks a commit task instance as succeeded.
    ///
    /// # Returns
    ///
    /// The current job state after the operation on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageServerError::StaleSession`] if the session has changed.
    /// * [`StorageServerError::JobNotFound`] if the job is not in the cache.
    /// * Forwards [`SharedJobControlBlock::succeed_commit_task_instance`]'s return values on
    ///   failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn succeed_commit_task_instance(
        &self,
        session_id: SessionId,
        job_id: JobId,
        task_instance_id: TaskInstanceId,
    ) -> Result<JobState, StorageServerError> {
        self.validate_session(session_id)?;
        let this = self.clone();
        self.spawn_body(async move {
            let jcb = this
                .inner
                .job_cache
                .get(job_id)
                .await
                .ok_or(StorageServerError::JobNotFound(job_id))?;
            let state = jcb.succeed_commit_task_instance(task_instance_id).await?;
            tracing::info!(
                job_id = ? job_id,
                task_id = ? TaskId::Commit,
                task_instance_id = ? task_instance_id,
                job_state = ? state,
                "Task instance succeeded.",
            );
            Ok(state)
        })
        .await
    }

    /// Marks a cleanup task instance as succeeded.
    ///
    /// # Returns
    ///
    /// The current job state after the operation on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageServerError::StaleSession`] if the session has changed.
    /// * [`StorageServerError::JobNotFound`] if the job is not in the cache.
    /// * Forwards [`SharedJobControlBlock::succeed_cleanup_task_instance`]'s return values on
    ///   failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn succeed_cleanup_task_instance(
        &self,
        session_id: SessionId,
        job_id: JobId,
        task_instance_id: TaskInstanceId,
    ) -> Result<JobState, StorageServerError> {
        self.validate_session(session_id)?;
        let this = self.clone();
        self.spawn_body(async move {
            let jcb = this
                .inner
                .job_cache
                .get(job_id)
                .await
                .ok_or(StorageServerError::JobNotFound(job_id))?;
            let state = jcb.succeed_cleanup_task_instance(task_instance_id).await?;
            tracing::info!(
                job_id = ? job_id,
                task_id = ? TaskId::Cleanup,
                task_instance_id = ? task_instance_id,
                job_state = ? state,
                "Task instance succeeded.",
            );
            Ok(state)
        })
        .await
    }

    /// Marks a task instance as failed.
    ///
    /// # Returns
    ///
    /// The current job state after the operation on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageServerError::StaleSession`] if the session has changed.
    /// * [`StorageServerError::JobNotFound`] if the job is not in the cache.
    /// * Forwards [`SharedJobControlBlock::fail_task_instance`]'s return values on failure.
    /// * Forwards [`Self::spawn_body`]'s return values on failure.
    pub async fn fail_task_instance(
        &self,
        session_id: SessionId,
        job_id: JobId,
        task_instance_id: TaskInstanceId,
        task_id: TaskId,
        error: String,
    ) -> Result<JobState, StorageServerError> {
        self.validate_session(session_id)?;
        let this = self.clone();
        self.spawn_body(async move {
            let jcb = this
                .inner
                .job_cache
                .get(job_id)
                .await
                .ok_or(StorageServerError::JobNotFound(job_id))?;
            let state = jcb
                .fail_task_instance(task_instance_id, task_id, error)
                .await?;
            tracing::info!(
                job_id = ? job_id,
                task_id = ? task_id,
                task_instance_id = ? task_instance_id,
                job_state = ? state,
                "Task instance failed.",
            );
            Ok(state)
        })
        .await
    }

    /// Adds a resource group with the given credentials.
    ///
    /// # Returns
    ///
    /// The ID of the created resource group on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ResourceGroupManagement::add`]'s return values on failure.
    pub async fn add_resource_group(
        &self,
        credentials: ExternalResourceGroupCredentials,
    ) -> Result<ResourceGroupId, StorageServerError> {
        let rg_id = self.inner.db.add(credentials).await?;
        tracing::info!(
            rg_id = ? rg_id,
            "Resource group added.",
        );
        Ok(rg_id)
    }

    /// Adds a resource group or verifies the password of an existing resource group.
    ///
    /// # Returns
    ///
    /// The ID of the created or verified resource group on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ResourceGroupManagement::add_or_verify`]'s return values on failure.
    pub async fn add_or_verify_resource_group(
        &self,
        credentials: ExternalResourceGroupCredentials,
    ) -> Result<ResourceGroupId, StorageServerError> {
        let rg_id = self.inner.db.add_or_verify(credentials).await?;
        tracing::info!(
            rg_id = ? rg_id,
            "Resource group added or verified.",
        );
        Ok(rg_id)
    }

    /// Verifies the password of a resource group.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ResourceGroupManagement::verify`]'s return values on failure.
    pub async fn verify_resource_group(
        &self,
        resource_group_id: ResourceGroupId,
        password: &[u8],
    ) -> Result<(), StorageServerError> {
        self.inner
            .db
            .verify(resource_group_id, password)
            .await
            .map_err(StorageServerError::from)
    }

    /// Polls the inbound queue for task entries.
    ///
    /// # Returns
    ///
    /// Up to `max_tasks` inbound queue entries received within the `wait` duration on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`InboundQueueReceiverHandle::recv_tasks`]'s return values on failure.
    pub async fn poll_ready_tasks(
        &self,
        max_tasks: usize,
        wait: Duration,
    ) -> Result<Vec<InboundQueueEntry<TaskIndex>>, StorageServerError> {
        self.inner
            .inbound_queue_receiver
            .recv_tasks(max_tasks, wait)
            .await
            .map_err(|e| CacheError::Internal(e).into())
    }

    /// Polls the inbound queue for commit-ready task entries.
    ///
    /// # Returns
    ///
    /// Up to `max_tasks` commit-ready task entries received within the `wait` duration on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`InboundQueueReceiverHandle::recv_commits`]'s return values on failure.
    pub async fn poll_commit_ready_tasks(
        &self,
        max_tasks: usize,
        wait: Duration,
    ) -> Result<Vec<InboundQueueEntry<CommitTaskMarker>>, StorageServerError> {
        self.inner
            .inbound_queue_receiver
            .recv_commits(max_tasks, wait)
            .await
            .map_err(|e| CacheError::Internal(e).into())
    }

    /// Polls the inbound queue for cleanup-ready task entries.
    ///
    /// # Returns
    ///
    /// Up to `max_tasks` cleanup-ready task entries received within the `wait` duration on
    /// success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`InboundQueueReceiverHandle::recv_cleanups`]'s return values on failure.
    pub async fn poll_cleanup_ready_tasks(
        &self,
        max_tasks: usize,
        wait: Duration,
    ) -> Result<Vec<InboundQueueEntry<CleanupTaskMarker>>, StorageServerError> {
        self.inner
            .inbound_queue_receiver
            .recv_cleanups(max_tasks, wait)
            .await
            .map_err(|e| CacheError::Internal(e).into())
    }

    /// Registers an execution manager.
    ///
    /// # Returns
    ///
    /// A tuple on success, containing:
    ///
    /// * The ID of the registered execution manager.
    /// * The ID of the resource group to which the execution manager is dedicated, if any.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ExecutionManagerLivenessManagement::register_execution_manager`]'s return
    ///   values on failure.
    pub async fn register_execution_manager(
        &self,
        ip_address: IpAddr,
        resource_group_credentials: Option<ExternalResourceGroupCredentials>,
    ) -> Result<(ExecutionManagerId, Option<ResourceGroupId>), StorageServerError> {
        let registration = self
            .inner
            .db
            .register_execution_manager(ip_address, resource_group_credentials)
            .await?;
        tracing::info!(
            em_id = ? registration.0,
            rg_id = ? registration.1,
            ip = ? ip_address,
            "Execution manager registered.",
        );
        Ok(registration)
    }

    /// Updates the heartbeat of an execution manager.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ExecutionManagerLivenessManagement::update_execution_manager_heartbeat`]'s
    ///   return values on failure.
    pub async fn update_execution_manager_heartbeat(
        &self,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<(), StorageServerError> {
        self.inner
            .db
            .update_execution_manager_heartbeat(execution_manager_id)
            .await?;
        tracing::info!(
            em_id = ? execution_manager_id,
            "Execution manager heartbeat updated.",
        );
        Ok(())
    }

    /// Registers a scheduler.
    ///
    /// Scheduler registration is mutually exclusive: only one registration request can be processed
    /// at a time. Registering a scheduler invalidates any scheduler that was previously registered.
    ///
    /// If this replaces an existing scheduler, all ready tasks are re-enqueued in a background task
    /// so they become visible in the inbound queue for the newly registered scheduler.
    ///
    /// # Returns
    ///
    /// The ID of the registered scheduler on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`SchedulerRegistrationManagement::register_scheduler`]'s return values on
    ///   failure.
    pub async fn register_scheduler(
        &self,
        host: Host,
        port: u16,
    ) -> Result<SchedulerId, StorageServerError> {
        let mut has_previous_scheduler_connection =
            self.inner.has_previous_scheduler_connection.lock().await;
        let scheduler_id = self
            .inner
            .db
            .register_scheduler(host.as_str(), port)
            .await?;
        tracing::info!(
            scheduler_id = ? scheduler_id,
            host = % host,
            port,
            "Scheduler registered.",
        );
        if *has_previous_scheduler_connection {
            tracing::info!(
                "Previous scheduler connection has been invalidated. Resending all ready-tasks in \
                 a background task."
            );
            let job_cache = self.inner.job_cache.clone();
            let shutdown = self.inner.shutdown.clone();
            tokio::spawn(async move {
                if let Err(e) = job_cache.resend_ready_tasks().await {
                    tracing::error!(
                        error = % e,
                        "Failed to resend ready-tasks after scheduler registration. Cancelling the \
                         service."
                    );
                    shutdown.shut_down_fatally();
                }
            });
        }
        *has_previous_scheduler_connection = true;
        drop(has_previous_scheduler_connection);
        Ok(scheduler_id)
    }

    /// Gets registered schedulers.
    ///
    /// # Returns
    ///
    /// The registered schedulers on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`SchedulerRegistrationManagement::get_schedulers`]'s return values on failure.
    pub async fn get_schedulers(&self) -> Result<Vec<RegisteredScheduler>, StorageServerError> {
        self.inner
            .db
            .get_schedulers()
            .await
            .map_err(StorageServerError::from)
    }

    /// Reads a job that is not in the cache from the database, and rejects it if it is a non-root
    /// job.
    ///
    /// # Returns
    ///
    /// The job's state on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`ExternalJobOrchestration::get_state_and_parent`]'s return values on failure.
    /// * Forwards [`ensure_root`]'s return values on failure.
    async fn uncached_root_state(&self, job_id: JobId) -> Result<JobState, StorageServerError> {
        let (state, parent_id) = self.inner.db.get_state_and_parent(job_id).await?;
        ensure_root(job_id, parent_id)?;
        Ok(state)
    }

    /// Repairs the tree rooted at `root_id` from its loaded jobs, in ascending ID order, as
    /// [`plan_repair`] plans it, in one fenced database transaction, and sets the repaired states
    /// in `tree_jobs`.
    ///
    /// # Returns
    ///
    /// The IDs of the repaired jobs, empty for a consistent tree, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`InternalJobOrchestration::has_failed_child`]'s return values on failure.
    /// * Forwards [`InternalJobOrchestration::apply_tree_transition`]'s return values on failure.
    async fn repair_tree(
        &self,
        root_id: JobId,
        tree_jobs: &mut [RecoverableJobContext],
    ) -> Result<Vec<JobId>, StorageServerError> {
        let rows: Vec<RecoveredRow> = tree_jobs.iter().map(RecoveredRow::from).collect();
        let mut failed_child = HashSet::new();
        for row in &rows {
            if row.state == JobState::BlockingOnChildren
                && self.inner.db.has_failed_child(row.id).await?
            {
                failed_child.insert(row.id);
            }
        }
        let states = rows.iter().map(|row| (row.id, row.state)).collect();
        let plan = plan_repair(
            &TreeStructure::from_rows(root_id, &rows),
            &states,
            &failed_child,
        );
        if plan.is_empty() {
            return Ok(Vec::new());
        }

        let repaired_states: HashMap<JobId, JobState> = plan
            .updates
            .iter()
            .map(|update| (update.job_id, update.to))
            .collect();
        let repaired_job_ids: Vec<JobId> =
            plan.updates.iter().map(|update| update.job_id).collect();
        tracing::error!(
            root_id = ? root_id,
            job_ids = ? repaired_job_ids,
            "Repairing job tree at startup."
        );
        self.inner
            .db
            .apply_tree_transition(TreeTransition {
                updates: plan.updates,
                child_inserts: Vec::new(),
            })
            .await?;
        for tree_job in tree_jobs {
            if let Some(state) = repaired_states.get(&tree_job.id) {
                tree_job.state = *state;
            }
        }
        Ok(repaired_job_ids)
    }

    /// Validates that the given `session_id` matches the session ID captured at service creation
    /// time.
    ///
    /// # Errors
    ///
    /// Returns [`StorageServerError::StaleSession`] if the session IDs don't match.
    fn validate_session(&self, session_id: SessionId) -> Result<(), StorageServerError> {
        if session_id != self.inner.session_id {
            return Err(StorageServerError::StaleSession(self.inner.session_id));
        }
        Ok(())
    }

    /// Checks a success report of a task of `jcb` against the limits, reserves its children
    /// against `jcb`'s per-job child limits, then decodes and builds every child, yielding to the
    /// runtime between children.
    ///
    /// # Returns
    ///
    /// A pair on success, containing:
    ///
    /// * The prepared children, in report order.
    /// * The children's reservation, which the caller releases if they are not stored.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`ReportRejection::OutputsTooLarge`] if the outputs exceed [`MAX_REPORT_OUTPUT_BYTES`].
    /// * [`ReportRejection::TooDeep`] if the children would be deeper than [`MAX_DEPTH`].
    /// * [`ReportRejection::TooManyChildren`] if there are more than [`MAX_CHILDREN_PER_JOB`]
    ///   children.
    /// * [`ReportRejection::ChildTooLarge`] if a child exceeds the database's statement payload.
    /// * [`ReportRejection::ChildrenTooLarge`] if the children exceed [`MAX_REPORT_CHILD_BYTES`].
    /// * [`ReportRejection::JobChildrenTooLarge`] if the children would take the job over a per-job
    ///   child limit.
    /// * [`ReportRejection::InvalidChild`] if [`PreparedChild::create`] fails for a child.
    async fn prepare_report(
        &self,
        jcb: &SharedJobControlBlock<StagingPublisher, TaskInstancePoolConnectorType>,
        task_outputs: &[TaskOutput],
        children: Vec<ChildJobSubmission>,
    ) -> Result<(Vec<PreparedChild>, ChildSetSummary), ReportRejection> {
        let depth = jcb.depth();
        let outputs_size: u64 = task_outputs.iter().map(|output| output.len() as u64).sum();
        if outputs_size > MAX_REPORT_OUTPUT_BYTES {
            return Err(ReportRejection::OutputsTooLarge {
                size: outputs_size,
                limit: MAX_REPORT_OUTPUT_BYTES,
            });
        }
        if children.is_empty() {
            return Ok((Vec::new(), ChildSetSummary { count: 0, bytes: 0 }));
        }
        if depth + 1 > MAX_DEPTH {
            return Err(ReportRejection::TooDeep {
                depth: depth + 1,
                limit: MAX_DEPTH,
            });
        }
        if children.len() > MAX_CHILDREN_PER_JOB {
            return Err(ReportRejection::TooManyChildren {
                count: children.len(),
                limit: MAX_CHILDREN_PER_JOB,
            });
        }
        let max_child_size = self.inner.db.max_statement_payload();
        if let Some((index, child)) = children
            .iter()
            .enumerate()
            .find(|(_, child)| child.size() > max_child_size)
        {
            return Err(ReportRejection::ChildTooLarge {
                index,
                size: child.size(),
                limit: max_child_size,
            });
        }
        let children_size: u64 = children.iter().map(ChildJobSubmission::size).sum();
        if children_size > MAX_REPORT_CHILD_BYTES {
            return Err(ReportRejection::ChildrenTooLarge {
                size: children_size,
                limit: MAX_REPORT_CHILD_BYTES,
            });
        }

        let reserved_children = ChildSetSummary {
            count: children.len(),
            bytes: children_size,
        };
        jcb.reserve_children(reserved_children).map_err(|totals| {
            ReportRejection::JobChildrenTooLarge {
                count: totals.count,
                bytes: totals.bytes,
                count_limit: MAX_CHILDREN_PER_JOB,
                bytes_limit: MAX_JOB_CHILD_BYTES,
            }
        })?;

        let mut prepared_children = Vec::with_capacity(children.len());
        for (index, child) in children.into_iter().enumerate() {
            tokio::task::yield_now().await;
            match PreparedChild::create(child).await {
                Ok(prepared_child) => prepared_children.push(prepared_child),
                Err(source) => {
                    jcb.release_children(reserved_children);
                    return Err(ReportRejection::InvalidChild { index, source });
                }
            }
        }
        Ok((prepared_children, reserved_children))
    }

    /// Runs a handler body as a task in the service's tracker, so that dropping the caller's future
    /// does not cut the body short, and waits for its result, raced against the service's
    /// cancellation token.
    ///
    /// # Type Parameters
    ///
    /// * `ReturnType` - The type of the value that the body returns on success.
    ///
    /// # Returns
    ///
    /// The body's result on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageServerError::Stopping`] if the service is shutting down.
    /// * [`InternalError::HandlerBodyFailed`] if the body panicked.
    /// * Forwards `body`'s return values on failure.
    async fn spawn_body<ReturnType: Send + 'static>(
        &self,
        body: impl Future<Output = Result<ReturnType, StorageServerError>> + Send + 'static,
    ) -> Result<ReturnType, StorageServerError> {
        let handle = self.inner.tracker.spawn(body);
        tokio::select! {
            result = handle => result.unwrap_or_else(|e| {
                Err(CacheError::Internal(InternalError::HandlerBodyFailed(e.to_string())).into())
            }),
            () = self.inner.shutdown.token().cancelled() => {
                Err(StorageServerError::Stopping("storage is shutting down".to_owned()))
            }
        }
    }
}

/// Inner data for [`ServiceState`], holding all storage services.
///
/// # Type Parameters
///
/// * `DbConnectorType` - The type of the DB-layer connector.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
struct ServiceStateInner<
    DbConnectorType: DbStorage + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
> {
    db: DbConnectorType,
    session_id: SessionId,
    job_cache: ServiceJobCache<TaskInstancePoolConnectorType>,
    publisher: StagingPublisher,
    inbound_queue_receiver: InboundQueueReceiverHandle,
    task_instance_pool_connector: TaskInstancePoolConnectorType,
    job_cache_gc_handle: JobCacheGcHandle,
    has_previous_scheduler_connection: tokio::sync::Mutex<bool>,
    shutdown: ShutdownHandle,
    tracker: TaskTracker,
    hooks: TestHooks,
}

/// Rejects a job that has a parent: the external job RPCs accept only root jobs.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`StorageServerError::BadRequest`] if `parent_id` is not [`None`].
fn ensure_root(job_id: JobId, parent_id: Option<JobId>) -> Result<(), StorageServerError> {
    if let Some(parent_id) = parent_id {
        return Err(StorageServerError::BadRequest(format!(
            "job {job_id} is a child job of job {parent_id}, not a root job"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use spider_core::compression::encode_zstd_bytes;
    use spider_core::job::JobState;
    use spider_core::task::DataTypeDescriptor;
    use spider_core::task::ExecutionPolicy;
    use spider_core::task::TaskDescriptor;
    use spider_core::task::TaskGraph as SubmittedTaskGraph;
    use spider_core::task::TdlContext;
    use spider_core::task::ValueTypeDescriptor;
    use spider_core::types::id::ExecutionManagerId;
    use spider_core::types::id::JobId;
    use spider_core::types::id::ResourceGroupId;
    use spider_core::types::io::TaskGraphInput;
    use spider_core::types::io::TaskGraphInputBuilder;
    use spider_core::types::io::TaskOutput;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::cache::error::StaleStateError;
    use crate::cache::job::SharedJobControlBlock;
    use crate::db::DbError;
    use crate::inbound_queue::InboundQueueSenderHandle;
    use crate::job_submission::compress_job_inputs;
    use crate::job_submission::compress_task_graph;
    use crate::job_submission::create_validated_submission;
    use crate::state::JobCacheGcHandle;
    use crate::state::StorageServerError;
    use crate::state::test_utils::MockDbConnector;
    use crate::state::test_utils::MockInboundQueueSender;
    use crate::state::test_utils::MockTaskInstancePoolConnector;

    type TestServiceState = ServiceState<MockDbConnector, MockTaskInstancePoolConnector>;

    const TEST_SESSION_ID: SessionId = 0;

    fn create_test_service() -> TestServiceState {
        create_test_service_with_db(MockDbConnector::default())
    }

    fn create_test_service_with_db(db: MockDbConnector) -> TestServiceState {
        create_test_service_with_db_and_session(db, TEST_SESSION_ID)
    }

    fn create_test_service_with_db_and_session(
        db: MockDbConnector,
        session_id: SessionId,
    ) -> TestServiceState {
        TestServiceState::new(ServiceStateParams {
            db,
            session_id,
            job_cache: JobCache::new(),
            inbound_queue_sender: MockInboundQueueSender,
            inbound_queue_receiver: create_inbound_queue_receiver(),
            task_instance_pool_connector: MockTaskInstancePoolConnector,
            job_cache_gc_handle: JobCacheGcHandle::new(tokio::sync::mpsc::unbounded_channel().0),
            shutdown: ShutdownHandle::new(CancellationToken::new()),
            hooks: TestHooks::default(),
        })
    }

    fn create_inbound_queue_receiver() -> InboundQueueReceiverHandle {
        use crate::inbound_queue::InboundQueueConfig;
        use crate::inbound_queue::create_inbound_queue;
        let (_sender, receiver) =
            create_inbound_queue(&InboundQueueConfig::default()).expect("inbound queue creation");
        receiver
    }

    /// Creates a [`ServiceState`] backed by [`InboundQueueReceiverHandle`].
    ///
    /// # Returns
    ///
    /// A tuple of the service state and the inbound queue sender handle on success.
    fn create_test_service_with_inbound_queue(
        db: MockDbConnector,
    ) -> (TestServiceState, InboundQueueSenderHandle) {
        use crate::inbound_queue::InboundQueueConfig;
        use crate::inbound_queue::create_inbound_queue;
        let (sender, receiver) =
            create_inbound_queue(&InboundQueueConfig::default()).expect("inbound queue creation");
        let service = TestServiceState::new(ServiceStateParams {
            db,
            session_id: 0,
            job_cache: JobCache::new(),
            inbound_queue_sender: sender.clone(),
            inbound_queue_receiver: receiver,
            task_instance_pool_connector: MockTaskInstancePoolConnector,
            job_cache_gc_handle: JobCacheGcHandle::new(tokio::sync::mpsc::unbounded_channel().0),
            shutdown: ShutdownHandle::new(CancellationToken::new()),
            hooks: TestHooks::default(),
        });
        (service, sender)
    }

    fn create_test_task_graph() -> SubmittedTaskGraph {
        let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
        let mut task_graph =
            SubmittedTaskGraph::new(None, None).expect("task graph creation should succeed");
        task_graph
            .insert_task(TaskDescriptor {
                tdl_context: TdlContext {
                    package: "test_pkg".to_owned(),
                    task_func: "test_fn".to_owned(),
                },
                execution_policy: Some(ExecutionPolicy::default()),
                inputs: vec![bytes_type.clone()],
                outputs: vec![bytes_type],
                input_sources: None,
            })
            .expect("task insertion should succeed");
        task_graph
    }

    /// # Returns
    ///
    /// A task graph input whose only positional input is the msgpack serialization of a 4-byte zero
    /// array.
    ///
    /// # Panics
    ///
    /// Panics if the input serialization fails.
    fn create_test_task_graph_input() -> TaskGraphInput {
        let mut builder = TaskGraphInputBuilder::new();
        builder
            .append_task_input(&[0u8; 4])
            .expect("task input appending should succeed");
        builder.build()
    }

    fn create_test_job_submission() -> (Vec<u8>, Vec<u8>) {
        let task_graph = create_test_task_graph();
        (
            compress_task_graph(&task_graph),
            compress_job_inputs(&create_test_task_graph_input()),
        )
    }

    fn create_test_serialized_outputs() -> Vec<u8> {
        let output_tuple = (1,);
        TaskOutputsSerializer::from_tuple(&output_tuple)
            .expect("output serialization should succeed")
    }

    fn create_empty_compressed_serialized_task_graph_input() -> Vec<u8> {
        compress_job_inputs(&TaskGraphInputBuilder::new().build())
    }

    type TestJcb = SharedJobControlBlock<StagingPublisher, MockTaskInstancePoolConnector>;

    /// Creates a ready job's tree in the service's cache, with the job's row seeded in the mock
    /// database.
    async fn create_test_jcb(service: &TestServiceState, job_id: JobId) -> TestJcb {
        let task_graph = create_test_task_graph();
        let job_submission =
            create_validated_submission(task_graph, create_test_task_graph_input());

        service.inner.db.states.insert(job_id, JobState::Ready);
        create_root_tree(
            &service.tree_env(),
            job_id,
            ResourceGroupId::random(),
            job_submission,
        )
        .await
        .expect("JCB creation should succeed")
    }

    #[tokio::test]
    async fn register_job_returns_job_id_and_inserts_into_cache() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        assert!(
            service.inner.job_cache.get(job_id).await.is_some(),
            "JCB should be in cache after register_job"
        );
        Ok(())
    }

    #[tokio::test]
    async fn register_job_returns_error_on_invalid_task_graph() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service
            .register_job(
                ResourceGroupId::random(),
                encode_zstd_bytes(b"invalid json")
                    .expect("invalid task graph compression should succeed"),
                create_empty_compressed_serialized_task_graph_input(),
            )
            .await;
        assert!(
            matches!(result, Err(StorageServerError::BadRequest(_))),
            "register_job should return a bad-request error on invalid task graph JSON"
        );
        Ok(())
    }

    #[tokio::test]
    async fn register_job_returns_error_on_input_size_mismatch() -> anyhow::Result<()> {
        let service = create_test_service();
        let task_graph = create_test_task_graph()
            .to_json()
            .expect("task graph serialization should succeed");
        let result = service
            .register_job(
                ResourceGroupId::random(),
                encode_zstd_bytes(task_graph.as_bytes())
                    .expect("task graph compression should succeed"),
                create_empty_compressed_serialized_task_graph_input(),
            )
            .await;
        assert!(
            matches!(result, Err(StorageServerError::BadRequest(_))),
            "register_job should return a bad-request error on input size mismatch"
        );
        Ok(())
    }

    #[tokio::test]
    async fn register_job_returns_error_on_empty_task_graph() -> anyhow::Result<()> {
        let service = create_test_service();
        let task_graph = SubmittedTaskGraph::new(None, None)
            .expect("empty task graph creation should succeed")
            .to_json()
            .expect("task graph serialization should succeed");
        let result = service
            .register_job(
                ResourceGroupId::random(),
                encode_zstd_bytes(task_graph.as_bytes())
                    .expect("task graph compression should succeed"),
                create_empty_compressed_serialized_task_graph_input(),
            )
            .await;
        assert!(
            matches!(result, Err(StorageServerError::BadRequest(_))),
            "register_job should return a bad-request error on empty task graph"
        );
        Ok(())
    }

    #[tokio::test]
    async fn start_job_starts_cached_job() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;

        service.start_job(job_id).await?;

        let state = service.get_job_state(job_id).await?;
        assert_eq!(state, JobState::Running);
        Ok(())
    }

    #[tokio::test]
    async fn start_job_twice_is_stale_and_not_fatal() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;

        service.start_job(job_id).await?;
        service
            .create_task_instance(
                TEST_SESSION_ID,
                job_id,
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await?;
        let result = service.start_job(job_id).await;
        assert!(
            matches!(
                result,
                Err(StorageServerError::Cache(CacheError::StaleState(
                    StaleStateError::JobAlreadyStarted
                )))
            ),
            "a second start should be stale, got {result:?}"
        );
        assert!(!service.shutdown().is_fatal());
        assert_eq!(service.get_job_state(job_id).await?, JobState::Running);
        Ok(())
    }

    #[tokio::test]
    async fn start_job_after_cancel_job_is_stale_and_not_fatal() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;

        assert_eq!(service.cancel_job(job_id).await?, JobState::Cancelled);
        let result = service.start_job(job_id).await;
        assert!(
            matches!(
                result,
                Err(StorageServerError::Cache(CacheError::StaleState(
                    StaleStateError::JobAlreadyStarted
                )))
            ),
            "a start after a cancel should be stale, got {result:?}"
        );
        assert!(!service.shutdown().is_fatal());
        assert_eq!(service.get_job_state(job_id).await?, JobState::Cancelled);
        Ok(())
    }

    #[tokio::test]
    async fn start_job_returns_job_not_found_when_not_in_cache() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service.start_job(JobId::random()).await;
        assert!(
            matches!(result, Err(StorageServerError::JobNotFound(_))),
            "start_job should return JobNotFound when job is not in cache"
        );
        Ok(())
    }

    #[tokio::test]
    async fn cancel_job_returns_job_not_found_if_not_exist() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service.cancel_job(JobId::random()).await;
        assert!(
            matches!(result, Err(StorageServerError::JobNotFound(_))),
            "cancel_job should return JobNotFound when job is not in cache or DB"
        );
        Ok(())
    }

    #[tokio::test]
    async fn cancel_job_returns_terminal_state_from_db_when_not_in_cache() -> anyhow::Result<()> {
        let db = MockDbConnector::default();
        let job_id = JobId::random();
        db.states.insert(job_id, JobState::Cancelled);

        let service = create_test_service_with_db(db);
        let state = service.cancel_job(job_id).await?;
        assert_eq!(
            state,
            JobState::Cancelled,
            "cancel_job should return Cancelled when job is already cancelled in DB"
        );
        Ok(())
    }

    #[tokio::test]
    async fn job_rpcs_reject_an_uncached_child_job() -> anyhow::Result<()> {
        for state in [JobState::Running, JobState::Succeeded, JobState::Failed] {
            let db = MockDbConnector::default();
            let job_id = JobId::random();
            db.states.insert(job_id, state);
            db.parents.insert(job_id, JobId::random());
            db.errors.insert(job_id, "child error".to_owned());
            db.outputs.insert(job_id, Vec::new());

            let service = create_test_service_with_db(db);
            assert!(
                matches!(
                    service.cancel_job(job_id).await,
                    Err(StorageServerError::BadRequest(_))
                ),
                "a cancel of a {state:?} child should be a bad request"
            );
            assert!(
                matches!(
                    service.get_job_state(job_id).await,
                    Err(StorageServerError::BadRequest(_))
                ),
                "the state of a {state:?} child should be a bad request"
            );
            assert!(
                matches!(
                    service.get_job_outputs(job_id).await,
                    Err(StorageServerError::BadRequest(_))
                ),
                "the outputs of a {state:?} child should be a bad request"
            );
            assert!(
                matches!(
                    service.get_job_error(job_id).await,
                    Err(StorageServerError::BadRequest(_))
                ),
                "the error of a {state:?} child should be a bad request"
            );
            assert!(
                matches!(
                    service.start_job(job_id).await,
                    Err(StorageServerError::JobNotFound(_))
                ),
                "a start of a {state:?} child should not find the job"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn cancel_job_transitions_to_terminal_state() -> anyhow::Result<()> {
        let service = create_test_service();
        let job_id = JobId::random();
        create_test_jcb(&service, job_id).await;

        let state = service.cancel_job(job_id).await?;
        assert!(
            matches!(state, JobState::Cancelled),
            "cancel should result in terminal state"
        );
        assert!(
            service.inner.job_cache.get(job_id).await.is_some(),
            "JCB should remain in cache after terminal cancel"
        );
        Ok(())
    }

    #[tokio::test]
    async fn get_job_state_serves_from_cache_when_jcb_present() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;

        let state = service.get_job_state(job_id).await?;
        assert_eq!(state, JobState::Ready);
        Ok(())
    }

    #[tokio::test]
    async fn get_job_state_falls_back_to_db_when_not_in_cache() -> anyhow::Result<()> {
        let db = MockDbConnector::default();
        let job_id = JobId::random();
        db.states.insert(job_id, JobState::Failed);

        let service = create_test_service_with_db(db);
        let state = service.get_job_state(job_id).await?;
        assert_eq!(state, JobState::Failed);
        Ok(())
    }

    #[tokio::test]
    async fn get_job_state_returns_error_for_unknown_job() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service.get_job_state(JobId::random()).await;
        assert!(result.is_err(), "get_job_state should fail for unknown job");
        Ok(())
    }

    #[tokio::test]
    async fn get_job_outputs_returns_outputs_from_db() -> anyhow::Result<()> {
        let db = MockDbConnector::default();
        let job_id = JobId::random();
        let outputs: Vec<TaskOutput> = vec![vec![1, 2, 3]];
        db.states.insert(job_id, JobState::Succeeded);
        db.outputs.insert(job_id, outputs.clone());

        let service = create_test_service_with_db(db);
        let result = service.get_job_outputs(job_id).await?;
        assert_eq!(result, outputs);
        Ok(())
    }

    #[tokio::test]
    async fn get_job_outputs_returns_outputs_from_cache_when_jcb_present() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        service.start_job(job_id).await?;

        let context = service
            .create_task_instance(
                TEST_SESSION_ID,
                job_id,
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await?;
        let serialized_outputs = create_test_serialized_outputs();
        let expected = TaskOutputsSerializer::deserialize(&serialized_outputs)
            .expect("test serialized outputs should deserialize successfully");
        service
            .succeed_task_instance(
                TEST_SESSION_ID,
                job_id,
                context.task_instance_id,
                0,
                serialized_outputs,
                Vec::new(),
            )
            .await?;

        let actual = service.get_job_outputs(job_id).await?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn get_job_outputs_returns_error_for_unknown_job() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service.get_job_outputs(JobId::random()).await;
        assert!(
            result.is_err(),
            "get_job_outputs should fail for unknown job"
        );
        Ok(())
    }

    #[tokio::test]
    async fn get_job_outputs_returns_error_when_job_not_succeeded() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        // JCB is in cache but job is still Ready (not Succeeded).
        let result = service.get_job_outputs(job_id).await;
        assert!(
            result.is_err(),
            "get_job_outputs should fail when job is not Succeeded"
        );
        Ok(())
    }

    #[tokio::test]
    async fn get_job_error_returns_error_message_from_db() -> anyhow::Result<()> {
        let db = MockDbConnector::default();
        let job_id = JobId::random();
        let error_msg = "something went wrong".to_owned();
        db.states.insert(job_id, JobState::Failed);
        db.errors.insert(job_id, error_msg.clone());

        let service = create_test_service_with_db(db);
        let result = service.get_job_error(job_id).await?;
        assert_eq!(result, error_msg);
        Ok(())
    }

    #[tokio::test]
    async fn get_job_error_returns_error_for_unknown_job() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service.get_job_error(JobId::random()).await;
        assert!(
            matches!(result, Err(StorageServerError::Db(DbError::JobNotFound(_)))),
            "get_job_error should fail for unknown job"
        );
        Ok(())
    }

    #[tokio::test]
    async fn create_task_instance_returns_execution_context() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        service.start_job(job_id).await?;

        let context = service
            .create_task_instance(
                TEST_SESSION_ID,
                job_id,
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await?;
        assert_eq!(
            context.task_instance_id, 1,
            "task instance ID should match mock pool counter"
        );
        Ok(())
    }

    #[tokio::test]
    async fn create_task_instance_returns_job_not_found_when_not_in_cache() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service
            .create_task_instance(
                TEST_SESSION_ID,
                JobId::random(),
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await;
        assert!(
            matches!(result, Err(StorageServerError::JobNotFound(_))),
            "create_task_instance should return JobNotFound when job is not in cache"
        );
        Ok(())
    }

    #[tokio::test]
    async fn succeed_task_instance_transitions_job_to_succeeded() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        service.start_job(job_id).await?;

        let context = service
            .create_task_instance(
                TEST_SESSION_ID,
                job_id,
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await?;
        let state = service
            .succeed_task_instance(
                TEST_SESSION_ID,
                job_id,
                context.task_instance_id,
                0,
                create_test_serialized_outputs(),
                Vec::new(),
            )
            .await?;
        assert_eq!(state, JobState::Succeeded);
        assert!(
            service.inner.job_cache.get(job_id).await.is_some(),
            "JCB should remain in cache after terminal succeed"
        );
        Ok(())
    }

    #[tokio::test]
    async fn succeed_task_instance_returns_job_not_found_when_not_in_cache() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service
            .succeed_task_instance(
                TEST_SESSION_ID,
                JobId::random(),
                1,
                0,
                create_test_serialized_outputs(),
                Vec::new(),
            )
            .await;
        assert!(
            matches!(result, Err(StorageServerError::JobNotFound(_))),
            "succeed_task_instance should return JobNotFound when job is not in cache"
        );
        Ok(())
    }

    #[tokio::test]
    async fn fail_task_instance_transitions_job_to_failed() -> anyhow::Result<()> {
        let service = create_test_service();
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        service.start_job(job_id).await?;

        let context = service
            .create_task_instance(
                TEST_SESSION_ID,
                job_id,
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await?;
        let state = service
            .fail_task_instance(
                TEST_SESSION_ID,
                job_id,
                context.task_instance_id,
                TaskId::Index(0),
                "test failure".to_owned(),
            )
            .await?;
        assert_eq!(state, JobState::Failed);
        assert!(
            service.inner.job_cache.get(job_id).await.is_some(),
            "JCB should remain in cache after terminal fail"
        );
        Ok(())
    }

    #[tokio::test]
    async fn fail_task_instance_returns_job_not_found_when_not_in_cache() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service
            .fail_task_instance(
                TEST_SESSION_ID,
                JobId::random(),
                1,
                TaskId::Index(0),
                "error".to_owned(),
            )
            .await;
        assert!(
            matches!(result, Err(StorageServerError::JobNotFound(_))),
            "fail_task_instance should return JobNotFound when job is not in cache"
        );
        Ok(())
    }

    #[tokio::test]
    async fn cancel_job_enqueues_terminal_job_for_cache_gc() -> anyhow::Result<()> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let service = TestServiceState::new(ServiceStateParams {
            db: MockDbConnector::default(),
            session_id: TEST_SESSION_ID,
            job_cache: JobCache::new(),
            inbound_queue_sender: MockInboundQueueSender,
            inbound_queue_receiver: create_inbound_queue_receiver(),
            task_instance_pool_connector: MockTaskInstancePoolConnector,
            job_cache_gc_handle: JobCacheGcHandle::new(sender),
            shutdown: ShutdownHandle::new(CancellationToken::new()),
            hooks: TestHooks::default(),
        });
        let job_id = JobId::random();
        create_test_jcb(&service, job_id).await;

        service.cancel_job(job_id).await?;

        assert_eq!(
            receiver.try_recv(),
            Ok(job_id),
            "cancelled job should be enqueued for cache GC"
        );
        Ok(())
    }

    #[tokio::test]
    async fn succeed_task_instance_enqueues_terminal_job_for_cache_gc() -> anyhow::Result<()> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let service = TestServiceState::new(ServiceStateParams {
            db: MockDbConnector::default(),
            session_id: TEST_SESSION_ID,
            job_cache: JobCache::new(),
            inbound_queue_sender: MockInboundQueueSender,
            inbound_queue_receiver: create_inbound_queue_receiver(),
            task_instance_pool_connector: MockTaskInstancePoolConnector,
            job_cache_gc_handle: JobCacheGcHandle::new(sender),
            shutdown: ShutdownHandle::new(CancellationToken::new()),
            hooks: TestHooks::default(),
        });
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        service.start_job(job_id).await?;
        let context = service
            .create_task_instance(
                TEST_SESSION_ID,
                job_id,
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await?;

        service
            .succeed_task_instance(
                TEST_SESSION_ID,
                job_id,
                context.task_instance_id,
                0,
                create_test_serialized_outputs(),
                Vec::new(),
            )
            .await?;

        assert_eq!(
            receiver.try_recv(),
            Ok(job_id),
            "succeeded job should be enqueued for cache GC"
        );
        Ok(())
    }

    #[tokio::test]
    async fn fail_task_instance_enqueues_terminal_job_for_cache_gc() -> anyhow::Result<()> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let service = TestServiceState::new(ServiceStateParams {
            db: MockDbConnector::default(),
            session_id: TEST_SESSION_ID,
            job_cache: JobCache::new(),
            inbound_queue_sender: MockInboundQueueSender,
            inbound_queue_receiver: create_inbound_queue_receiver(),
            task_instance_pool_connector: MockTaskInstancePoolConnector,
            job_cache_gc_handle: JobCacheGcHandle::new(sender),
            shutdown: ShutdownHandle::new(CancellationToken::new()),
            hooks: TestHooks::default(),
        });
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;
        service.start_job(job_id).await?;
        let context = service
            .create_task_instance(
                TEST_SESSION_ID,
                job_id,
                TaskId::Index(0),
                ExecutionManagerId::random(),
            )
            .await?;

        service
            .fail_task_instance(
                TEST_SESSION_ID,
                job_id,
                context.task_instance_id,
                TaskId::Index(0),
                "test failure".to_owned(),
            )
            .await?;

        assert_eq!(
            receiver.recv().await,
            Some(job_id),
            "failed job should be enqueued for cache GC"
        );
        Ok(())
    }

    #[tokio::test]
    async fn task_instance_orchestration_return_stale_session_on_mismatch() -> anyhow::Result<()> {
        // Create a service with a higher session ID to simulate a server restart.
        const CURRENT_SESSION_ID: SessionId = 10;
        const STALE_SESSION_ID: SessionId = CURRENT_SESSION_ID - 1;
        const TASK_INDEX: TaskIndex = 0;
        const TASK_INSTANCE_ID: TaskInstanceId = 1;

        let db = MockDbConnector::default();
        let service = create_test_service_with_db_and_session(db, CURRENT_SESSION_ID);

        // Register a job so the JCB is in cache.
        let (compressed_serialized_task_graph, compressed_serialized_task_graph_input) =
            create_test_job_submission();
        let job_id = service
            .register_job(
                ResourceGroupId::random(),
                compressed_serialized_task_graph,
                compressed_serialized_task_graph_input,
            )
            .await?;

        {
            let result = service
                .create_task_instance(
                    STALE_SESSION_ID,
                    job_id,
                    TaskId::Index(TASK_INDEX),
                    ExecutionManagerId::random(),
                )
                .await;
            assert!(
                matches!(result, Err(StorageServerError::StaleSession(_))),
                "create_task_instance should return StaleSession on session mismatch"
            );
        }

        {
            let result = service
                .succeed_task_instance(
                    STALE_SESSION_ID,
                    job_id,
                    TASK_INSTANCE_ID,
                    TASK_INDEX,
                    create_test_serialized_outputs(),
                    Vec::new(),
                )
                .await;
            assert!(
                matches!(result, Err(StorageServerError::StaleSession(_))),
                "succeed_task_instance should return StaleSession on session mismatch"
            );
        }

        {
            let result = service
                .fail_task_instance(
                    STALE_SESSION_ID,
                    job_id,
                    TASK_INSTANCE_ID,
                    TaskId::Index(TASK_INDEX),
                    "error".to_owned(),
                )
                .await;
            assert!(
                matches!(result, Err(StorageServerError::StaleSession(_))),
                "fail_task_instance should return StaleSession on session mismatch"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn add_resource_group_returns_id() -> anyhow::Result<()> {
        let service = create_test_service();
        assert!(
            service
                .add_resource_group(ExternalResourceGroupCredentials::new(
                    "external_123".to_owned(),
                    vec![1, 2, 3],
                ))
                .await
                .is_ok()
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_add_or_verify_resource_group_creates_and_reuses_id() -> anyhow::Result<()> {
        let db = MockDbConnector::default();
        let service = create_test_service_with_db(db.clone());
        let credentials =
            ExternalResourceGroupCredentials::new("external_123".to_owned(), vec![1, 2, 3]);
        let rg_id = service
            .add_or_verify_resource_group(credentials.clone())
            .await?;
        service
            .verify_resource_group(rg_id, credentials.get_password())
            .await?;
        assert_eq!(
            service.add_or_verify_resource_group(credentials).await?,
            rg_id
        );
        assert_eq!(db.resource_groups.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_add_or_verify_resource_group_rejects_wrong_password() -> anyhow::Result<()> {
        let db = MockDbConnector::default();
        let service = create_test_service_with_db(db.clone());
        let credentials =
            ExternalResourceGroupCredentials::new("external_123".to_owned(), vec![1, 2, 3]);
        let rg_id = service.add_resource_group(credentials.clone()).await?;
        let result = service
            .add_or_verify_resource_group(ExternalResourceGroupCredentials::new(
                credentials.get_external_resource_group_id().to_owned(),
                vec![4, 5, 6],
            ))
            .await;
        assert!(matches!(
            result,
            Err(StorageServerError::Db(DbError::InvalidPassword(id))) if id == rg_id
        ));
        assert_eq!(
            service.add_or_verify_resource_group(credentials).await?,
            rg_id
        );
        assert_eq!(db.resource_groups.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn verify_resource_group_succeeds_for_correct_password() -> anyhow::Result<()> {
        let service = create_test_service();
        let password = vec![1, 2, 3];
        let rg_id = service
            .add_resource_group(ExternalResourceGroupCredentials::new(
                "external_123".to_owned(),
                password.clone(),
            ))
            .await?;
        service.verify_resource_group(rg_id, &password).await?;
        Ok(())
    }

    #[tokio::test]
    async fn verify_resource_group_fails_for_wrong_password() -> anyhow::Result<()> {
        let service = create_test_service();
        let rg_id = service
            .add_resource_group(ExternalResourceGroupCredentials::new(
                "external_123".to_owned(),
                vec![1, 2, 3],
            ))
            .await?;
        let result = service.verify_resource_group(rg_id, &[4, 5, 6]).await;
        assert!(result.is_err(), "verify should fail for wrong password");
        Ok(())
    }

    #[tokio::test]
    async fn poll_ready_tasks_returns_entries_from_inbound_queue() -> anyhow::Result<()> {
        const TASK_INDEX: TaskIndex = 0;
        let (service, sender) = create_test_service_with_inbound_queue(MockDbConnector::default());
        let rg_id = ResourceGroupId::random();
        let job_id = JobId::random();
        sender
            .send_task_ready(rg_id, job_id, vec![TASK_INDEX])
            .await
            .expect("send_task_ready should succeed");

        let entries = service
            .poll_ready_tasks(10, Duration::from_millis(100))
            .await?;
        assert_eq!(entries.len(), 1, "should receive one inbound queue entry");
        assert_eq!(entries[0].job_id, job_id);
        assert_eq!(entries[0].task_kind, TASK_INDEX);
        assert_eq!(entries[0].resource_group_id, rg_id);
        Ok(())
    }

    #[tokio::test]
    async fn poll_ready_tasks_returns_empty_when_no_tasks() -> anyhow::Result<()> {
        let (service, _sender) = create_test_service_with_inbound_queue(MockDbConnector::default());
        let entries = service
            .poll_ready_tasks(10, Duration::from_millis(10))
            .await?;
        assert!(
            entries.is_empty(),
            "should receive no entries from empty inbound queue"
        );
        Ok(())
    }

    #[tokio::test]
    async fn poll_commit_ready_tasks_returns_entries_from_inbound_queue() -> anyhow::Result<()> {
        let (service, sender) = create_test_service_with_inbound_queue(MockDbConnector::default());
        let rg_id = ResourceGroupId::random();
        let job_id = JobId::random();
        sender
            .send_commit_ready(rg_id, job_id)
            .await
            .expect("send_commit_ready should succeed");

        let entries = service
            .poll_commit_ready_tasks(10, Duration::from_millis(100))
            .await?;
        assert_eq!(entries.len(), 1, "should receive one commit-ready entry");
        assert_eq!(entries[0].job_id, job_id);
        assert_eq!(entries[0].task_kind, CommitTaskMarker);
        assert_eq!(entries[0].resource_group_id, rg_id);
        Ok(())
    }

    #[tokio::test]
    async fn poll_cleanup_ready_tasks_returns_entries_from_inbound_queue() -> anyhow::Result<()> {
        let (service, sender) = create_test_service_with_inbound_queue(MockDbConnector::default());
        let rg_id = ResourceGroupId::random();
        let job_id = JobId::random();
        sender
            .send_cleanup_ready(rg_id, job_id)
            .await
            .expect("send_cleanup_ready should succeed");

        let entries = service
            .poll_cleanup_ready_tasks(10, Duration::from_millis(100))
            .await?;
        assert_eq!(entries.len(), 1, "should receive one cleanup-ready entry");
        assert_eq!(entries[0].job_id, job_id);
        assert_eq!(entries[0].task_kind, CleanupTaskMarker);
        assert_eq!(entries[0].resource_group_id, rg_id);
        Ok(())
    }

    #[tokio::test]
    async fn register_execution_manager_returns_id() -> anyhow::Result<()> {
        let service = create_test_service();
        let (_, resource_group_id) = service
            .register_execution_manager("127.0.0.1".parse()?, None)
            .await?;
        assert_eq!(resource_group_id, None);
        Ok(())
    }

    #[tokio::test]
    async fn register_execution_manager_resolves_resource_group() -> anyhow::Result<()> {
        let service = create_test_service();
        let password = b"password";
        let resource_group_id = service
            .add_resource_group(ExternalResourceGroupCredentials::new(
                "external_123".to_owned(),
                password.to_vec(),
            ))
            .await?;

        let (_, registered_resource_group_id) = service
            .register_execution_manager(
                "127.0.0.1".parse()?,
                Some(ExternalResourceGroupCredentials::new(
                    "external_123".to_owned(),
                    password.to_vec(),
                )),
            )
            .await?;

        assert_eq!(registered_resource_group_id, Some(resource_group_id));
        Ok(())
    }

    #[tokio::test]
    async fn register_execution_manager_rejects_unknown_resource_group() -> anyhow::Result<()> {
        let service = create_test_service();

        let result = service
            .register_execution_manager(
                "127.0.0.1".parse()?,
                Some(ExternalResourceGroupCredentials::new(
                    "unknown".to_owned(),
                    b"password".to_vec(),
                )),
            )
            .await;

        assert!(matches!(
            result,
            Err(StorageServerError::Db(
                DbError::ExternalResourceGroupNotFound(_)
            ))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn register_execution_manager_rejects_invalid_password() -> anyhow::Result<()> {
        let service = create_test_service();
        service
            .add_resource_group(ExternalResourceGroupCredentials::new(
                "external_123".to_owned(),
                b"password".to_vec(),
            ))
            .await?;

        let result = service
            .register_execution_manager(
                "127.0.0.1".parse()?,
                Some(ExternalResourceGroupCredentials::new(
                    "external_123".to_owned(),
                    b"wrong-password".to_vec(),
                )),
            )
            .await;

        assert!(matches!(
            result,
            Err(StorageServerError::Db(DbError::InvalidPassword(_)))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn register_scheduler_resends_ready_tasks_only_when_replacing_previous_scheduler()
    -> anyhow::Result<()> {
        let (service, _sender) = create_test_service_with_inbound_queue(MockDbConnector::default());

        let (task_graph, inputs) = create_test_job_submission();
        let job_id = service
            .register_job(ResourceGroupId::random(), task_graph, inputs)
            .await?;
        service.start_job(job_id).await?;

        // Starting the job enqueues its initial ready task; drain it so the queue is empty before
        // probing whether a registration triggers a resend.
        let initial = service
            .poll_ready_tasks(10, Duration::from_millis(100))
            .await?;
        assert_eq!(
            initial.len(),
            1,
            "starting the job should enqueue its initial ready task"
        );

        // The first registration has no previous scheduler to replace, so it must not resend.
        service
            .register_scheduler(
                Host::new("scheduler-a.example.com".to_owned())
                    .expect("scheduler host should not be empty"),
                8080,
            )
            .await?;
        tokio::task::yield_now().await;
        let after_first = service
            .poll_ready_tasks(10, Duration::from_millis(100))
            .await?;
        assert!(
            after_first.is_empty(),
            "the first scheduler registration must not resend ready tasks"
        );

        // The second registration replaces the first scheduler, so it must resend ready tasks. The
        // resend runs in a spawned background task, so yield to let it run before polling.
        service
            .register_scheduler(
                Host::new("scheduler-b.example.com".to_owned())
                    .expect("scheduler host should not be empty"),
                8081,
            )
            .await?;
        tokio::task::yield_now().await;
        let after_second = service
            .poll_ready_tasks(10, Duration::from_millis(100))
            .await?;
        assert_eq!(
            after_second.len(),
            1,
            "the second scheduler registration must resend the job's ready tasks"
        );
        assert_eq!(after_second[0].job_id, job_id);
        Ok(())
    }

    #[tokio::test]
    async fn update_execution_manager_heartbeat_succeeds_for_registered_em() -> anyhow::Result<()> {
        let service = create_test_service();
        let em_id = service
            .register_execution_manager("127.0.0.1".parse()?, None)
            .await?
            .0;
        service.update_execution_manager_heartbeat(em_id).await?;
        Ok(())
    }

    #[tokio::test]
    async fn update_execution_manager_heartbeat_fails_for_unknown_em() -> anyhow::Result<()> {
        let service = create_test_service();
        let result = service
            .update_execution_manager_heartbeat(ExecutionManagerId::random())
            .await;

        assert!(
            matches!(
                result,
                Err(StorageServerError::Db(DbError::IllegalExecutionManagerId(
                    _
                )))
            ),
            "update heartbeat should fail for unregistered execution manager"
        );
        Ok(())
    }
}
