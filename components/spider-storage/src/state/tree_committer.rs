//! The per-tree committer: one long-running task per job tree that is the tree's only writer.
//!
//! The committer receives the tree's job-level events, plans each round of them, writes the round
//! in one fenced database transaction, applies it to the cached jobs, and replies to every event.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;

use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use tokio::sync::mpsc;
use tokio_util::task::TaskTracker;

use crate::cache::error::CacheError;
use crate::cache::error::InternalError;
use crate::cache::job::RegularDoneInputs;
use crate::cache::job::SharedJobControlBlock;
use crate::cache::task::PreparedChild;
use crate::cache::tree::ChildSetSummary;
use crate::cache::tree::Lane;
use crate::cache::tree::Pending;
use crate::cache::tree::Plan;
use crate::cache::tree::PlannedRegistration;
use crate::cache::tree::PlannerEvent;
use crate::cache::tree::PlannerInput;
use crate::cache::tree::Publication;
use crate::cache::tree::RecoveredRow;
use crate::cache::tree::TreeControlBlock;
use crate::cache::tree::TreeEventKind;
use crate::cache::tree::TreeNode;
use crate::cache::tree::TreeOutcome;
use crate::cache::tree::TreeReply;
use crate::cache::tree::TreeStructure;
#[cfg(any(test, debug_assertions, feature = "test-hooks"))]
use crate::cache::tree::check_invariants;
use crate::cache::tree::plan_events;
use crate::db::ChildJobInsert;
use crate::db::DbError;
use crate::db::InternalJobOrchestration;
use crate::db::JobStateUpdate;
use crate::db::RecoverableJobContext;
use crate::db::TreeTransition;
use crate::inbound_queue::InboundQueueSender;
use crate::job_submission::ValidatedJobSubmission;
use crate::limits::MAX_EVENTS_PER_ROUND;
use crate::state::JobCache;
use crate::state::JobCacheGcHandle;
use crate::state::ShutdownHandle;
use crate::state::StorageServerError;
use crate::task_instance_pool::TaskInstancePoolConnector;
use crate::test_hooks::HookContext;
use crate::test_hooks::HookPoint;
use crate::test_hooks::TestHooks;

/// Everything a tree's committer uses besides its tree.
///
/// # Type Parameters
///
/// * `InboundQueueSenderType` - The type of the inbound queue sender that publishes ready work.
/// * `DbConnectorType` - The type of the DB-layer connector.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
#[derive(Clone)]
pub struct TreeEnv<
    InboundQueueSenderType: InboundQueueSender,
    DbConnectorType: InternalJobOrchestration,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> {
    pub db: DbConnectorType,
    pub job_cache: JobCache<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    pub gc: JobCacheGcHandle,
    pub publisher: InboundQueueSenderType,
    pub task_instance_pool: TaskInstancePoolConnectorType,
    pub shutdown: ShutdownHandle,
    pub tracker: TaskTracker,
    pub hooks: TestHooks,
}

/// Creates a new root job's tree: its control block, its cached job, and its committer.
///
/// # Returns
///
/// The root's control block on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`SharedJobControlBlock::create`]'s return values on failure.
/// * Forwards [`JobCache::insert`]'s return values on failure.
pub async fn create_root_tree<
    InboundQueueSenderType: InboundQueueSender + 'static,
    DbConnectorType: InternalJobOrchestration + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
>(
    env: &TreeEnv<InboundQueueSenderType, DbConnectorType, TaskInstancePoolConnectorType>,
    job_id: JobId,
    owner_id: ResourceGroupId,
    job_submission: ValidatedJobSubmission,
) -> Result<
    SharedJobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    StorageServerError,
> {
    let (tree, receiver) = new_tree_control_block(env, job_id);
    let jcb = SharedJobControlBlock::create(
        job_id,
        owner_id,
        job_submission,
        env.publisher.clone(),
        env.task_instance_pool.clone(),
        tree,
    )
    .await?;
    env.job_cache.insert(jcb.clone()).await?;
    spawn_committer(env, CommitterState::new_root(jcb.clone()), receiver);
    Ok(jcb)
}

/// Recovers the tree rooted at `root_id` from its jobs loaded from the database.
///
/// The jobs are given in ascending ID order with their repaired states. The tree's non-terminal
/// jobs get control blocks and are cached, and the tree gets a committer unless every job is
/// terminal.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`SharedJobControlBlock::recover`]'s return values on failure.
/// * Forwards [`JobCache::insert_many`]'s return values on failure.
pub async fn create_recovered_tree<
    InboundQueueSenderType: InboundQueueSender + 'static,
    DbConnectorType: InternalJobOrchestration + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
>(
    env: &TreeEnv<InboundQueueSenderType, DbConnectorType, TaskInstancePoolConnectorType>,
    root_id: JobId,
    recoverable_job_contexts: Vec<RecoverableJobContext>,
) -> Result<(), StorageServerError> {
    let rows: Vec<RecoveredRow> = recoverable_job_contexts
        .iter()
        .map(RecoveredRow::from)
        .collect();
    let structure = TreeStructure::from_rows(root_id, &rows);
    let (tree, receiver) = new_tree_control_block(env, root_id);
    let mut tree_jcbs = HashMap::with_capacity(rows.len());
    let mut jcbs = Vec::with_capacity(rows.len());
    for recoverable_job_context in recoverable_job_contexts {
        let Some(node) = structure.node(recoverable_job_context.id) else {
            continue;
        };
        let jcb = SharedJobControlBlock::recover(
            recoverable_job_context,
            node.depth,
            env.publisher.clone(),
            env.task_instance_pool.clone(),
            tree.clone(),
        )
        .await?;
        tree_jcbs.insert(jcb.id(), jcb.clone());
        jcbs.push(jcb);
    }
    env.job_cache.insert_many(jcbs).await?;
    if !structure.is_quiescent() {
        spawn_committer(
            env,
            CommitterState {
                structure,
                jcbs: tree_jcbs,
            },
            receiver,
        );
    }
    Ok(())
}

/// The state that a tree's committer owns: the tree's structure and its non-terminal jobs.
struct CommitterState<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> {
    structure: TreeStructure,
    jcbs: HashMap<
        JobId,
        SharedJobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    >,
}

impl<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
> CommitterState<InboundQueueSenderType, TaskInstancePoolConnectorType>
{
    /// Factory function.
    ///
    /// # Returns
    ///
    /// The state of the one-job tree rooted at `root`.
    fn new_root(
        root: SharedJobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    ) -> Self {
        let root_id = root.id();
        let structure = TreeStructure::new_root(
            root_id,
            TreeNode {
                parent: None,
                depth: 0,
                resource_group_id: root.owner_id(),
                has_commit: root.has_commit_task(),
                has_cleanup: root.has_cleanup_task(),
                active_children: 0,
            },
        );
        Self {
            structure,
            jcbs: HashMap::from([(root_id, root)]),
        }
    }

    /// Writes an applied plan's counters into the tree's structure, and drops the jobs that became
    /// terminal.
    ///
    /// # Panics
    ///
    /// Panics if a counter belongs to a job that is not in the tree.
    fn update_structure(&mut self, plan: &Plan) {
        for (job_id, active_children) in &plan.counters {
            self.structure
                .set_active_children(*job_id, *active_children);
        }
        for job_id in &plan.terminated {
            self.structure.remove(*job_id);
            self.jcbs.remove(job_id);
        }
    }

    /// Adds the children that `registration` registered under the IDs `child_ids`, one per
    /// prepared child in the same order, to the tree, and adds their ready tasks to
    /// `publications`.
    ///
    /// # Type Parameters
    ///
    /// * `DbConnectorType` - The type of the DB-layer connector of `env`.
    ///
    /// # Returns
    ///
    /// The control blocks of the children on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`Self::jcb`]'s return values on failure.
    ///
    /// # Panics
    ///
    /// Panics if the number of IDs differs from the number of prepared children.
    fn bind_children<DbConnectorType: InternalJobOrchestration>(
        &mut self,
        registration: &PlannedRegistration,
        child_ids: Vec<JobId>,
        prepared_children: Vec<PreparedChild>,
        env: &TreeEnv<InboundQueueSenderType, DbConnectorType, TaskInstancePoolConnectorType>,
        publications: &mut Vec<Publication>,
    ) -> Result<
        Vec<SharedJobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>>,
        CacheError,
    > {
        assert_eq!(
            child_ids.len(),
            prepared_children.len(),
            "every registered child should get one ID"
        );
        let parent = self.jcb(registration.parent_id)?.clone();
        let mut children = Vec::with_capacity(child_ids.len());
        for (child_id, mut prepared_child) in child_ids.into_iter().zip(prepared_children) {
            let task_indices = std::mem::take(&mut prepared_child.ready_task_indices);
            let child = SharedJobControlBlock::from_prepared_child(
                child_id,
                &parent,
                prepared_child,
                env.publisher.clone(),
                env.task_instance_pool.clone(),
            );
            self.structure.insert(
                child_id,
                TreeNode {
                    parent: Some(registration.parent_id),
                    depth: registration.child_depth,
                    resource_group_id: registration.resource_group_id,
                    has_commit: child.has_commit_task(),
                    has_cleanup: child.has_cleanup_task(),
                    active_children: 0,
                },
            );
            self.jcbs.insert(child_id, child.clone());
            publications.push(Publication::Ready {
                job_id: child_id,
                task_indices,
            });
            children.push(child);
        }
        Ok(children)
    }

    /// # Returns
    ///
    /// The control block of the tree's non-terminal job `job_id` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::JobNotFound`] if the job is not a non-terminal job of the tree.
    fn jcb(
        &self,
        job_id: JobId,
    ) -> Result<
        &SharedJobControlBlock<InboundQueueSenderType, TaskInstancePoolConnectorType>,
        CacheError,
    > {
        self.jcbs
            .get(&job_id)
            .ok_or_else(|| InternalError::JobNotFound(job_id).into())
    }
}

/// An event in the committer's buffer, with its input once collected, and the children that its
/// job's tasks prepared once taken.
struct Queued {
    pending: Pending,
    input: Option<PlannerInput>,
    children: Vec<PreparedChild>,
}

impl Queued {
    const fn new(pending: Pending) -> Self {
        Self {
            pending,
            input: None,
            children: Vec::new(),
        }
    }
}

/// Owns the tree's event receiver. Unless disarmed, dropping it marks the service's shutdown as
/// fatal, so that every exit of the committer other than a quiescent one or a shutdown poisons the
/// service.
struct PoisonOnAbnormalExit {
    receiver: mpsc::UnboundedReceiver<Pending>,
    shutdown: ShutdownHandle,
    root_id: JobId,
    disarmed: bool,
}

impl PoisonOnAbnormalExit {
    /// Disarms the guard, then marks the service's shutdown as fatal.
    fn poison(&mut self) {
        self.disarmed = true;
        tracing::error!(
            root_id = ? self.root_id,
            "Tree committer exited abnormally. Shutting down the service."
        );
        self.shutdown.shut_down_fatally();
    }
}

impl Drop for PoisonOnAbnormalExit {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        if !std::thread::panicking() && self.shutdown.token().is_cancelled() {
            tracing::debug!(
                root_id = ? self.root_id,
                "Tree committer dropped during the runtime's teardown."
            );
            return;
        }
        self.poison();
    }
}

/// # Returns
///
/// A new tree control block for the tree rooted at `root_id`, and its event receiver.
fn new_tree_control_block<
    InboundQueueSenderType: InboundQueueSender,
    DbConnectorType: InternalJobOrchestration,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
>(
    env: &TreeEnv<InboundQueueSenderType, DbConnectorType, TaskInstancePoolConnectorType>,
    root_id: JobId,
) -> (TreeControlBlock, mpsc::UnboundedReceiver<Pending>) {
    TreeControlBlock::new(root_id, env.shutdown.token().clone(), env.hooks.clone())
}

/// Spawns the committer of the tree whose committer state is `state` into the environment's
/// tracker.
fn spawn_committer<
    InboundQueueSenderType: InboundQueueSender + 'static,
    DbConnectorType: InternalJobOrchestration + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
>(
    env: &TreeEnv<InboundQueueSenderType, DbConnectorType, TaskInstancePoolConnectorType>,
    state: CommitterState<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    receiver: mpsc::UnboundedReceiver<Pending>,
) {
    env.tracker
        .spawn(run_committer(state, receiver, env.clone()));
}

/// Runs a tree's committer until the tree is quiescent or the service shuts down.
///
/// Any other exit, an error or a panic, poisons the service through [`PoisonOnAbnormalExit`]: the
/// fatal flag is set, the service token is cancelled, and every waiting event's reply channel is
/// dropped.
async fn run_committer<
    InboundQueueSenderType: InboundQueueSender + 'static,
    DbConnectorType: InternalJobOrchestration + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
>(
    mut state: CommitterState<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    receiver: mpsc::UnboundedReceiver<Pending>,
    env: TreeEnv<InboundQueueSenderType, DbConnectorType, TaskInstancePoolConnectorType>,
) {
    let root_id = state.structure.root_id();
    let token = env.shutdown.token().clone();
    let mut guard = PoisonOnAbnormalExit {
        receiver,
        shutdown: env.shutdown.clone(),
        root_id,
        disarmed: false,
    };
    let mut buffer: VecDeque<Queued> = VecDeque::new();
    loop {
        if buffer.is_empty() {
            let mut arrivals = Vec::with_capacity(MAX_EVENTS_PER_ROUND);
            let num_received = tokio::select! {
                num_received = guard.receiver.recv_many(&mut arrivals, MAX_EVENTS_PER_ROUND) => {
                    num_received
                }
                () = token.cancelled() => 0,
            };
            if num_received == 0 && !token.is_cancelled() {
                tracing::error!(
                    root_id = ? root_id,
                    "The event channel of a live tree closed."
                );
                guard.poison();
                return;
            }
            buffer.extend(arrivals.into_iter().map(Queued::new));
        } else {
            receive_queued(&mut guard.receiver, &mut buffer);
        }

        let hit = env
            .hooks
            .hit(HookPoint::CommitterAfterReceive, || {
                round_context(root_id, &buffer)
            })
            .await;
        match hit {
            Ok(()) if cfg!(feature = "test-hooks") => {
                receive_queued(&mut guard.receiver, &mut buffer);
            }
            Ok(()) => {}
            Err(fault) if fault.is_stopping() => {}
            Err(fault) => {
                tracing::error!(fault = ? fault, "Injected committer fault.");
                guard.poison();
                return;
            }
        }

        if token.is_cancelled() {
            close_and_answer(&mut guard.receiver, &mut buffer, || {
                Err(CacheError::ServiceStopping)
            });
            break;
        }

        let events = take_events(&mut buffer);
        #[cfg(not(spider_unboxed_round))]
        let result = Box::pin(run_tree_state_transaction(&mut state, events, &env)).await;
        #[cfg(spider_unboxed_round)]
        let result = run_tree_state_transaction(&mut state, events, &env).await;
        match result {
            Ok(requeued) => {
                for queued in requeued.into_iter().rev() {
                    buffer.push_front(queued);
                }
            }
            Err(e) => {
                tracing::error!(
                    root_id = ? root_id,
                    error = % e,
                    "Tree state transaction failed."
                );
                guard.poison();
                return;
            }
        }

        if state.structure.is_quiescent() {
            close_and_answer(&mut guard.receiver, &mut buffer, || {
                Ok(TreeOutcome::Untracked)
            });
            break;
        }
    }
    guard.disarmed = true;
}

/// Runs one tree state transaction over `events`: collects their inputs, plans them, writes the
/// plan in one database transaction, applies it to the cached jobs and the tree's structure,
/// enqueues the newly terminal jobs for cache GC, replies to the planned events, and publishes the
/// ready work.
///
/// # Returns
///
/// The events to put back at the front of the buffer, in their original order, on success: the
/// deferred events with their collected inputs, or every event, with its input to be collected
/// again, if the service started shutting down before the database transaction or if
/// [`fail_culprit`] turns the database's error into a failure of one event's job.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`collect_inputs`]'s return values on failure.
/// * Forwards [`fail_culprit`]'s return values on failure.
/// * Forwards [`SharedJobControlBlock::apply_tree_state`]'s return values on failure.
/// * Forwards [`CommitterState::bind_children`]'s return values on failure.
/// * Forwards [`JobCache::insert_many`]'s return values on failure.
/// * Forwards [`publish`]'s return values on failure.
///
/// # Panics
///
/// Panics if:
///
/// * Forwards [`CommitterState::bind_children`]'s panics.
/// * In debug, test and `test-hooks` builds, the tree breaks an invariant after the apply.
async fn run_tree_state_transaction<
    InboundQueueSenderType: InboundQueueSender + 'static,
    DbConnectorType: InternalJobOrchestration + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
>(
    state: &mut CommitterState<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    mut events: Vec<Queued>,
    env: &TreeEnv<InboundQueueSenderType, DbConnectorType, TaskInstancePoolConnectorType>,
) -> Result<Vec<Queued>, StorageServerError> {
    let root_id = state.structure.root_id();
    let collected = collect_inputs(state, &mut events).await?;
    match env
        .hooks
        .hit(HookPoint::CommitterBeforePlan, || {
            round_context(root_id, &events)
        })
        .await
    {
        Ok(()) => {}
        Err(fault) if fault.is_stopping() => return Ok(requeue_all(events)),
        Err(fault) => return Err(CacheError::from(fault).into()),
    }

    let mut plan = plan_round(&state.structure, &collected, &mut events);

    let injected_db_error = match env
        .hooks
        .hit(HookPoint::CommitterBeforeDbTransaction, || {
            round_context(root_id, &events)
        })
        .await
    {
        Ok(()) => None,
        Err(fault) if fault.is_stopping() => return Ok(requeue_all(events)),
        Err(fault) => Some(fault.into_db_error().map_err(CacheError::from)?),
    };

    let mut child_ids = Vec::new();
    if !plan.is_empty() {
        let result = match injected_db_error {
            Some(error) => Err(error),
            None => {
                write_plan(
                    std::mem::take(&mut plan.updates),
                    plan.registration.as_ref(),
                    &events,
                    &env.db,
                )
                .await
            }
        };
        child_ids = match result {
            Ok(child_ids) => child_ids,
            Err(error) => return fail_culprit(events, plan.registration.as_ref(), error),
        };
        hit_after_commit(&env.hooks, HookPoint::CommitterAfterCommit, || {
            round_context(root_id, &events)
        })
        .await?;
        apply_plan(state, &plan, &env.hooks, || round_context(root_id, &events)).await?;
    }

    if let Some(registration) = plan.registration {
        let prepared_children = std::mem::take(&mut events[registration.event_index].children);
        let children = state.bind_children(
            &registration,
            child_ids,
            prepared_children,
            env,
            &mut plan.publications,
        )?;
        env.job_cache.insert_many(children).await?;
    }

    let mut resource_group_ids = HashMap::new();
    for publication in &plan.publications {
        let job_id = publication.job_id();
        resource_group_ids.insert(job_id, state.jcb(job_id)?.owner_id());
    }
    state.update_structure(&plan);
    for job_id in &plan.terminated {
        env.gc.enqueue_terminated_job(*job_id);
    }

    let requeued = reply(events, std::mem::take(&mut plan.outcomes));

    for publication in plan.publications {
        let resource_group_id = resource_group_ids[&publication.job_id()];
        publish(&env.publisher, resource_group_id, publication).await?;
    }
    #[cfg(any(test, debug_assertions, feature = "test-hooks"))]
    assert_invariants(state).await;
    Ok(requeued)
}

/// Plans `events` against the tree's `structure` and the `collected` job states.
///
/// # Returns
///
/// Forwards [`plan_events`]'s return values.
///
/// # Panics
///
/// Panics if an event's input was not collected, or if [`plan_events`] panics.
fn plan_round(
    structure: &TreeStructure,
    collected: &HashMap<JobId, spider_core::job::JobState>,
    events: &mut [Queued],
) -> Plan {
    let mut planner_events: Vec<PlannerEvent<'_>> = events
        .iter_mut()
        .map(|queued| PlannerEvent {
            event: &queued.pending.event,
            input: queued
                .input
                .as_mut()
                .expect("every event's input should be collected before planning"),
        })
        .collect();
    plan_events(structure, collected, &mut planner_events)
}

/// Writes `updates`, and the children of `registration`, which the registering event in `events`
/// holds, in one database transaction.
///
/// # Type Parameters
///
/// * `DbConnectorType` - The type of the DB-layer connector.
///
/// # Returns
///
/// The IDs of the inserted children, in order, on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`InternalJobOrchestration::apply_tree_transition`]'s return values on failure.
async fn write_plan<DbConnectorType: InternalJobOrchestration>(
    updates: Vec<JobStateUpdate>,
    registration: Option<&PlannedRegistration>,
    events: &[Queued],
    db: &DbConnectorType,
) -> Result<Vec<JobId>, DbError> {
    let transition = TreeTransition {
        updates,
        child_inserts: registration
            .map(|registration| child_inserts(registration, events))
            .unwrap_or_default(),
    };
    db.apply_tree_transition(transition).await
}

/// Applies `plan`'s state changes to the cached jobs, in its apply order. After each job's apply,
/// `hooks` fire at [`HookPoint::CommitterAfterApplyJob`] with the job in the context that
/// `round_context` builds.
///
/// # Type Parameters
///
/// * `InboundQueueSenderType` - The type of the inbound queue sender of the tree's jobs.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`CommitterState::jcb`]'s return values on failure.
/// * Forwards [`SharedJobControlBlock::apply_tree_state`]'s return values on failure.
/// * Forwards [`hit_after_commit`]'s return values on failure.
async fn apply_plan<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
>(
    state: &CommitterState<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    plan: &Plan,
    hooks: &TestHooks,
    round_context: impl Fn() -> HookContext + Send + Sync,
) -> Result<(), CacheError> {
    for transition in &plan.apply_order {
        state
            .jcb(transition.job_id)?
            .apply_tree_state(transition.from, transition.to)
            .await?;
        hit_after_commit(hooks, HookPoint::CommitterAfterApplyJob, || HookContext {
            job_ids: vec![transition.job_id],
            ..round_context()
        })
        .await?;
    }
    Ok(())
}

/// Checks the tree's invariants against the cached states of its jobs.
///
/// # Type Parameters
///
/// * `InboundQueueSenderType` - The type of the inbound queue sender of the tree's jobs.
/// * `TaskInstancePoolConnectorType` - The type of the task instance pool connector.
///
/// # Panics
///
/// Panics if the tree breaks an invariant.
#[cfg(any(test, debug_assertions, feature = "test-hooks"))]
async fn assert_invariants<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
>(
    state: &CommitterState<InboundQueueSenderType, TaskInstancePoolConnectorType>,
) {
    let mut states = HashMap::with_capacity(state.jcbs.len());
    for (job_id, jcb) in &state.jcbs {
        states.insert(*job_id, jcb.state().await);
    }
    if let Err(violation) = check_invariants(&state.structure, &states) {
        panic!("the tree breaks an invariant after a tree state transaction: {violation}");
    }
}

/// Collects the states of the jobs that the planner reads for the round, and the input of every
/// event that has none yet, with the children that a `RegularTasksDone` event's job prepared. An
/// event whose job is no longer in the tree gets no input.
///
/// # Returns
///
/// The collected job states on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`CommitterState::jcb`]'s return values on failure.
/// * [`InternalError::TaskGraphCorrupted`] if a `RegularTasksDone` event's `registers` flag
///   disagrees with whether its job's tasks prepared children.
/// * Forwards [`SharedJobControlBlock::collect_ready_task_indices`]'s return values on failure.
/// * Forwards [`SharedJobControlBlock::collect_regular_done_inputs`]'s return values on failure.
async fn collect_inputs<
    InboundQueueSenderType: InboundQueueSender,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector,
>(
    state: &CommitterState<InboundQueueSenderType, TaskInstancePoolConnectorType>,
    events: &mut [Queued],
) -> Result<HashMap<JobId, spider_core::job::JobState>, CacheError> {
    let job_ids = state
        .structure
        .jobs_to_collect(events.iter().map(|queued| &queued.pending.event));
    let mut collected = HashMap::with_capacity(job_ids.len());
    for job_id in job_ids {
        collected.insert(job_id, state.jcb(job_id)?.state().await);
    }

    for queued in events.iter_mut() {
        if queued.input.is_some() {
            continue;
        }
        let Some(jcb) = state.jcbs.get(&queued.pending.event.job_id) else {
            queued.input = Some(PlannerInput::Empty);
            continue;
        };
        let input = match queued.pending.event.kind {
            TreeEventKind::Start => jcb
                .collect_ready_task_indices()
                .await?
                .map_or(PlannerInput::Empty, |ready_task_indices| {
                    PlannerInput::Start { ready_task_indices }
                }),
            TreeEventKind::RegularTasksDone { registers } => {
                match jcb.collect_regular_done_inputs().await? {
                    None => PlannerInput::Empty,
                    Some(RegularDoneInputs { outputs, children }) => {
                        queued.children.extend(children);
                        if registers == queued.children.is_empty() {
                            return Err(InternalError::TaskGraphCorrupted(format!(
                                "job {:?} reported `registers: {registers}` with {} prepared \
                                 children",
                                queued.pending.event.job_id,
                                queued.children.len()
                            ))
                            .into());
                        }
                        PlannerInput::RegularTasksDone {
                            outputs,
                            children: registers.then(|| ChildSetSummary {
                                count: queued.children.len(),
                                bytes: queued.children.iter().map(PreparedChild::size).sum(),
                            }),
                        }
                    }
                }
            }
            _ => PlannerInput::Empty,
        };
        queued.input = Some(input);
    }
    Ok(collected)
}

/// Stages one publication with `publisher`.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`InboundQueueSender::send_task_ready`]'s return values on failure.
/// * Forwards [`InboundQueueSender::send_commit_ready`]'s return values on failure.
/// * Forwards [`InboundQueueSender::send_cleanup_ready`]'s return values on failure.
async fn publish<InboundQueueSenderType: InboundQueueSender>(
    publisher: &InboundQueueSenderType,
    resource_group_id: ResourceGroupId,
    publication: Publication,
) -> Result<(), CacheError> {
    match publication {
        Publication::Ready {
            job_id,
            task_indices,
        } => {
            publisher
                .send_task_ready(resource_group_id, job_id, task_indices)
                .await?;
        }
        Publication::Commit(job_id) => {
            publisher
                .send_commit_ready(resource_group_id, job_id)
                .await?;
        }
        Publication::Cleanup(job_id) => {
            publisher
                .send_cleanup_ready(resource_group_id, job_id)
                .await?;
        }
    }
    Ok(())
}

/// Answers every event that has an outcome, and keeps the others.
///
/// # Returns
///
/// The events without an outcome, in their original order.
fn reply(events: Vec<Queued>, mut outcomes: BTreeMap<usize, TreeOutcome>) -> Vec<Queued> {
    let mut requeued = Vec::new();
    for (index, queued) in events.into_iter().enumerate() {
        let Some(outcome) = outcomes.remove(&index) else {
            requeued.push(queued);
            continue;
        };
        if let (TreeOutcome::Stale(error), TreeEventKind::TerminalFailure { message, .. }) =
            (&outcome, &queued.pending.event.kind)
        {
            tracing::info!(
                job_id = ? queued.pending.event.job_id,
                error = % error,
                failure = message.as_str(),
                "Discarded a stale task failure."
            );
        }
        // The event's waiter may have dropped its future, which is not an error of the committer.
        let _ = queued.pending.reply.send(Ok(outcome));
    }
    requeued
}

/// # Returns
///
/// The database part of `registration`: one insert per prepared child of the registering event,
/// in order.
fn child_inserts<'event_lifetime>(
    registration: &PlannedRegistration,
    events: &'event_lifetime [Queued],
) -> Vec<ChildJobInsert<'event_lifetime>> {
    events[registration.event_index]
        .children
        .iter()
        .map(|child| ChildJobInsert {
            parent_id: registration.parent_id,
            root_id: registration.root_id,
            resource_group_id: registration.resource_group_id,
            compressed_task_graph: &child.submission.compressed_serialized_task_graph,
            compressed_job_inputs: &child.submission.compressed_serialized_job_inputs,
        })
        .collect()
}

/// Turns a database error that one event of the round causes deterministically into a terminal
/// failure of that event's job on the regular lane, which keeps the event's reply channel, and
/// drops the event's prepared children:
///
/// * [`DbError::OutputsTooLarge`] fails the job whose `RegularTasksDone` event stores the outputs.
/// * Another deterministic payload error of a round with a `registration` fails the registering
///   parent.
///
/// # Returns
///
/// Forwards [`requeue_all`]'s return values on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards `error` if it is not deterministic, or if no event of the round causes it.
fn fail_culprit(
    mut events: Vec<Queued>,
    registration: Option<&PlannedRegistration>,
    error: DbError,
) -> Result<Vec<Queued>, StorageServerError> {
    let culprit = match &error {
        DbError::OutputsTooLarge { job_id, .. } => events
            .iter()
            .position(|queued| {
                queued.pending.event.job_id == *job_id
                    && matches!(
                        queued.pending.event.kind,
                        TreeEventKind::RegularTasksDone { .. }
                    )
            })
            .map(|index| (index, error.to_string())),
        error if error.is_deterministic_payload_error() => registration.map(|registration| {
            (
                registration.event_index,
                "child payload rejected by the database".to_owned(),
            )
        }),
        _ => None,
    };
    let Some((index, message)) = culprit else {
        return Err(error.into());
    };
    let queued = &mut events[index];
    tracing::error!(
        job_id = ? queued.pending.event.job_id,
        error = % error,
        "The database rejected the payload of a job's event. Failing the job."
    );
    queued.pending.event.kind = TreeEventKind::TerminalFailure {
        lane: Lane::Regular,
        message,
    };
    queued.children.clear();
    Ok(requeue_all(events))
}

/// # Returns
///
/// `events`, each with its input cleared so that the next round collects it again. Children that
/// were already taken from their job's tasks stay with their event.
fn requeue_all(mut events: Vec<Queued>) -> Vec<Queued> {
    for queued in &mut events {
        queued.input = None;
    }
    events
}

/// Hits a hook point that lies after the round's database transaction has committed. A pause that
/// ends because the service is stopping lets the round finish, as the round would without hooks.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`TestHooks::hit`]'s return values on failure, except a pause that ended because the
///   service is stopping.
async fn hit_after_commit(
    hooks: &TestHooks,
    point: HookPoint,
    context: impl FnOnce() -> HookContext,
) -> Result<(), CacheError> {
    match hooks.hit(point, context).await {
        Ok(()) => Ok(()),
        Err(fault) if fault.is_stopping() => Ok(()),
        Err(fault) => Err(fault.into()),
    }
}

/// # Returns
///
/// The hook context of a round of the tree rooted at `root_id` over the events of `queued`.
fn round_context<'queued_lifetime>(
    root_id: JobId,
    queued: impl IntoIterator<Item = &'queued_lifetime Queued>,
) -> HookContext {
    HookContext::round(
        root_id,
        queued.into_iter().map(|queued| &queued.pending.event),
    )
}

/// Moves every event already in the channel to the back of the buffer, without waiting.
fn receive_queued(receiver: &mut mpsc::UnboundedReceiver<Pending>, buffer: &mut VecDeque<Queued>) {
    while let Ok(pending) = receiver.try_recv() {
        buffer.push_back(Queued::new(pending));
    }
}

/// Takes the events of the next round from the buffer: the oldest `CancelRoot` first, wherever it
/// is, then the other events from the front of the buffer, up to [`MAX_EVENTS_PER_ROUND`] events.
/// Only the oldest registering event is taken; the later ones stay in the buffer, in place.
fn take_events(buffer: &mut VecDeque<Queued>) -> Vec<Queued> {
    let mut events = Vec::new();
    if let Some(cancel_index) = buffer
        .iter()
        .position(|queued| queued.pending.event.kind == TreeEventKind::CancelRoot)
    {
        events.extend(buffer.remove(cancel_index));
    }
    let mut kept = VecDeque::new();
    let mut takes_registration = false;
    while events.len() < MAX_EVENTS_PER_ROUND
        && let Some(queued) = buffer.pop_front()
    {
        if queued.pending.event.kind.registers() {
            if takes_registration {
                kept.push_back(queued);
                continue;
            }
            takes_registration = true;
        }
        events.push(queued);
    }
    kept.append(buffer);
    *buffer = kept;
    events
}

/// Closes the receiver, so that every later send fails, then answers every event still buffered
/// or queued with `reply`.
fn close_and_answer(
    receiver: &mut mpsc::UnboundedReceiver<Pending>,
    buffer: &mut VecDeque<Queued>,
    reply: impl Fn() -> TreeReply,
) {
    receiver.close();
    receive_queued(receiver, buffer);
    for queued in buffer.drain(..) {
        // The event's waiter may have dropped its future, which is not an error of the committer.
        let _ = queued.pending.reply.send(reply());
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use spider_core::job::JobState;
    use spider_core::task::DataTypeDescriptor;
    use spider_core::task::ExecutionPolicy;
    use spider_core::task::TaskDescriptor;
    use spider_core::task::TaskGraph as SubmittedTaskGraph;
    use spider_core::task::TdlContext;
    use spider_core::task::TerminationTaskDescriptor;
    use spider_core::task::ValueTypeDescriptor;
    use spider_core::types::id::JobId;
    use spider_core::types::id::ResourceGroupId;
    use spider_core::types::id::TaskId;
    use spider_core::types::io::TaskGraphInputBuilder;

    use super::*;
    use crate::cache::error::StaleStateError;
    use crate::cache::tree::TreeEvent;
    use crate::db::ExternalJobOrchestration;
    use crate::job_submission::ChildJobSubmission;
    use crate::job_submission::compress_job_inputs;
    use crate::job_submission::compress_task_graph;
    use crate::job_submission::create_validated_submission;
    use crate::state::test_utils::MockDbConnector;
    use crate::state::test_utils::MockInboundQueueSender;
    use crate::state::test_utils::MockTaskInstancePoolConnector;
    use crate::state::test_utils::create_test_tree_env;

    type TestEnv = TreeEnv<MockInboundQueueSender, MockDbConnector, MockTaskInstancePoolConnector>;
    type TestJcb = SharedJobControlBlock<MockInboundQueueSender, MockTaskInstancePoolConnector>;

    /// # Returns
    ///
    /// `num_children` prepared one-task children.
    async fn prepare_children(num_children: usize) -> Vec<PreparedChild> {
        let (task_graph, inputs) = one_task_graph(false);
        let submission = ChildJobSubmission {
            compressed_serialized_task_graph: compress_task_graph(&task_graph),
            compressed_serialized_job_inputs: compress_job_inputs(&inputs),
        };
        let mut children = Vec::new();
        for _ in 0..num_children {
            children.push(
                PreparedChild::create(submission.clone())
                    .await
                    .expect("child preparation should succeed"),
            );
        }
        children
    }

    /// Starts `jcb` and reports the success of a new instance of its only task, with `children`.
    ///
    /// # Returns
    ///
    /// Forwards [`SharedJobControlBlock::succeed_task_instance`]'s return values.
    async fn run_only_task(
        jcb: &TestJcb,
        children: Vec<PreparedChild>,
    ) -> anyhow::Result<JobState> {
        if jcb.state().await == JobState::Ready {
            jcb.start().await?;
        }
        let context = jcb
            .create_task_instance(
                TaskId::Index(0),
                spider_core::types::id::ExecutionManagerId::random(),
            )
            .await?;
        Ok(jcb
            .succeed_task_instance(context.task_instance_id, 0, vec![vec![7]], children)
            .await?)
    }

    /// # Returns
    ///
    /// The IDs of `parent_id`'s children in the mock database, ascending.
    fn child_ids(env: &TestEnv, parent_id: JobId) -> Vec<JobId> {
        let mut child_ids: Vec<JobId> = env
            .db
            .parents
            .iter()
            .filter(|entry| *entry.value() == parent_id)
            .map(|entry| *entry.key())
            .collect();
        child_ids.sort_unstable_by_key(JobId::get);
        child_ids
    }

    /// # Returns
    ///
    /// A one-task graph whose task takes and returns one byte-typed value, with a cleanup task if
    /// `with_cleanup`, and its inputs.
    fn one_task_graph(
        with_cleanup: bool,
    ) -> (SubmittedTaskGraph, spider_core::types::io::TaskGraphInput) {
        let tdl_context = TdlContext {
            package: "test_pkg".to_owned(),
            task_func: "test_fn".to_owned(),
        };
        let cleanup_task = with_cleanup.then(|| TerminationTaskDescriptor {
            tdl_context: tdl_context.clone(),
            execution_policy: None,
        });
        let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
        let mut task_graph = SubmittedTaskGraph::new(None, cleanup_task)
            .expect("task graph creation should succeed");
        task_graph
            .insert_task(TaskDescriptor {
                tdl_context,
                execution_policy: Some(ExecutionPolicy::default()),
                inputs: vec![bytes_type.clone()],
                outputs: vec![bytes_type],
                input_sources: None,
            })
            .expect("task insertion should succeed");
        let mut inputs = TaskGraphInputBuilder::new();
        inputs
            .append_task_input(&[0u8; 4])
            .expect("task input appending should succeed");
        (task_graph, inputs.build())
    }

    /// Registers a one-task job in the mock database and creates its tree.
    async fn create_job(env: &TestEnv, with_cleanup: bool) -> TestJcb {
        let (task_graph, inputs) = one_task_graph(with_cleanup);
        let submission = create_validated_submission(task_graph, inputs);
        let owner_id = ResourceGroupId::random();
        let job_id = env
            .db
            .register(owner_id, &submission)
            .await
            .expect("mock registration should succeed");
        create_root_tree(env, job_id, owner_id, submission)
            .await
            .expect("tree creation should succeed")
    }

    fn event(jcb: &TestJcb, kind: TreeEventKind) -> TreeEvent {
        TreeEvent {
            job_id: jcb.id(),
            kind,
        }
    }

    /// Waits until every task of the environment's tracker has exited.
    async fn wait_for_committers(env: &TestEnv) {
        env.tracker.close();
        tokio::time::timeout(Duration::from_secs(5), env.tracker.wait())
            .await
            .expect("every committer should exit");
    }

    #[tokio::test]
    async fn committer_exits_when_its_root_becomes_terminal() -> anyhow::Result<()> {
        let (env, mut gc_receiver) =
            create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
        let jcb = create_job(&env, false).await;

        jcb.start().await?;
        let context = jcb
            .create_task_instance(
                TaskId::Index(0),
                spider_core::types::id::ExecutionManagerId::random(),
            )
            .await?;
        let state = jcb
            .succeed_task_instance(context.task_instance_id, 0, vec![vec![1, 2]], Vec::new())
            .await?;
        assert_eq!(state, JobState::Succeeded);
        assert_eq!(gc_receiver.try_recv(), Ok(jcb.id()));
        assert_eq!(
            env.db.states.get(&jcb.id()).map(|state| *state),
            Some(JobState::Succeeded)
        );
        wait_for_committers(&env).await;

        let start = jcb.start().await;
        assert!(
            matches!(
                start,
                Err(CacheError::StaleState(StaleStateError::JobAlreadyStarted))
            ),
            "a start after the committer exited should be stale, got {start:?}"
        );
        let cancel = jcb.cancel().await;
        assert!(
            matches!(
                cancel,
                Err(CacheError::StaleState(
                    StaleStateError::JobAlreadyTerminated(JobState::Succeeded)
                ))
            ),
            "a cancel after the committer exited should be stale, got {cancel:?}"
        );
        assert!(!env.shutdown.is_fatal());
        Ok(())
    }

    #[tokio::test]
    async fn committer_stays_while_its_root_is_not_terminal() -> anyhow::Result<()> {
        let (env, _gc_receiver) =
            create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
        let jcb = create_job(&env, true).await;

        jcb.start().await?;
        assert_eq!(jcb.cancel().await?, JobState::CleanupReady);
        let reply = jcb
            .tree()
            .submit(event(&jcb, TreeEventKind::CancelRoot))
            .await?;
        assert!(
            matches!(
                reply,
                Ok(TreeOutcome::Stale(
                    StaleStateError::JobCancellationAlreadyRequested
                ))
            ),
            "a second cancel should be planned stale, got {reply:?}"
        );
        assert!(!jcb.tree().is_stopping());
        assert!(!env.tracker.is_empty(), "the committer should still run");
        Ok(())
    }

    #[tokio::test]
    async fn a_registration_blocks_the_parent_until_every_child_succeeds() -> anyhow::Result<()> {
        let (env, mut gc_receiver) =
            create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
        let parent = create_job(&env, false).await;

        let state = run_only_task(&parent, prepare_children(2).await).await?;
        assert_eq!(state, JobState::BlockingOnChildren);
        assert_eq!(parent.state().await, JobState::BlockingOnChildren);
        assert_eq!(
            env.db.states.get(&parent.id()).map(|state| *state),
            Some(JobState::BlockingOnChildren)
        );
        assert_eq!(
            env.db
                .outputs
                .get(&parent.id())
                .map(|outputs| outputs.clone()),
            Some(vec![vec![7]])
        );
        let child_ids = child_ids(&env, parent.id());
        assert_eq!(child_ids.len(), 2);
        let mut children = Vec::new();
        for child_id in &child_ids {
            let child = env
                .job_cache
                .get(*child_id)
                .await
                .expect("a registered child should be cached");
            assert_eq!(child.state().await, JobState::Running);
            assert_eq!(child.parent_id(), Some(parent.id()));
            assert_eq!(child.depth(), 1);
            assert_eq!(child.tree().root_id(), parent.id());
            children.push(child);
        }

        assert_eq!(
            run_only_task(&children[0], Vec::new()).await?,
            JobState::Succeeded
        );
        assert_eq!(parent.state().await, JobState::BlockingOnChildren);
        assert_eq!(
            run_only_task(&children[1], Vec::new()).await?,
            JobState::Succeeded
        );
        assert_eq!(parent.state().await, JobState::Succeeded);
        assert_eq!(
            env.db.states.get(&parent.id()).map(|state| *state),
            Some(JobState::Succeeded)
        );
        let mut terminated = Vec::new();
        while let Ok(job_id) = gc_receiver.try_recv() {
            terminated.push(job_id);
        }
        assert_eq!(terminated, vec![child_ids[0], child_ids[1], parent.id()]);
        wait_for_committers(&env).await;
        assert!(!env.shutdown.is_fatal());
        Ok(())
    }

    #[tokio::test]
    async fn outputs_over_the_job_limit_fail_the_job_and_abort_its_tree() -> anyhow::Result<()> {
        let (env, _gc_receiver) =
            create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
        let root = create_job(&env, false).await;
        assert_eq!(
            run_only_task(&root, prepare_children(2).await).await?,
            JobState::BlockingOnChildren
        );
        let child_ids = child_ids(&env, root.id());
        let failing = env
            .job_cache
            .get(child_ids[0])
            .await
            .expect("a registered child should be cached");
        let context = failing
            .create_task_instance(
                TaskId::Index(0),
                spider_core::types::id::ExecutionManagerId::random(),
            )
            .await?;
        let output_size = env.db.max_statement_payload() + 1;
        let state = failing
            .succeed_task_instance(
                context.task_instance_id,
                0,
                vec![vec![0u8; usize::try_from(output_size)?]],
                Vec::new(),
            )
            .await?;

        assert_eq!(state, JobState::Failed);
        assert_eq!(
            env.db.errors.get(&child_ids[0]).map(|error| error.clone()),
            Some(
                DbError::OutputsTooLarge {
                    job_id: child_ids[0],
                    size: output_size,
                    limit: output_size - 1,
                }
                .to_string()
            )
        );
        for (job_id, state) in [
            (root.id(), JobState::Failed),
            (child_ids[0], JobState::Failed),
            (child_ids[1], JobState::Cancelled),
        ] {
            assert_eq!(env.db.states.get(&job_id).map(|state| *state), Some(state));
        }
        assert_eq!(env.db.outputs.get(&child_ids[0]).map(|_| ()), None);
        wait_for_committers(&env).await;
        assert!(!env.shutdown.is_fatal());
        Ok(())
    }

    #[test]
    fn a_guard_dropped_after_the_token_is_cancelled_does_not_poison() {
        for cancels_first in [false, true] {
            let shutdown = ShutdownHandle::new(tokio_util::sync::CancellationToken::new());
            let (_sender, receiver) = mpsc::unbounded_channel();
            let guard = PoisonOnAbnormalExit {
                receiver,
                shutdown: shutdown.clone(),
                root_id: JobId::from(1),
                disarmed: false,
            };
            if cancels_first {
                shutdown.token().cancel();
            }
            drop(guard);
            assert_eq!(shutdown.is_fatal(), !cancels_first);
            assert!(shutdown.token().is_cancelled());
        }
    }

    #[test]
    fn take_events_takes_the_oldest_cancel_first_and_leaves_later_registrations_in_the_buffer() {
        let registration = TreeEventKind::RegularTasksDone { registers: true };
        let mut buffer: VecDeque<Queued> = [
            (2, registration.clone()),
            (3, TreeEventKind::CommitDone),
            (4, registration.clone()),
            (1, TreeEventKind::CancelRoot),
            (5, TreeEventKind::CleanupDone),
        ]
        .into_iter()
        .map(|(id, kind)| {
            let (reply, _) = tokio::sync::oneshot::channel();
            Queued::new(Pending {
                event: TreeEvent {
                    job_id: JobId::from(id),
                    kind,
                },
                reply,
            })
        })
        .collect();
        let events_of = |queued: &mut dyn Iterator<Item = &Queued>| {
            queued
                .map(|queued| {
                    (
                        queued.pending.event.job_id.get(),
                        queued.pending.event.kind.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };

        let round = take_events(&mut buffer);
        assert_eq!(
            events_of(&mut round.iter()),
            vec![
                (1, TreeEventKind::CancelRoot),
                (2, registration.clone()),
                (3, TreeEventKind::CommitDone),
                (5, TreeEventKind::CleanupDone),
            ]
        );
        assert_eq!(events_of(&mut buffer.iter()), vec![(4, registration)]);
        assert!(buffer.iter().all(|queued| queued.input.is_none()));
    }

    #[test]
    fn committer_future_size() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime creation should succeed");
        let _guard = runtime.enter();
        let (env, _gc_receiver) =
            create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
        let job_id = JobId::from(1);
        let (_tree, receiver) = new_tree_control_block(&env, job_id);
        let state: CommitterState<MockInboundQueueSender, MockTaskInstancePoolConnector> =
            CommitterState {
                structure: TreeStructure::new_root(
                    job_id,
                    TreeNode {
                        parent: None,
                        depth: 0,
                        resource_group_id: ResourceGroupId::from(1),
                        has_commit: false,
                        has_cleanup: false,
                        active_children: 0,
                    },
                ),
                jcbs: HashMap::new(),
            };
        let future = run_committer(state, receiver, env);
        println!(
            "Size of the committer future (unboxed round: {}): {} bytes.",
            cfg!(spider_unboxed_round),
            std::mem::size_of_val(&future)
        );
    }

    #[cfg(feature = "test-hooks")]
    mod hooked {
        use super::*;
        use crate::test_hooks::EventKindTag;
        use crate::test_hooks::HookAction;
        use crate::test_hooks::HookCount;
        use crate::test_hooks::HookFilter;

        #[tokio::test]
        async fn shutdown_finishes_the_round_in_flight_and_answers_the_queued_event_with_stopping()
        -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let jcb = create_job(&env, false).await;
            let pause = env.hooks.arm(
                HookPoint::CommitterAfterCommit,
                HookFilter::root(jcb.id()),
                HookAction::Pause,
                HookCount::Once,
            );

            let start = jcb.tree().submit(event(&jcb, TreeEventKind::Start));
            pause.reached(1).await;
            let cancel = jcb.tree().submit(event(&jcb, TreeEventKind::CancelRoot));
            env.shutdown.token().cancel();

            let start = start.await;
            assert!(
                matches!(start, Ok(Ok(TreeOutcome::Applied(JobState::Running)))),
                "the round in flight should be applied, got {start:?}"
            );
            let cancel = cancel.await;
            assert!(
                matches!(cancel, Ok(Err(CacheError::ServiceStopping))),
                "the queued event should be answered Stopping, got {cancel:?}"
            );
            wait_for_committers(&env).await;
            let late = jcb
                .tree()
                .submit(event(&jcb, TreeEventKind::CancelRoot))
                .await;
            assert!(matches!(late, Ok(Ok(TreeOutcome::Untracked))));
            assert!(!env.shutdown.is_fatal());
            assert_eq!(jcb.state().await, JobState::Running);
            assert_eq!(
                env.db.states.get(&jcb.id()).map(|state| *state),
                Some(JobState::Running)
            );
            Ok(())
        }

        #[tokio::test]
        async fn a_round_requeued_before_the_db_transaction_collects_its_inputs_again()
        -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let jcb = create_job(&env, false).await;
            let mut state = CommitterState::new_root(jcb.clone());
            let (reply, mut reply_receiver) = tokio::sync::oneshot::channel();
            let pending = Pending {
                event: event(&jcb, TreeEventKind::Start),
                reply,
            };
            let _pause = env.hooks.arm(
                HookPoint::CommitterBeforeDbTransaction,
                HookFilter::root(jcb.id()),
                HookAction::Pause,
                HookCount::Once,
            );
            env.shutdown.token().cancel();

            let requeued =
                run_tree_state_transaction(&mut state, vec![Queued::new(pending)], &env).await?;
            assert_eq!(
                requeued
                    .iter()
                    .map(|queued| queued.input.as_ref())
                    .collect::<Vec<_>>(),
                vec![None]
            );
            assert_eq!(
                env.db.states.get(&jcb.id()).map(|state| *state),
                Some(JobState::Ready)
            );

            let requeued = run_tree_state_transaction(&mut state, requeued, &env).await?;
            assert!(
                requeued.is_empty(),
                "the replanned round should be answered"
            );
            let reply = reply_receiver.try_recv();
            assert!(
                matches!(reply, Ok(Ok(TreeOutcome::Applied(JobState::Running)))),
                "the replanned start should be applied, got {reply:?}"
            );
            assert_eq!(jcb.state().await, JobState::Running);
            assert_eq!(
                env.db.states.get(&jcb.id()).map(|state| *state),
                Some(JobState::Running)
            );
            Ok(())
        }

        /// Holds the committer of `jcb`'s tree after it receives the first of `kinds`, until it has
        /// received all of them, so that they share one round.
        ///
        /// # Returns
        ///
        /// A pair, containing:
        ///
        /// * The receivers of the events' replies, in submission order.
        /// * A hook that observes every round planned after this call.
        async fn submit_in_one_round(
            env: &TestEnv,
            jcb: &TestJcb,
            kinds: Vec<TreeEventKind>,
        ) -> (
            Vec<tokio::sync::oneshot::Receiver<TreeReply>>,
            crate::test_hooks::ArmedHook,
        ) {
            let pause = env.hooks.arm(
                HookPoint::CommitterAfterReceive,
                HookFilter::root(jcb.id()),
                HookAction::Pause,
                HookCount::Once,
            );
            let rounds = env.hooks.arm(
                HookPoint::CommitterBeforePlan,
                HookFilter::root(jcb.id()),
                HookAction::Observe,
                HookCount::Always,
            );
            let mut replies = Vec::new();
            for kind in kinds {
                replies.push(jcb.tree().submit(event(jcb, kind)));
                pause.reached(1).await;
            }
            pause.release();
            (replies, rounds)
        }

        fn failure(message: &str) -> TreeEventKind {
            TreeEventKind::TerminalFailure {
                lane: crate::cache::tree::Lane::Regular,
                message: message.to_owned(),
            }
        }

        #[tokio::test]
        async fn two_failures_of_one_job_in_one_round_fail_it_once() -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let jcb = create_job(&env, false).await;
            jcb.start().await?;

            let (replies, rounds) =
                submit_in_one_round(&env, &jcb, vec![failure("first"), failure("second")]).await;
            let mut outcomes = Vec::new();
            for reply in replies {
                outcomes.push(format!("{:?}", reply.await?));
            }
            assert_eq!(
                outcomes,
                vec![
                    "Ok(Applied(Failed))".to_owned(),
                    "Ok(Stale(JobAlreadyTerminated(Failed)))".to_owned(),
                ]
            );
            assert_eq!(rounds.hits(), 1);
            assert_eq!(
                rounds.last_context().map(|context| context.event_kinds),
                Some(vec![EventKindTag::TerminalFailure; 2])
            );
            assert_eq!(
                env.db.errors.get(&jcb.id()).map(|error| error.clone()),
                Some("first".to_owned())
            );
            assert_eq!(jcb.state().await, JobState::Failed);
            assert!(!env.shutdown.is_fatal());
            Ok(())
        }

        #[tokio::test]
        async fn a_cancel_is_planned_before_a_failure_that_arrived_first() -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let jcb = create_job(&env, true).await;
            jcb.start().await?;

            let (replies, rounds) =
                submit_in_one_round(&env, &jcb, vec![failure("boom"), TreeEventKind::CancelRoot])
                    .await;
            let mut outcomes = Vec::new();
            for reply in replies {
                outcomes.push(format!("{:?}", reply.await?));
            }
            assert_eq!(
                outcomes,
                vec![
                    "Ok(Stale(JobNoLongerRunning))".to_owned(),
                    "Ok(Applied(CleanupReady))".to_owned(),
                ]
            );
            assert_eq!(
                rounds.last_context().map(|context| context.event_kinds),
                Some(vec![
                    EventKindTag::CancelRoot,
                    EventKindTag::TerminalFailure
                ])
            );
            assert_eq!(
                env.db.errors.get(&jcb.id()).map(|error| error.clone()),
                None
            );
            assert_eq!(
                env.db.states.get(&jcb.id()).map(|state| *state),
                Some(JobState::CleanupReady)
            );
            assert_eq!(jcb.state().await, JobState::CleanupReady);
            Ok(())
        }

        #[tokio::test]
        async fn an_injected_event_that_would_change_its_job_again_waits_for_the_next_round()
        -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let jcb = create_job(&env, false).await;
            let first_round = env.hooks.arm(
                HookPoint::CommitterBeforePlan,
                HookFilter::root(jcb.id()),
                HookAction::Observe,
                HookCount::Once,
            );

            let (replies, rounds) =
                submit_in_one_round(&env, &jcb, vec![TreeEventKind::Start, failure("boom")]).await;
            let mut outcomes = Vec::new();
            for reply in replies {
                outcomes.push(format!("{:?}", reply.await?));
            }
            assert_eq!(
                outcomes,
                vec![
                    "Ok(Applied(Running))".to_owned(),
                    "Ok(Applied(Failed))".to_owned(),
                ]
            );
            assert_eq!(rounds.hits(), 2);
            assert_eq!(
                first_round
                    .last_context()
                    .map(|context| context.event_kinds),
                Some(vec![EventKindTag::Start, EventKindTag::TerminalFailure])
            );
            assert_eq!(
                rounds.last_context().map(|context| context.event_kinds),
                Some(vec![EventKindTag::TerminalFailure])
            );
            assert_eq!(
                env.db.states.get(&jcb.id()).map(|state| *state),
                Some(JobState::Failed)
            );
            Ok(())
        }

        /// Arms `action` at `point`, submits a start and a cancel, and checks that the committer
        /// poisons the service and wakes both waiters.
        async fn assert_fault_poisons_the_tree(point: HookPoint, action: HookAction) {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let jcb = create_job(&env, false).await;
            let _hook = env
                .hooks
                .arm(point, HookFilter::root(jcb.id()), action, HookCount::Once);

            let start = jcb.tree().submit(event(&jcb, TreeEventKind::Start));
            let cancel = jcb.tree().submit(event(&jcb, TreeEventKind::CancelRoot));
            for reply in [start.await, cancel.await] {
                assert!(
                    reply.is_err(),
                    "the reply channel should close without a reply, got {reply:?}"
                );
            }
            wait_for_committers(&env).await;
            assert!(env.shutdown.is_fatal());
            assert!(env.shutdown.token().is_cancelled());
            let later = jcb.start().await;
            assert!(
                matches!(later, Err(CacheError::ServiceStopping)),
                "a later event should be answered Stopping, got {later:?}"
            );
        }

        #[tokio::test]
        async fn a_rejected_registration_payload_fails_the_parent_in_the_next_round()
        -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let parent = create_job(&env, false).await;
            parent.start().await?;
            let fail_db = env.hooks.arm(
                HookPoint::CommitterBeforeDbTransaction,
                HookFilter::root(parent.id()),
                HookAction::FailDb {
                    deterministic_payload: true,
                },
                HookCount::Once,
            );
            let pause = env.hooks.arm(
                HookPoint::CommitterAfterReceive,
                HookFilter::root(parent.id()),
                HookAction::Pause,
                HookCount::Once,
            );
            let rounds = env.hooks.arm(
                HookPoint::CommitterBeforePlan,
                HookFilter::root(parent.id()),
                HookAction::Observe,
                HookCount::Always,
            );

            let report = tokio::spawn({
                let parent = parent.clone();
                let children = prepare_children(2).await;
                async move { run_only_task(&parent, children).await }
            });
            pause.reached(1).await;
            let start = parent.tree().submit(event(&parent, TreeEventKind::Start));
            pause.release();

            assert_eq!(report.await??, JobState::Failed);
            let start = start.await;
            assert!(
                matches!(
                    start,
                    Ok(Ok(TreeOutcome::Stale(StaleStateError::JobAlreadyStarted)))
                ),
                "the other event of the round should be answered once, got {start:?}"
            );
            assert_eq!(fail_db.hits(), 1);
            assert_eq!(rounds.hits(), 2);
            assert_eq!(
                rounds.last_context().map(|context| context.event_kinds),
                Some(vec![EventKindTag::TerminalFailure, EventKindTag::Start])
            );
            assert_eq!(
                env.db.errors.get(&parent.id()).map(|error| error.clone()),
                Some("child payload rejected by the database".to_owned())
            );
            assert_eq!(
                env.db.states.get(&parent.id()).map(|state| *state),
                Some(JobState::Failed)
            );
            assert_eq!(child_ids(&env, parent.id()), Vec::new());
            assert!(!env.shutdown.is_fatal());
            Ok(())
        }

        #[tokio::test]
        async fn a_completion_rolled_back_with_a_rejected_registration_is_planned_again()
        -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let root = create_job(&env, false).await;
            assert_eq!(
                run_only_task(&root, prepare_children(2).await).await?,
                JobState::BlockingOnChildren
            );
            let mut children = Vec::new();
            for child_id in child_ids(&env, root.id()) {
                children.push(
                    env.job_cache
                        .get(child_id)
                        .await
                        .expect("a registered child should be cached"),
                );
            }
            let (registering, completing) = (children[0].clone(), children[1].clone());
            let fail_db = env.hooks.arm(
                HookPoint::CommitterBeforeDbTransaction,
                HookFilter::root(root.id()),
                HookAction::FailDb {
                    deterministic_payload: true,
                },
                HookCount::Once,
            );
            let pause = env.hooks.arm(
                HookPoint::CommitterAfterReceive,
                HookFilter::root(root.id()),
                HookAction::Pause,
                HookCount::Once,
            );
            let registration_submitted = env.hooks.arm(
                HookPoint::EventSubmitted,
                HookFilter::job(registering.id()),
                HookAction::Observe,
                HookCount::Always,
            );
            let rounds = env.hooks.arm(
                HookPoint::CommitterBeforePlan,
                HookFilter::root(root.id()),
                HookAction::Observe,
                HookCount::Always,
            );

            let completion = tokio::spawn({
                let completing = completing.clone();
                async move { run_only_task(&completing, Vec::new()).await }
            });
            pause.reached(1).await;
            let registration = tokio::spawn({
                let registering = registering.clone();
                let grandchildren = prepare_children(1).await;
                async move { run_only_task(&registering, grandchildren).await }
            });
            registration_submitted.reached(1).await;
            pause.release();

            assert_eq!(registration.await??, JobState::Failed);
            assert_eq!(completion.await??, JobState::Succeeded);
            assert_eq!(fail_db.hits(), 1);
            assert_eq!(rounds.hits(), 2);
            assert_eq!(
                rounds.last_context().map(|context| context.event_kinds),
                Some(vec![
                    EventKindTag::RegularTasksDone,
                    EventKindTag::TerminalFailure
                ])
            );
            for (job_id, state) in [
                (root.id(), JobState::Failed),
                (registering.id(), JobState::Failed),
                (completing.id(), JobState::Succeeded),
            ] {
                assert_eq!(env.db.states.get(&job_id).map(|state| *state), Some(state));
            }
            assert_eq!(
                env.db
                    .outputs
                    .get(&completing.id())
                    .map(|outputs| outputs.clone()),
                Some(vec![vec![7]])
            );
            assert_eq!(child_ids(&env, registering.id()), Vec::new());
            assert!(!env.shutdown.is_fatal());
            Ok(())
        }

        #[tokio::test]
        async fn a_payload_error_without_a_registration_poisons_the_tree() {
            assert_fault_poisons_the_tree(
                HookPoint::CommitterBeforeDbTransaction,
                HookAction::FailDb {
                    deterministic_payload: true,
                },
            )
            .await;
        }

        #[tokio::test]
        async fn a_transient_error_of_a_registration_poisons_the_tree() -> anyhow::Result<()> {
            let (env, _gc_receiver) =
                create_test_tree_env(MockDbConnector::default(), MockInboundQueueSender);
            let parent = create_job(&env, false).await;
            parent.start().await?;
            let _fail_db = env.hooks.arm(
                HookPoint::CommitterBeforeDbTransaction,
                HookFilter::root(parent.id()),
                HookAction::FailDb {
                    deterministic_payload: false,
                },
                HookCount::Once,
            );

            let report = run_only_task(&parent, prepare_children(1).await).await;
            assert!(
                report.is_err(),
                "the report should fail with the tree, got {report:?}"
            );
            wait_for_committers(&env).await;
            assert!(env.shutdown.is_fatal());
            assert_eq!(child_ids(&env, parent.id()), Vec::new());
            Ok(())
        }

        #[tokio::test]
        async fn crash_before_the_db_transaction_poisons_the_tree() {
            assert_fault_poisons_the_tree(
                HookPoint::CommitterBeforeDbTransaction,
                HookAction::Crash,
            )
            .await;
        }

        #[tokio::test]
        async fn panic_in_the_apply_poisons_the_tree() {
            assert_fault_poisons_the_tree(HookPoint::CommitterAfterApplyJob, HookAction::Panic)
                .await;
        }
    }
}
