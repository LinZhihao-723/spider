//! Job trees: event submission, tree structure and the pure planner.
//!
//! Every job of a tree submits its job-level events to the tree's committer through the tree
//! control block. The planner turns a round of events into one tree transition.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use spider_core::job::JobState;
use spider_core::task::TaskIndex;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use spider_core::types::io::TaskOutput;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::cache::error::CacheError;
use crate::cache::error::InternalError;
use crate::cache::error::StaleStateError;
use crate::db::JobStateUpdate;
use crate::db::RecoverableJobContext;
use crate::limits::MAX_CHILDREN_PER_JOB;
use crate::limits::MAX_ERROR_MESSAGE_CHARS;
use crate::limits::MAX_EVENTS_PER_ROUND;
use crate::limits::MAX_JOB_CHILD_BYTES;
use crate::limits::MAX_ROUND_PAYLOAD;
use crate::test_hooks::EventKindTag;
use crate::test_hooks::HookContext;
use crate::test_hooks::HookPoint;
use crate::test_hooks::TestHooks;

/// The lane of a task: regular, commit or cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Regular,
    Commit,
    Cleanup,
}

impl Lane {
    /// # Returns
    ///
    /// Whether a job in `state` runs tasks of this lane.
    #[must_use]
    pub const fn accepts(self, state: JobState) -> bool {
        matches!(
            (self, state),
            (Self::Regular, JobState::Running)
                | (Self::Commit, JobState::CommitReady)
                | (Self::Cleanup, JobState::CleanupReady)
        )
    }

    /// # Returns
    ///
    /// The stale-state error for a report of this lane that finds its job in `state`.
    #[must_use]
    pub const fn stale_error(self, state: JobState) -> StaleStateError {
        if state.is_terminal() {
            return StaleStateError::JobAlreadyTerminated(state);
        }
        match self {
            Self::Regular => StaleStateError::JobNoLongerRunning,
            Self::Commit => StaleStateError::JobNoLongerCommitReady,
            Self::Cleanup => StaleStateError::JobNoLongerCleanupReady,
        }
    }
}

/// The kind of a job-level event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeEventKind {
    /// The root is started.
    Start,

    /// The job's last regular task succeeded. `registers` is whether its tasks created children.
    RegularTasksDone { registers: bool },

    /// The job's commit task succeeded.
    CommitDone,

    /// The job's cleanup task succeeded.
    CleanupDone,

    /// A task of the job failed with no retry left.
    TerminalFailure { lane: Lane, message: String },

    /// The root is cancelled.
    CancelRoot,
}

impl TreeEventKind {
    /// # Returns
    ///
    /// The stale-state error for an event of this kind whose job is in `state`.
    #[must_use]
    pub const fn stale_error(&self, state: JobState) -> StaleStateError {
        match self {
            Self::Start => StaleStateError::JobAlreadyStarted,
            Self::RegularTasksDone { .. } => Lane::Regular.stale_error(state),
            Self::CommitDone => Lane::Commit.stale_error(state),
            Self::CleanupDone => Lane::Cleanup.stale_error(state),
            Self::TerminalFailure { lane, .. } => lane.stale_error(state),
            Self::CancelRoot => match state {
                JobState::CleanupReady => StaleStateError::JobCancellationAlreadyRequested,
                JobState::Cancelled => StaleStateError::JobAlreadyCancelled,
                state => StaleStateError::JobAlreadyTerminated(state),
            },
        }
    }

    /// # Returns
    ///
    /// Whether the event registers children of its job.
    #[must_use]
    pub const fn registers(&self) -> bool {
        matches!(self, Self::RegularTasksDone { registers: true })
    }

    /// # Returns
    ///
    /// Whether planning the event can read the state of every job of the tree, because it can
    /// deactivate jobs anywhere in the tree.
    #[must_use]
    pub const fn reads_every_job(&self) -> bool {
        matches!(
            self,
            Self::CancelRoot
                | Self::TerminalFailure { .. }
                | Self::RegularTasksDone { registers: true }
        )
    }
}

/// A job-level event of one job of a tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeEvent {
    pub job_id: JobId,
    pub kind: TreeEventKind,
}

/// The planned result of an event, and the committer's answer to it.
#[derive(Debug)]
pub enum TreeOutcome {
    /// The event was written and applied. Carries the job's new state.
    Applied(JobState),

    /// The event lost a race, and nothing was written for it.
    Stale(StaleStateError),

    /// The tree no longer tracks the event's job, so the event was not planned: the tree's
    /// committer has exited, or the job became terminal in an earlier tree state transaction.
    Untracked,
}

/// The reply to an event.
pub type TreeReply = Result<TreeOutcome, CacheError>;

/// An event on its way to the tree's committer, with the channel for its reply.
pub struct Pending {
    pub event: TreeEvent,
    pub reply: oneshot::Sender<TreeReply>,
}

/// The handle every job of a tree uses to submit events to the tree's committer.
#[derive(Clone)]
pub struct TreeControlBlock {
    root_id: JobId,
    sender: mpsc::UnboundedSender<Pending>,
    token: CancellationToken,
    hooks: TestHooks,
}

impl TreeControlBlock {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A pair, containing:
    ///
    /// * The control block of the tree rooted at `root_id`, whose replies are raced against
    ///   `token`.
    /// * The receiver of the tree's events, for the tree's committer.
    #[must_use]
    pub fn new(
        root_id: JobId,
        token: CancellationToken,
        hooks: TestHooks,
    ) -> (Self, mpsc::UnboundedReceiver<Pending>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self {
                root_id,
                sender,
                token,
                hooks,
            },
            receiver,
        )
    }

    #[must_use]
    pub const fn root_id(&self) -> JobId {
        self.root_id
    }

    /// # Returns
    ///
    /// Whether the storage service is shutting down.
    #[must_use]
    pub fn is_stopping(&self) -> bool {
        self.token.is_cancelled()
    }

    #[must_use]
    pub const fn hooks(&self) -> &TestHooks {
        &self.hooks
    }

    /// Sends `event` to the tree's committer without waiting.
    ///
    /// If the committer has exited, the event is answered [`TreeOutcome::Untracked`] at once.
    ///
    /// # Returns
    ///
    /// The receiver of the event's reply.
    #[must_use]
    pub fn submit(&self, event: TreeEvent) -> oneshot::Receiver<TreeReply> {
        let (reply, receiver) = oneshot::channel();
        let job_id = event.job_id;
        let kind = EventKindTag::of(&event.kind);
        match self.sender.send(Pending { event, reply }) {
            Ok(()) => self.hooks.observe(HookPoint::EventSubmitted, || {
                HookContext::event(self.root_id, job_id, kind)
            }),
            Err(mpsc::error::SendError(pending)) => {
                // The send cannot fail: `receiver` is held until it is returned.
                let _ = pending.reply.send(Ok(TreeOutcome::Untracked));
            }
        }
        receiver
    }

    /// Submits `event` and waits for its reply, raced against the service's cancellation token.
    ///
    /// # Returns
    ///
    /// The committer's reply.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`CacheError::ServiceStopping`] if the service is shutting down.
    /// * [`InternalError::TreePoisoned`] if the committer dropped the event without a reply.
    pub async fn submit_and_wait(&self, event: TreeEvent) -> TreeReply {
        let receiver = self.submit(event);
        tokio::select! {
            reply = receiver => reply.unwrap_or_else(|_| Err(InternalError::TreePoisoned.into())),
            () = self.token.cancelled() => Err(CacheError::ServiceStopping),
        }
    }
}

/// A non-terminal job of a tree, as the planner sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeNode {
    pub parent: Option<JobId>,
    pub depth: u32,
    pub resource_group_id: ResourceGroupId,
    pub has_commit: bool,
    pub has_cleanup: bool,
    pub active_children: u32,
}

/// The structure of a tree's non-terminal jobs.
#[derive(Debug, Clone)]
pub struct TreeStructure {
    root_id: JobId,
    nodes: HashMap<JobId, TreeNode>,
}

impl TreeStructure {
    /// # Returns
    ///
    /// The structure of a tree that holds only its root.
    #[must_use]
    pub fn new_root(root_id: JobId, root: TreeNode) -> Self {
        Self {
            root_id,
            nodes: HashMap::from([(root_id, root)]),
        }
    }

    /// Factory function.
    ///
    /// Rebuilds the structure of the tree rooted at `root_id` from the rows of its jobs, given in
    /// ascending ID order. Only the non-terminal rows become nodes. A root's depth is 0, another
    /// node's depth is its parent's depth plus 1 if the parent is among `rows`, else 1, and a
    /// node's counter is its number of non-terminal children.
    ///
    /// # Returns
    ///
    /// The rebuilt structure.
    #[must_use]
    pub fn from_rows(root_id: JobId, rows: &[RecoveredRow]) -> Self {
        let mut depths: HashMap<JobId, u32> = HashMap::with_capacity(rows.len());
        let mut nodes = HashMap::with_capacity(rows.len());
        for row in rows {
            let depth = row.parent_id.map_or(0, |parent_id| {
                depths
                    .get(&parent_id)
                    .map_or(1, |parent_depth| parent_depth + 1)
            });
            depths.insert(row.id, depth);
            if row.state.is_terminal() {
                continue;
            }
            nodes.insert(
                row.id,
                TreeNode {
                    parent: row.parent_id,
                    depth,
                    resource_group_id: row.resource_group_id,
                    has_commit: row.has_commit,
                    has_cleanup: row.has_cleanup,
                    active_children: 0,
                },
            );
        }
        let parent_ids: Vec<JobId> = nodes.values().filter_map(|node| node.parent).collect();
        for parent_id in parent_ids {
            if let Some(parent) = nodes.get_mut(&parent_id) {
                parent.active_children += 1;
            }
        }
        Self { root_id, nodes }
    }

    #[must_use]
    pub const fn root_id(&self) -> JobId {
        self.root_id
    }

    #[must_use]
    pub fn node(&self, job_id: JobId) -> Option<&TreeNode> {
        self.nodes.get(&job_id)
    }

    /// # Returns
    ///
    /// The IDs of the tree's non-terminal jobs.
    pub fn job_ids(&self) -> impl Iterator<Item = JobId> + '_ {
        self.nodes.keys().copied()
    }

    /// # Returns
    ///
    /// The IDs of the tree's non-terminal jobs, in ascending order.
    #[must_use]
    pub fn sorted_job_ids(&self) -> Vec<JobId> {
        let mut job_ids: Vec<JobId> = self.job_ids().collect();
        job_ids.sort_unstable_by_key(JobId::get);
        job_ids
    }

    /// # Returns
    ///
    /// The IDs of `job_id`'s proper ancestors that are in the tree, nearest first.
    pub fn ancestors(&self, job_id: JobId) -> impl Iterator<Item = JobId> + '_ {
        let parent_in_tree = |job_id: &JobId| {
            self.nodes
                .get(job_id)?
                .parent
                .filter(|parent_id| self.nodes.contains_key(parent_id))
        };
        std::iter::successors(parent_in_tree(&job_id), parent_in_tree)
    }

    /// # Returns
    ///
    /// The IDs of the tree's jobs whose states the planner reads to plan `events`: every job of the
    /// tree if an event can deactivate jobs anywhere, otherwise each event's job and its ancestors.
    pub fn jobs_to_collect<'event_lifetime>(
        &self,
        events: impl IntoIterator<Item = &'event_lifetime TreeEvent>,
    ) -> Vec<JobId> {
        let events: Vec<&TreeEvent> = events.into_iter().collect();
        if events.iter().any(|event| event.kind.reads_every_job()) {
            return self.job_ids().collect();
        }
        let mut job_ids = HashSet::new();
        for event in events {
            if self.nodes.contains_key(&event.job_id) {
                job_ids.insert(event.job_id);
                job_ids.extend(self.ancestors(event.job_id));
            }
        }
        job_ids.into_iter().collect()
    }

    /// Adds a non-terminal job to the tree.
    ///
    /// # Returns
    ///
    /// The node that `job_id` had before, if any.
    pub fn insert(&mut self, job_id: JobId, node: TreeNode) -> Option<TreeNode> {
        self.nodes.insert(job_id, node)
    }

    /// Sets the number of active children of `job_id`.
    ///
    /// # Panics
    ///
    /// Panics if `job_id` is not in the tree.
    pub fn set_active_children(&mut self, job_id: JobId, active_children: u32) {
        self.nodes
            .get_mut(&job_id)
            .expect("a job whose counter changes should be in the tree")
            .active_children = active_children;
    }

    /// Removes a job that has become terminal.
    ///
    /// # Returns
    ///
    /// The removed node, if it was in the tree.
    pub fn remove(&mut self, job_id: JobId) -> Option<TreeNode> {
        self.nodes.remove(&job_id)
    }

    /// # Returns
    ///
    /// Whether every job of the tree is terminal.
    #[must_use]
    pub fn is_quiescent(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// A job row loaded at startup, as the repair planner sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveredRow {
    pub id: JobId,
    pub parent_id: Option<JobId>,
    pub state: JobState,
    pub resource_group_id: ResourceGroupId,
    pub has_commit: bool,
    pub has_cleanup: bool,
}

impl From<&RecoverableJobContext> for RecoveredRow {
    fn from(context: &RecoverableJobContext) -> Self {
        let task_graph = context.submission.task_graph();
        Self {
            id: context.id,
            parent_id: context.parent_id,
            state: context.state,
            resource_group_id: context.resource_group_id,
            has_commit: task_graph.get_commit_task_descriptor().is_some(),
            has_cleanup: task_graph.get_cleanup_task_descriptor().is_some(),
        }
    }
}

/// What the planner needs to know about the children that a job's tasks registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildSetSummary {
    pub count: usize,
    pub bytes: u64,
}

/// The inputs that the committer collects for an event, besides the job states.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PlannerInput {
    /// The ready task indices of a `Ready` root.
    Start { ready_task_indices: Vec<TaskIndex> },

    /// The outputs of a `Running` job whose regular tasks are done, and the children its tasks
    /// registered, if any.
    RegularTasksDone {
        outputs: Vec<TaskOutput>,
        children: Option<ChildSetSummary>,
    },

    /// Nothing, because the event needs nothing or its job is no longer in the state that the
    /// event expects.
    #[default]
    Empty,
}

impl PlannerInput {
    /// # Returns
    ///
    /// The number of bytes of the outputs and the child blobs that the input carries.
    fn payload_bytes(&self) -> u64 {
        match self {
            Self::RegularTasksDone { outputs, children } => {
                outputs_bytes(outputs) + children.map_or(0, |children| children.bytes)
            }
            Self::Start { .. } | Self::Empty => 0,
        }
    }
}

/// An event and its collected input, as handed to [`plan_events`].
pub struct PlannerEvent<'event_lifetime> {
    pub event: &'event_lifetime TreeEvent,
    pub input: &'event_lifetime mut PlannerInput,
}

/// Ready work that a tree transition makes available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Publication {
    Ready {
        job_id: JobId,
        task_indices: Vec<TaskIndex>,
    },
    Commit(JobId),
    Cleanup(JobId),
}

impl Publication {
    #[must_use]
    pub const fn job_id(&self) -> JobId {
        match self {
            Self::Ready { job_id, .. } | Self::Commit(job_id) | Self::Cleanup(job_id) => *job_id,
        }
    }
}

/// One job's state change, as the cache applies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobTransition {
    pub job_id: JobId,
    pub from: JobState,
    pub to: JobState,
}

/// The children that a tree state transaction registers under one parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedRegistration {
    pub parent_id: JobId,
    pub root_id: JobId,
    pub resource_group_id: ResourceGroupId,
    pub child_depth: u32,
    pub children: ChildSetSummary,

    /// The index of the registering event in the round.
    pub event_index: usize,
}

/// The plan of one tree state transaction.
#[derive(Debug, Default)]
pub struct Plan {
    /// The guarded row updates to write, at most one per job.
    pub updates: Vec<JobStateUpdate>,

    /// The new `active_children` counters, by parent.
    pub counters: HashMap<JobId, u32>,

    /// The children to register, at most one parent's.
    pub registration: Option<PlannedRegistration>,

    /// The ready work to stage after the apply.
    pub publications: Vec<Publication>,

    /// The jobs that become terminal.
    pub terminated: Vec<JobId>,

    /// The outcome of every planned event, by its index in the round.
    pub outcomes: BTreeMap<usize, TreeOutcome>,

    /// The indices of the events left for the next round, in their original order.
    pub deferred: Vec<usize>,

    /// The state changes to apply to the cache, deepest job first and root last.
    pub apply_order: Vec<JobTransition>,

    /// The number of bytes of outputs and child blobs that the plan writes.
    pub payload_bytes: u64,
}

impl Plan {
    /// # Returns
    ///
    /// Whether the plan writes nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    /// # Returns
    ///
    /// Whether `event_plan`, which writes `payload_bytes`, can join this plan, with
    /// `reserved_bytes` held for a later event of the round:
    ///
    /// * It changes no job that this plan changes.
    /// * It writes no payload, or the payload carried so far, including `reserved_bytes`, is empty,
    ///   or the payload stays within [`MAX_ROUND_PAYLOAD`].
    fn admits(
        &self,
        overlay: &Overlay<'_>,
        event_plan: &EventPlan,
        payload_bytes: u64,
        reserved_bytes: u64,
    ) -> bool {
        if event_plan
            .updates
            .iter()
            .any(|update| overlay.planned.contains_key(&update.job_id))
        {
            return false;
        }
        let carried_bytes = self.payload_bytes + reserved_bytes;
        payload_bytes == 0
            || carried_bytes == 0
            || carried_bytes + payload_bytes <= MAX_ROUND_PAYLOAD
    }

    /// Adds the plan of the round's event at `index`, an event of `job_id` that writes
    /// `payload_bytes`, to this plan.
    ///
    /// # Panics
    ///
    /// Panics if the event registers children and `job_id` is not in the tree.
    fn merge(
        &mut self,
        structure: &TreeStructure,
        index: usize,
        job_id: JobId,
        event_plan: EventPlan,
        payload_bytes: u64,
    ) {
        if let Some(children) = event_plan.registration {
            let parent = structure
                .node(job_id)
                .expect("a registering job should be in the tree");
            self.registration = Some(PlannedRegistration {
                parent_id: job_id,
                root_id: structure.root_id(),
                resource_group_id: parent.resource_group_id,
                child_depth: parent.depth + 1,
                children,
                event_index: index,
            });
        }
        self.payload_bytes += payload_bytes;
        self.outcomes.insert(index, event_plan.outcome);
        self.merge_updates(
            event_plan.updates,
            event_plan.counters,
            event_plan.publications,
        );
    }

    /// Adds `updates`, with their apply order and the jobs that they terminate, `counters` and
    /// `publications` to this plan.
    fn merge_updates(
        &mut self,
        updates: Vec<JobStateUpdate>,
        counters: HashMap<JobId, u32>,
        publications: Vec<Publication>,
    ) {
        for update in &updates {
            self.apply_order.push(JobTransition {
                job_id: update.job_id,
                from: update.from,
                to: update.to,
            });
            if update.to.is_terminal() {
                self.terminated.push(update.job_id);
            }
        }
        self.updates.extend(updates);
        self.counters.extend(counters);
        self.publications.extend(publications);
    }

    /// Sorts the apply order: the root last, and the other jobs deepest first.
    fn finalize(&mut self, structure: &TreeStructure) {
        let root_id = structure.root_id();
        self.apply_order.sort_by_key(|transition| {
            let depth = structure
                .node(transition.job_id)
                .map_or(0, |node| node.depth);
            (transition.job_id == root_id, std::cmp::Reverse(depth))
        });
    }
}

/// A tree invariant that a tree's structure and job states break.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invariant {invariant} broken at job {job_id}: {detail}")]
pub struct Violation {
    pub invariant: &'static str,
    pub job_id: JobId,
    pub detail: String,
}

impl Violation {
    fn new(invariant: &'static str, job_id: JobId, detail: impl Into<String>) -> Self {
        Self {
            invariant,
            job_id,
            detail: detail.into(),
        }
    }
}

/// Checks the invariants I1 to I4 on what a tree's structure can see: its non-terminal jobs with
/// their `states`. A job whose state is missing from `states` breaks the check.
///
/// Terminal jobs are not in the structure, so the part of I2 that needs them ("every child of a
/// `BlockingOnChildren` job is `Succeeded` or active") is checked only for the non-terminal
/// children.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`Violation`] for the first broken invariant found.
pub(crate) fn check_invariants(
    structure: &TreeStructure,
    states: &HashMap<JobId, JobState>,
) -> Result<(), Violation> {
    let checker = InvariantChecker { structure, states };
    checker.check_closed_tree()?;
    let mut num_children: HashMap<JobId, u32> = HashMap::new();
    for job_id in structure.sorted_job_ids() {
        if let Some(parent_id) = checker.check_job(job_id)?
            && structure.node(parent_id).is_some()
        {
            *num_children.entry(parent_id).or_default() += 1;
        }
    }
    checker.check_counters(&num_children)
}

/// Plans one round of events against the tree's structure and the collected job states.
///
/// The events are planned in this order: the first `CancelRoot`, then the others in their order
/// in `events`, then the first registering event. Each event is planned against the tree with the
/// plan of the earlier events applied. An event is deferred to the next round, and keeps its
/// input, if:
///
/// * it registers children and an earlier event of the round already does;
/// * it would change a job that an earlier event of the round changes;
/// * it would raise the round's payload above [`MAX_ROUND_PAYLOAD`], unless the round carries no
///   payload yet. If the oldest event other than the first `CancelRoot` registers children, its
///   payload counts from the start of the round, so that it is never deferred for the budget;
/// * it comes after the first [`MAX_EVENTS_PER_ROUND`] events; or
/// * an event of its job that comes earlier in `events` is deferred for one of the three reasons
///   above.
///
/// Every other event is planned, and takes its input out of its [`PlannerEvent`].
///
/// # Returns
///
/// The plan of the round.
///
/// # Panics
///
/// Panics if:
///
/// * The state of a job of the tree that an event reads was not collected.
/// * A `Ready` root's `Start`, or a `Running` job's `RegularTasksDone`, has no collected input.
/// * The tree breaks invariant I2 or I3: a failing job has an ancestor that is not
///   `BlockingOnChildren`, or a `BlockingOnChildren` parent counts no active child.
#[must_use]
pub(crate) fn plan_events(
    structure: &TreeStructure,
    collected: &HashMap<JobId, JobState>,
    events: &mut [PlannerEvent<'_>],
) -> Plan {
    let mut plan = Plan::default();
    let mut overlay = Overlay {
        structure,
        collected,
        planned: HashMap::new(),
        counters: HashMap::new(),
    };
    let round_order = RoundOrder::new(events);
    plan.deferred.extend(round_order.extra_registrations);
    let mut reserved_bytes = round_order
        .oldest_registration
        .map_or(0, |index| events[index].input.payload_bytes());
    let mut first_deferred_indices: HashMap<JobId, usize> = HashMap::new();

    for (position, index) in round_order.planned.into_iter().enumerate() {
        if Some(index) == round_order.oldest_registration {
            reserved_bytes = 0;
        }
        let event = events[index].event;
        let input = &mut *events[index].input;
        let follows_a_deferred_event = first_deferred_indices
            .get(&event.job_id)
            .is_some_and(|first_deferred_index| *first_deferred_index < index);
        let admitted = if position >= MAX_EVENTS_PER_ROUND || follows_a_deferred_event {
            None
        } else {
            let event_plan = plan_event(&overlay, event, input);
            let payload_bytes = event_plan.payload_bytes(input);
            plan.admits(&overlay, &event_plan, payload_bytes, reserved_bytes)
                .then_some((event_plan, payload_bytes))
        };
        let Some((event_plan, payload_bytes)) = admitted else {
            plan.deferred.push(index);
            let first_deferred_index = first_deferred_indices.entry(event.job_id).or_insert(index);
            *first_deferred_index = (*first_deferred_index).min(index);
            continue;
        };
        let event_plan = event_plan.with_input(event.job_id, std::mem::take(input));
        overlay.merge(&event_plan);
        plan.merge(structure, index, event.job_id, event_plan, payload_bytes);
    }
    plan.deferred.sort_unstable();
    plan.finalize(structure);
    plan
}

/// Plans the repair of a tree rebuilt at startup from its loaded rows, in one plan:
///
/// * R1: every `Ready`, `Running` or `BlockingOnChildren` job other than the root whose parent is
///   not `BlockingOnChildren` is aborted, in ascending ID order, so that its descendants are
///   aborted too.
/// * R4: every `BlockingOnChildren` job with a `CleanupReady` child, or in `failed_child` (a job
///   with a `Failed` or `Cancelled` child), fails with its blocking chain, and the rest of the tree
///   is aborted.
/// * R2: every `BlockingOnChildren` job left without an active child is unblocked, and the success
///   climbs.
/// * R5: if the tree still breaks an invariant, or if the root has a parent or a parent link does
///   not point to a smaller ID, the plan is replaced by one that fails every job of the tree, or
///   cancels it if it is `Ready`.
///
/// Each step is planned against the tree with the earlier steps applied. The plan answers no event,
/// and recovery uses only its updates.
///
/// # Returns
///
/// The repair plan, which writes nothing for a consistent tree.
///
/// # Panics
///
/// Panics if the state of a job of the tree is missing from `states`.
#[must_use]
pub(crate) fn plan_repair(
    structure: &TreeStructure,
    states: &HashMap<JobId, JobState>,
    failed_child: &HashSet<JobId>,
) -> Plan {
    let root_id = structure.root_id();
    let job_ids = structure.sorted_job_ids();
    for job_id in &job_ids {
        let parent = structure
            .node(*job_id)
            .expect("a listed job should be in the tree")
            .parent;
        if let Some(parent_id) = parent
            && (*job_id == root_id || parent_id.get() >= job_id.get())
        {
            let violation = Violation::new("I1", *job_id, format!("parent link to {parent_id}"));
            return plan_tree_failure(structure, states, &violation);
        }
    }

    let mut repair = Repair {
        overlay: Overlay {
            structure,
            collected: states,
            planned: HashMap::new(),
            counters: HashMap::new(),
        },
        plan: Plan::default(),
    };
    for job_id in &job_ids {
        let overlay = &repair.overlay;
        let is_orphan = *job_id != root_id
            && matches!(
                overlay.state(*job_id),
                NodeState::Present(
                    JobState::Ready | JobState::Running | JobState::BlockingOnChildren
                )
            )
            && !overlay
                .parent(*job_id)
                .is_some_and(|parent_id| overlay.is(parent_id, JobState::BlockingOnChildren));
        if is_orphan {
            let mut event_plan =
                EventPlan::new(TreeOutcome::Applied(overlay.abort_target(*job_id)));
            deactivate(overlay, &mut event_plan, *job_id);
            repair.merge(event_plan);
        }
    }

    let parents_of_cleanup_ready: HashSet<JobId> = job_ids
        .iter()
        .filter(|job_id| repair.overlay.is(**job_id, JobState::CleanupReady))
        .filter_map(|job_id| repair.overlay.parent(*job_id))
        .collect();
    for job_id in &job_ids {
        if repair.overlay.is(*job_id, JobState::BlockingOnChildren)
            && (failed_child.contains(job_id) || parents_of_cleanup_ready.contains(job_id))
        {
            let event_plan = fail_and_abort(
                &repair.overlay,
                *job_id,
                JobState::BlockingOnChildren,
                "tree repair: a child job failed or was cancelled",
            );
            repair.merge(event_plan);
        }
    }

    for job_id in &job_ids {
        if repair.overlay.is(*job_id, JobState::BlockingOnChildren)
            && repair.overlay.active_children(*job_id) == 0
        {
            let mut event_plan = EventPlan::new(TreeOutcome::Applied(JobState::Succeeded));
            unblock(&repair.overlay, &mut event_plan, *job_id);
            repair.merge(event_plan);
        }
    }

    let Repair { overlay, mut plan } = repair;
    let mut repaired_structure = structure.clone();
    for (job_id, active_children) in &plan.counters {
        repaired_structure.set_active_children(*job_id, *active_children);
    }
    for job_id in &plan.terminated {
        repaired_structure.remove(*job_id);
    }
    let mut repaired_states = states.clone();
    repaired_states.extend(overlay.planned);
    if let Err(violation) = check_invariants(&repaired_structure, &repaired_states) {
        return plan_tree_failure(structure, states, &violation);
    }
    plan.finalize(structure);
    plan
}

/// The maximum number of characters of a failure message that the failed ancestors' error
/// repeats.
const MAX_PROPAGATED_MESSAGE_CHARS: usize = 1024;

/// A repair being planned: the tree with the planned steps applied, and the plan so far.
struct Repair<'plan_lifetime> {
    overlay: Overlay<'plan_lifetime>,
    plan: Plan,
}

impl Repair<'_> {
    /// Adds the plan of one repair step to the plan, and applies it to the overlay.
    fn merge(&mut self, event_plan: EventPlan) {
        self.overlay.merge(&event_plan);
        self.plan.merge_updates(
            event_plan.updates,
            event_plan.counters,
            event_plan.publications,
        );
    }
}

/// Checks the invariants of a tree's structure with its jobs' states, for [`check_invariants`].
struct InvariantChecker<'tree_lifetime> {
    structure: &'tree_lifetime TreeStructure,
    states: &'tree_lifetime HashMap<JobId, JobState>,
}

impl InvariantChecker<'_> {
    /// # Returns
    ///
    /// The state of `job_id` if it is in the tree.
    fn state_in_tree(&self, job_id: JobId) -> Option<JobState> {
        self.structure
            .node(job_id)
            .and_then(|_| self.states.get(&job_id).copied())
    }

    /// Checks I4: if the root is `CleanupReady` or terminal, no job of the tree is abortable.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`Violation`] if the tree is closed and a job is `Ready`, `Running` or
    ///   `BlockingOnChildren`.
    fn check_closed_tree(&self) -> Result<(), Violation> {
        if matches!(
            self.state_in_tree(self.structure.root_id()),
            Some(
                JobState::Ready
                    | JobState::Running
                    | JobState::BlockingOnChildren
                    | JobState::CommitReady
            )
        ) {
            return Ok(());
        }
        for job_id in self.structure.sorted_job_ids() {
            if matches!(
                self.states.get(&job_id),
                Some(JobState::Ready | JobState::Running | JobState::BlockingOnChildren)
            ) {
                return Err(Violation::new(
                    "I4",
                    job_id,
                    "an abortable job in a closed tree",
                ));
            }
        }
        Ok(())
    }

    /// Checks what I1, I2, I3 and I5 require of the tree's job `job_id` and its ancestors.
    ///
    /// # Returns
    ///
    /// The job's parent, if any, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`Violation`] for the first broken invariant found.
    ///
    /// # Panics
    ///
    /// Panics if `job_id` is not in the tree.
    fn check_job(&self, job_id: JobId) -> Result<Option<JobId>, Violation> {
        let node = self
            .structure
            .node(job_id)
            .expect("a checked job should be in the tree");
        let Some(state) = self.states.get(&job_id).copied() else {
            return Err(Violation::new(
                "I5",
                job_id,
                "no state for a job of the tree",
            ));
        };
        if state.is_terminal() {
            return Err(Violation::new(
                "I5",
                job_id,
                format!("{state} job in the tree"),
            ));
        }
        let is_root = job_id == self.structure.root_id();
        if is_root != node.parent.is_none() {
            return Err(Violation::new("I1", job_id, "only the root has no parent"));
        }
        if state == JobState::Ready && !is_root {
            return Err(Violation::new("I1", job_id, "a non-root job is Ready"));
        }
        let Some(parent_id) = node.parent else {
            return Ok(None);
        };
        if parent_id.get() >= job_id.get() {
            return Err(Violation::new(
                "I1",
                job_id,
                format!("parent {parent_id} has a larger id"),
            ));
        }
        let parent_state = self.state_in_tree(parent_id);
        if parent_state == Some(JobState::BlockingOnChildren) && state == JobState::CleanupReady {
            return Err(Violation::new(
                "I2",
                job_id,
                "a CleanupReady child of a blocking parent",
            ));
        }
        match state {
            JobState::Running | JobState::BlockingOnChildren => {
                for ancestor_id in std::iter::successors(Some(parent_id), |ancestor_id| {
                    self.structure
                        .node(*ancestor_id)
                        .and_then(|node| node.parent)
                }) {
                    if self.state_in_tree(ancestor_id) != Some(JobState::BlockingOnChildren) {
                        return Err(Violation::new(
                            "I3",
                            job_id,
                            format!("ancestor {ancestor_id} of an active job is not blocking"),
                        ));
                    }
                }
            }
            JobState::CommitReady
                if !matches!(
                    parent_state,
                    None | Some(JobState::BlockingOnChildren | JobState::CleanupReady)
                ) =>
            {
                return Err(Violation::new(
                    "I3",
                    job_id,
                    format!("a CommitReady child of an active {parent_state:?} parent"),
                ));
            }
            _ => {}
        }
        Ok(Some(parent_id))
    }

    /// Checks the counters of I2: a `BlockingOnChildren` job has at least one active child, and its
    /// counter equals its number of children in `num_children`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`Violation`] for the first `BlockingOnChildren` job whose counter is wrong.
    fn check_counters(&self, num_children: &HashMap<JobId, u32>) -> Result<(), Violation> {
        for job_id in self.structure.sorted_job_ids() {
            let node = self
                .structure
                .node(job_id)
                .expect("a listed job should be in the tree");
            if self.states.get(&job_id) != Some(&JobState::BlockingOnChildren) {
                continue;
            }
            let active_children = num_children.get(&job_id).copied().unwrap_or_default();
            if active_children == 0 || node.active_children != active_children {
                return Err(Violation::new(
                    "I2",
                    job_id,
                    format!(
                        "{active_children} active children, counter {}",
                        node.active_children
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// A job's state as the planner reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    /// The job is a non-terminal job of the tree, or a job that the round already changes.
    Present(JobState),

    /// The job is not in the tree: it became terminal in an earlier tree state transaction.
    Absent,
}

/// The plan of one event.
struct EventPlan {
    updates: Vec<JobStateUpdate>,
    counters: HashMap<JobId, u32>,
    registration: Option<ChildSetSummary>,
    publications: Vec<Publication>,

    /// Whether the update of the event's own job stores the outputs of the event's input.
    stores_outputs: bool,

    outcome: TreeOutcome,
}

impl EventPlan {
    /// # Returns
    ///
    /// A plan that writes nothing yet and answers the event with `outcome`.
    fn new(outcome: TreeOutcome) -> Self {
        Self {
            updates: Vec::new(),
            counters: HashMap::new(),
            registration: None,
            publications: Vec::new(),
            stores_outputs: false,
            outcome,
        }
    }

    /// # Returns
    ///
    /// A plan that writes nothing and answers the event with `error`.
    fn stale(error: StaleStateError) -> Self {
        Self::new(TreeOutcome::Stale(error))
    }

    /// Adds the guarded update of `job_id` from `from` to `to`.
    fn update(&mut self, job_id: JobId, from: JobState, to: JobState) {
        self.updates.push(JobStateUpdate::new(job_id, from, to));
    }

    /// Adds the guarded update of `job_id` from `from` to `Failed` with `message`.
    fn fail(&mut self, job_id: JobId, from: JobState, message: String) {
        self.updates.push(JobStateUpdate {
            error_message: Some(message),
            ..JobStateUpdate::new(job_id, from, JobState::Failed)
        });
    }

    /// # Returns
    ///
    /// The number of bytes that the plan writes, given the event's `input`.
    fn payload_bytes(&self, input: &PlannerInput) -> u64 {
        let stored_outputs_bytes = match input {
            PlannerInput::RegularTasksDone { outputs, .. } if self.stores_outputs => {
                outputs_bytes(outputs)
            }
            _ => 0,
        };
        stored_outputs_bytes + self.registration.map_or(0, |children| children.bytes)
    }

    /// # Returns
    ///
    /// This plan, with the outputs of `input` moved into the update of the event's job `job_id` if
    /// the plan stores them.
    ///
    /// # Panics
    ///
    /// Panics if the plan stores outputs but has no update of `job_id`.
    fn with_input(mut self, job_id: JobId, input: PlannerInput) -> Self {
        if let PlannerInput::RegularTasksDone { outputs, .. } = input
            && self.stores_outputs
        {
            self.updates
                .iter_mut()
                .find(|update| update.job_id == job_id)
                .expect("a plan that stores outputs should update the event's job")
                .outputs = Some(outputs);
        }
        self
    }
}

/// The tree as the planner reads it: the structure and the collected job states, with the plan
/// of the round's earlier events applied.
struct Overlay<'plan_lifetime> {
    structure: &'plan_lifetime TreeStructure,
    collected: &'plan_lifetime HashMap<JobId, JobState>,
    planned: HashMap<JobId, JobState>,
    counters: HashMap<JobId, u32>,
}

impl Overlay<'_> {
    /// # Returns
    ///
    /// The state of `job_id`: as the round already changes it, else as collected, else
    /// [`NodeState::Absent`] if the job is not in the tree.
    ///
    /// # Panics
    ///
    /// Panics if `job_id` is in the tree but its state was not collected.
    fn state(&self, job_id: JobId) -> NodeState {
        if let Some(state) = self.planned.get(&job_id) {
            return NodeState::Present(*state);
        }
        if self.structure.node(job_id).is_none() {
            return NodeState::Absent;
        }
        NodeState::Present(
            *self
                .collected
                .get(&job_id)
                .expect("the state of every job that the planner reads should be collected"),
        )
    }

    /// # Returns
    ///
    /// The state of `job_id`, which is in the tree.
    ///
    /// # Panics
    ///
    /// Panics if `job_id` is not in the tree, or if [`Self::state`] panics.
    fn job_state(&self, job_id: JobId) -> JobState {
        let NodeState::Present(state) = self.state(job_id) else {
            panic!("job {job_id} should be in the tree");
        };
        state
    }

    /// # Returns
    ///
    /// Whether `job_id` is in the tree and in `state`.
    ///
    /// # Panics
    ///
    /// Panics if [`Self::state`] panics.
    fn is(&self, job_id: JobId, state: JobState) -> bool {
        self.state(job_id) == NodeState::Present(state)
    }

    /// # Returns
    ///
    /// The tree node of `job_id`.
    ///
    /// # Panics
    ///
    /// Panics if `job_id` is not in the tree.
    fn node(&self, job_id: JobId) -> &TreeNode {
        self.structure
            .node(job_id)
            .expect("every job that the planner changes should be in the tree")
    }

    /// # Returns
    ///
    /// The parent of `job_id`, or `None` if `job_id` is the root or is not in the tree.
    fn parent(&self, job_id: JobId) -> Option<JobId> {
        self.structure.node(job_id)?.parent
    }

    /// # Returns
    ///
    /// The number of active children of `job_id`, as the round already changes it.
    ///
    /// # Panics
    ///
    /// Panics if [`Self::node`] panics.
    fn active_children(&self, job_id: JobId) -> u32 {
        self.counters
            .get(&job_id)
            .copied()
            .unwrap_or_else(|| self.node(job_id).active_children)
    }

    /// # Returns
    ///
    /// Whether a cancel or an abort deactivates `job_id`: it is `Running` or
    /// `BlockingOnChildren`.
    ///
    /// # Panics
    ///
    /// Panics if [`Self::state`] panics.
    fn is_abortable(&self, job_id: JobId) -> bool {
        matches!(
            self.state(job_id),
            NodeState::Present(JobState::Running | JobState::BlockingOnChildren)
        )
    }

    /// # Returns
    ///
    /// The state that a cancel or an abort moves `job_id` to.
    ///
    /// # Panics
    ///
    /// Panics if [`Self::state`] or [`Self::node`] panics.
    fn abort_target(&self, job_id: JobId) -> JobState {
        if self.is(job_id, JobState::Ready) || !self.node(job_id).has_cleanup {
            JobState::Cancelled
        } else {
            JobState::CleanupReady
        }
    }

    /// # Returns
    ///
    /// The ancestors of `job_id`, nearest first.
    ///
    /// # Panics
    ///
    /// Panics if an ancestor is not `BlockingOnChildren` (invariant I3), or if [`Self::state`]
    /// panics.
    fn blocking_ancestors(&self, job_id: JobId) -> Vec<JobId> {
        std::iter::successors(self.parent(job_id), |ancestor_id| self.parent(*ancestor_id))
            .inspect(|ancestor_id| {
                assert!(
                    self.is(*ancestor_id, JobState::BlockingOnChildren),
                    "ancestor {ancestor_id} of an active job should be blocking on its children"
                );
            })
            .collect()
    }

    /// Applies an admitted event's plan, so that later events of the round read its result.
    fn merge(&mut self, event_plan: &EventPlan) {
        for update in &event_plan.updates {
            self.planned.insert(update.job_id, update.to);
        }
        self.counters.extend(&event_plan.counters);
    }
}

/// The order in which [`plan_events`] plans a round's events.
struct RoundOrder {
    /// The indices of the events to plan, in planning order: the first `CancelRoot`, then the
    /// others in their order, then the first registering event.
    planned: Vec<usize>,

    /// The indices of the registering events after the first, which wait for a later round.
    extra_registrations: Vec<usize>,

    /// The index of the first registering event, if it is the oldest event of the round other than
    /// the first `CancelRoot`.
    oldest_registration: Option<usize>,
}

impl RoundOrder {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// The planning order of `events`.
    fn new(events: &[PlannerEvent<'_>]) -> Self {
        let cancel_index = events
            .iter()
            .position(|planner_event| planner_event.event.kind == TreeEventKind::CancelRoot);
        let oldest_index = (0..events.len()).find(|index| Some(*index) != cancel_index);
        let mut planned: Vec<usize> = cancel_index.into_iter().collect();
        let mut registration_index = None;
        let mut extra_registrations = Vec::new();
        for (index, planner_event) in events.iter().enumerate() {
            if Some(index) == cancel_index {
                continue;
            }
            if planner_event.event.kind.registers() {
                if registration_index.is_none() {
                    registration_index = Some(index);
                } else {
                    extra_registrations.push(index);
                }
                continue;
            }
            planned.push(index);
        }
        planned.extend(registration_index);
        Self {
            planned,
            extra_registrations,
            oldest_registration: registration_index.filter(|index| Some(*index) == oldest_index),
        }
    }
}

/// # Returns
///
/// The plan of `event`, given its collected `input`. An event whose job is no longer in the tree
/// is [`TreeOutcome::Untracked`].
///
/// # Panics
///
/// Panics if the planner of the event's kind panics.
fn plan_event(overlay: &Overlay<'_>, event: &TreeEvent, input: &PlannerInput) -> EventPlan {
    let job_id = event.job_id;
    if overlay.state(job_id) == NodeState::Absent {
        return EventPlan::new(TreeOutcome::Untracked);
    }
    match &event.kind {
        TreeEventKind::Start => plan_start(overlay, job_id, input),
        TreeEventKind::RegularTasksDone { .. } => plan_regular_done(overlay, job_id, input),
        TreeEventKind::CommitDone => plan_commit_done(overlay, job_id),
        TreeEventKind::CleanupDone => plan_cleanup_done(overlay, job_id),
        TreeEventKind::TerminalFailure { lane, message } => {
            plan_failure(overlay, job_id, *lane, message)
        }
        TreeEventKind::CancelRoot => plan_cancel(overlay, job_id),
    }
}

/// # Returns
///
/// The plan of starting the root: `Ready → Running` with its ready tasks published, or stale if
/// the root is not `Ready`.
///
/// # Panics
///
/// Panics if [`Overlay::job_state`] panics, or if a `Ready` root's `input` carries no ready task
/// indices.
fn plan_start(overlay: &Overlay<'_>, root_id: JobId, input: &PlannerInput) -> EventPlan {
    let state = overlay.job_state(root_id);
    if state != JobState::Ready {
        return EventPlan::stale(TreeEventKind::Start.stale_error(state));
    }
    let PlannerInput::Start { ready_task_indices } = input else {
        panic!("a ready root's start should carry its ready task indices");
    };
    let mut plan = EventPlan::new(TreeOutcome::Applied(JobState::Running));
    plan.update(root_id, JobState::Ready, JobState::Running);
    plan.publications.push(Publication::Ready {
        job_id: root_id,
        task_indices: ready_task_indices.clone(),
    });
    plan
}

/// # Returns
///
/// The plan of the job's regular tasks being done, or stale if the job is not `Running`:
///
/// * Without children, the job succeeds as [`plan_success`] plans it.
/// * With children over [`MAX_CHILDREN_PER_JOB`] or [`MAX_JOB_CHILD_BYTES`], the job fails as
///   [`fail_and_abort`] plans it.
/// * Otherwise the job moves to `BlockingOnChildren` with its outputs, and registers its children.
///
/// # Panics
///
/// Panics if [`Overlay::job_state`], [`plan_success`] or [`fail_and_abort`] panics, or if a
/// `Running` job's `input` carries no outputs.
fn plan_regular_done(overlay: &Overlay<'_>, job_id: JobId, input: &PlannerInput) -> EventPlan {
    let state = overlay.job_state(job_id);
    if state != JobState::Running {
        return EventPlan::stale(Lane::Regular.stale_error(state));
    }
    let PlannerInput::RegularTasksDone { children, .. } = input else {
        panic!("a running job's regular-tasks-done event should carry its outputs");
    };
    let children = match children {
        Some(children) if children.count > 0 => *children,
        _ => return plan_success(overlay, job_id, true),
    };
    if children.count > MAX_CHILDREN_PER_JOB || children.bytes > MAX_JOB_CHILD_BYTES {
        return fail_and_abort(
            overlay,
            job_id,
            JobState::Running,
            "child-job limit exceeded",
        );
    }
    let mut plan = EventPlan::new(TreeOutcome::Applied(JobState::BlockingOnChildren));
    plan.update(job_id, JobState::Running, JobState::BlockingOnChildren);
    plan.stores_outputs = true;
    plan.counters.insert(
        job_id,
        u32::try_from(children.count).expect("a child count within the limit should fit a u32"),
    );
    plan.registration = Some(children);
    plan
}

/// # Returns
///
/// The plan of the job succeeding from its current state: `CommitReady` with the commit task
/// published if the plan stores the job's outputs and the job has a commit task, otherwise
/// `Succeeded`, after which the parent is decremented and the success climbs as
/// [`climb_after_success`] plans it.
///
/// # Panics
///
/// Panics if [`Overlay::job_state`], [`Overlay::node`] or [`climb_after_success`] panics.
fn plan_success(overlay: &Overlay<'_>, job_id: JobId, stores_outputs: bool) -> EventPlan {
    let from = overlay.job_state(job_id);
    let to = if stores_outputs && overlay.node(job_id).has_commit {
        JobState::CommitReady
    } else {
        JobState::Succeeded
    };
    let mut plan = EventPlan::new(TreeOutcome::Applied(to));
    plan.update(job_id, from, to);
    plan.stores_outputs = stores_outputs;
    if to == JobState::CommitReady {
        plan.publications.push(Publication::Commit(job_id));
    } else {
        climb_after_success(overlay, &mut plan, job_id);
    }
    plan
}

/// Plans the effect of `job_id` becoming `Succeeded` on its parent: if the parent is
/// `BlockingOnChildren`, its counter is decremented, and the parent is unblocked as [`unblock`]
/// plans it when the counter reaches 0. A parent in any other state, or one no longer in the
/// tree, stops the climb.
///
/// # Panics
///
/// Panics if a `BlockingOnChildren` parent counts no active child, or if [`unblock`] panics.
fn climb_after_success(overlay: &Overlay<'_>, plan: &mut EventPlan, job_id: JobId) {
    let Some(parent_id) = overlay.parent(job_id) else {
        return;
    };
    if !overlay.is(parent_id, JobState::BlockingOnChildren) {
        return;
    }
    let active_children = overlay
        .active_children(parent_id)
        .checked_sub(1)
        .expect("a blocking parent should count its active children");
    plan.counters.insert(parent_id, active_children);
    if active_children == 0 {
        unblock(overlay, plan, parent_id);
    }
}

/// Plans a `BlockingOnChildren` job whose children have all succeeded: `CommitReady` with the
/// commit task published if it has a commit task, otherwise `Succeeded`, and the climb continues.
///
/// # Panics
///
/// Panics if [`Overlay::node`] or [`climb_after_success`] panics.
fn unblock(overlay: &Overlay<'_>, plan: &mut EventPlan, job_id: JobId) {
    let to = if overlay.node(job_id).has_commit {
        JobState::CommitReady
    } else {
        JobState::Succeeded
    };
    plan.update(job_id, JobState::BlockingOnChildren, to);
    if to == JobState::CommitReady {
        plan.publications.push(Publication::Commit(job_id));
    } else {
        climb_after_success(overlay, plan, job_id);
    }
}

/// # Returns
///
/// The plan of the job's commit task being done, as [`plan_success`] plans it from
/// `CommitReady`, or stale if the job is not `CommitReady`.
///
/// # Panics
///
/// Panics if [`Overlay::job_state`] or [`plan_success`] panics.
fn plan_commit_done(overlay: &Overlay<'_>, job_id: JobId) -> EventPlan {
    let state = overlay.job_state(job_id);
    if state != JobState::CommitReady {
        return EventPlan::stale(Lane::Commit.stale_error(state));
    }
    plan_success(overlay, job_id, false)
}

/// # Returns
///
/// The plan of the job's cleanup task being done: `CleanupReady → Cancelled`, or stale if the job
/// is not `CleanupReady`.
///
/// # Panics
///
/// Panics if [`Overlay::job_state`] panics.
fn plan_cleanup_done(overlay: &Overlay<'_>, job_id: JobId) -> EventPlan {
    let state = overlay.job_state(job_id);
    if state != JobState::CleanupReady {
        return EventPlan::stale(Lane::Cleanup.stale_error(state));
    }
    let mut plan = EventPlan::new(TreeOutcome::Applied(JobState::Cancelled));
    plan.update(job_id, JobState::CleanupReady, JobState::Cancelled);
    plan
}

/// # Returns
///
/// The plan of a terminal failure on `lane`, as [`fail_and_abort`] plans it, or stale if `lane`
/// does not accept the job's state.
///
/// # Panics
///
/// Panics if [`Overlay::job_state`] or [`fail_and_abort`] panics.
fn plan_failure(overlay: &Overlay<'_>, job_id: JobId, lane: Lane, message: &str) -> EventPlan {
    let from = overlay.job_state(job_id);
    if !lane.accepts(from) {
        return EventPlan::stale(lane.stale_error(from));
    }
    fail_and_abort(overlay, job_id, from, message)
}

/// # Returns
///
/// The plan of the job failing from `from` with `message`, cut to [`MAX_ERROR_MESSAGE_CHARS`]
/// characters. If the job's parent is `BlockingOnChildren` and the failure is not a cleanup
/// failure, every ancestor fails too, with an error naming the job, and every other abortable job
/// of the tree is aborted as [`abort_rest`] plans it. A failure that does not propagate aborts the
/// rest only if the job was `BlockingOnChildren`.
///
/// # Panics
///
/// Panics if [`Overlay::blocking_ancestors`] or [`abort_rest`] panics.
fn fail_and_abort(
    overlay: &Overlay<'_>,
    job_id: JobId,
    from: JobState,
    message: &str,
) -> EventPlan {
    let mut plan = EventPlan::new(TreeOutcome::Applied(JobState::Failed));
    plan.fail(
        job_id,
        from,
        truncate_chars(message, MAX_ERROR_MESSAGE_CHARS).to_owned(),
    );
    let propagates = from != JobState::CleanupReady
        && overlay
            .parent(job_id)
            .is_some_and(|parent_id| overlay.is(parent_id, JobState::BlockingOnChildren));
    if !propagates {
        if from == JobState::BlockingOnChildren {
            abort_rest(overlay, &mut plan, job_id, &[]);
        }
        return plan;
    }
    let chain = overlay.blocking_ancestors(job_id);
    let chain_message = format!(
        "descendant job {job_id} failed: {}",
        truncate_chars(message, MAX_PROPAGATED_MESSAGE_CHARS)
    );
    for ancestor_id in &chain {
        plan.fail(
            *ancestor_id,
            JobState::BlockingOnChildren,
            chain_message.clone(),
        );
    }
    abort_rest(overlay, &mut plan, job_id, &chain);
    plan
}

/// Plans the abort of every abortable job of the tree other than `job_id` and `chain`: each moves
/// to its [`Overlay::abort_target`], with its cleanup task published if that is `CleanupReady`.
/// `CommitReady` jobs are left to finish.
///
/// # Panics
///
/// Panics if [`Overlay::state`] or [`Overlay::abort_target`] panics.
fn abort_rest(overlay: &Overlay<'_>, plan: &mut EventPlan, job_id: JobId, chain: &[JobId]) {
    for other_id in overlay.structure.sorted_job_ids() {
        if other_id == job_id || chain.contains(&other_id) || !overlay.is_abortable(other_id) {
            continue;
        }
        deactivate(overlay, plan, other_id);
    }
}

/// # Returns
///
/// The plan of cancelling the root: the root and every other abortable job of the tree move to
/// their [`Overlay::abort_target`], with their cleanup tasks published where that is
/// `CleanupReady`, or the cancel is stale if the root is `CleanupReady` or terminal. Non-root
/// `CommitReady` jobs are left to finish.
///
/// # Panics
///
/// Panics if:
///
/// * `root_id` is not the root of the tree.
/// * [`Overlay::job_state`] or [`Overlay::abort_target`] panics.
fn plan_cancel(overlay: &Overlay<'_>, root_id: JobId) -> EventPlan {
    assert_eq!(
        root_id,
        overlay.structure.root_id(),
        "a cancel must name the root of its tree"
    );
    let state = overlay.job_state(root_id);
    if state == JobState::CleanupReady || state.is_terminal() {
        return EventPlan::stale(TreeEventKind::CancelRoot.stale_error(state));
    }
    let mut plan = EventPlan::new(TreeOutcome::Applied(overlay.abort_target(root_id)));
    for job_id in overlay.structure.sorted_job_ids() {
        if job_id == root_id || overlay.is_abortable(job_id) {
            deactivate(overlay, &mut plan, job_id);
        }
    }
    plan
}

/// Plans the move of `job_id` to its [`Overlay::abort_target`], with its cleanup task published if
/// that is `CleanupReady`.
///
/// # Panics
///
/// Panics if [`Overlay::job_state`] or [`Overlay::abort_target`] panics.
fn deactivate(overlay: &Overlay<'_>, plan: &mut EventPlan, job_id: JobId) {
    let to = overlay.abort_target(job_id);
    plan.update(job_id, overlay.job_state(job_id), to);
    if to == JobState::CleanupReady {
        plan.publications.push(Publication::Cleanup(job_id));
    }
}

/// # Returns
///
/// The plan that ends every job of the tree because the tree breaks `violation`: a `Ready` job is
/// cancelled, and every other job fails with the violation as its error.
///
/// # Panics
///
/// Panics if the state of a job of the tree is missing from `states`.
fn plan_tree_failure(
    structure: &TreeStructure,
    states: &HashMap<JobId, JobState>,
    violation: &Violation,
) -> Plan {
    let message = format!("tree invariant violation: {violation}");
    let mut event_plan = EventPlan::new(TreeOutcome::Applied(JobState::Failed));
    for job_id in structure.sorted_job_ids() {
        let from = *states
            .get(&job_id)
            .expect("the state of every job of the tree should be loaded");
        if from == JobState::Ready {
            event_plan.update(job_id, from, JobState::Cancelled);
        } else {
            event_plan.fail(job_id, from, message.clone());
        }
    }
    let mut plan = Plan::default();
    plan.merge_updates(
        event_plan.updates,
        event_plan.counters,
        event_plan.publications,
    );
    plan.finalize(structure);
    plan
}

/// # Returns
///
/// The total number of bytes of `outputs`.
fn outputs_bytes(outputs: &[TaskOutput]) -> u64 {
    outputs.iter().map(|output| output.len() as u64).sum()
}

/// # Returns
///
/// The first `max_chars` characters of `message`.
fn truncate_chars(message: &str, max_chars: usize) -> &str {
    message
        .char_indices()
        .nth(max_chars)
        .map_or(message, |(end, _)| &message[..end])
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spider_core::job::JobState;
    use spider_core::types::id::JobId;
    use spider_core::types::id::ResourceGroupId;

    use super::*;

    const ROOT_ID: JobId = JobId::from(1);

    const ROOT_NODE: TreeNode = TreeNode {
        parent: None,
        depth: 0,
        resource_group_id: ResourceGroupId::from(1),
        has_commit: false,
        has_cleanup: false,
        active_children: 0,
    };

    /// A tree under test: its structure and the states the committer would collect.
    struct TestTree {
        structure: TreeStructure,
        states: HashMap<JobId, JobState>,
    }

    impl TestTree {
        fn new(root: TreeNode, root_state: JobState) -> Self {
            Self {
                structure: TreeStructure::new_root(ROOT_ID, root),
                states: HashMap::from([(ROOT_ID, root_state)]),
            }
        }

        /// Adds job `id` under `parent` in `state`, and counts it as an active child of `parent`.
        ///
        /// # Returns
        ///
        /// The new node, for its flags to be set.
        fn add(&mut self, id: u64, parent: u64, state: JobState) -> &mut TreeNode {
            let parent_id = JobId::from(parent);
            let parent_node = self
                .structure
                .nodes
                .get_mut(&parent_id)
                .expect("the parent should be added first");
            parent_node.active_children += 1;
            let depth = parent_node.depth + 1;
            let job_id = JobId::from(id);
            self.structure.insert(
                job_id,
                TreeNode {
                    parent: Some(parent_id),
                    depth,
                    ..ROOT_NODE
                },
            );
            self.states.insert(job_id, state);
            self.structure
                .nodes
                .get_mut(&job_id)
                .expect("the job was just added")
        }

        fn check(&self) -> Result<(), Violation> {
            check_invariants(&self.structure, &self.states)
        }

        /// Plans `events` as one round, collecting every job's state.
        ///
        /// # Returns
        ///
        /// The plan, and the inputs left with the events after planning.
        fn plan(&self, events: Vec<(TreeEvent, PlannerInput)>) -> (Plan, Vec<PlannerInput>) {
            let (events, mut inputs): (Vec<TreeEvent>, Vec<PlannerInput>) =
                events.into_iter().unzip();
            let mut planner_events: Vec<PlannerEvent<'_>> = events
                .iter()
                .zip(inputs.iter_mut())
                .map(|(event, input)| PlannerEvent { event, input })
                .collect();
            let plan = plan_events(&self.structure, &self.states, &mut planner_events);
            drop(planner_events);
            (plan, inputs)
        }
    }

    fn event(id: u64, kind: TreeEventKind) -> TreeEvent {
        TreeEvent {
            job_id: JobId::from(id),
            kind,
        }
    }

    fn failure(id: u64, lane: Lane, message: &str) -> (TreeEvent, PlannerInput) {
        (
            event(
                id,
                TreeEventKind::TerminalFailure {
                    lane,
                    message: message.to_owned(),
                },
            ),
            PlannerInput::Empty,
        )
    }

    fn regular_done(id: u64, outputs: Vec<TaskOutput>) -> (TreeEvent, PlannerInput) {
        (
            event(id, TreeEventKind::RegularTasksDone { registers: false }),
            PlannerInput::RegularTasksDone {
                outputs,
                children: None,
            },
        )
    }

    fn registration(id: u64, children: ChildSetSummary) -> (TreeEvent, PlannerInput) {
        (
            event(id, TreeEventKind::RegularTasksDone { registers: true }),
            PlannerInput::RegularTasksDone {
                outputs: vec![vec![1]],
                children: Some(children),
            },
        )
    }

    fn cancel() -> (TreeEvent, PlannerInput) {
        (event(1, TreeEventKind::CancelRoot), PlannerInput::Empty)
    }

    /// # Returns
    ///
    /// The planned transitions as `(job, from, to)`, sorted by job.
    fn transitions(plan: &Plan) -> Vec<(u64, JobState, JobState)> {
        let mut transitions: Vec<_> = plan
            .updates
            .iter()
            .map(|update| (update.job_id.get(), update.from, update.to))
            .collect();
        transitions.sort_by_key(|transition| transition.0);
        transitions
    }

    /// # Returns
    ///
    /// The outcome of the event at `index` as text: `Applied(state)`, the stale error, or
    /// `Untracked`.
    fn outcome(plan: &Plan, index: usize) -> String {
        match plan
            .outcomes
            .get(&index)
            .expect("the event should be planned")
        {
            TreeOutcome::Applied(state) => format!("Applied({state})"),
            TreeOutcome::Stale(error) => error.to_string(),
            TreeOutcome::Untracked => "Untracked".to_owned(),
        }
    }

    /// # Returns
    ///
    /// The error message that the plan's update of job `id` stores, if any.
    fn error_of(plan: &Plan, id: u64) -> Option<&str> {
        plan.updates
            .iter()
            .find(|update| update.job_id == JobId::from(id))
            .and_then(|update| update.error_message.as_deref())
    }

    #[test]
    fn start_moves_a_ready_root_to_running_and_publishes_its_ready_tasks() {
        let tree = TestTree::new(ROOT_NODE, JobState::Ready);
        let (plan, inputs) = tree.plan(vec![(
            event(1, TreeEventKind::Start),
            PlannerInput::Start {
                ready_task_indices: vec![0, 2],
            },
        )]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::Ready, JobState::Running)]
        );
        assert_eq!(outcome(&plan, 0), "Applied(Running)");
        assert_eq!(
            plan.publications,
            vec![Publication::Ready {
                job_id: ROOT_ID,
                task_indices: vec![0, 2],
            }]
        );
        assert_eq!(inputs, vec![PlannerInput::Empty]);
    }

    #[test]
    fn start_of_a_root_that_is_not_ready_is_stale() {
        for state in [
            JobState::Running,
            JobState::Cancelled,
            JobState::CleanupReady,
        ] {
            let tree = TestTree::new(ROOT_NODE, state);
            let (plan, _) = tree.plan(vec![(event(1, TreeEventKind::Start), PlannerInput::Empty)]);
            assert!(plan.is_empty());
            assert_eq!(
                outcome(&plan, 0),
                StaleStateError::JobAlreadyStarted.to_string()
            );
        }
    }

    #[test]
    fn regular_tasks_done_moves_to_commit_ready_or_succeeded_with_the_outputs() {
        let outputs = vec![vec![1u8, 2, 3]];
        for (has_commit, to, publications) in [
            (
                true,
                JobState::CommitReady,
                vec![Publication::Commit(ROOT_ID)],
            ),
            (false, JobState::Succeeded, Vec::new()),
        ] {
            let tree = TestTree::new(
                TreeNode {
                    has_commit,
                    ..ROOT_NODE
                },
                JobState::Running,
            );
            let (plan, _) = tree.plan(vec![regular_done(1, outputs.clone())]);
            assert_eq!(transitions(&plan), vec![(1, JobState::Running, to)]);
            assert_eq!(plan.updates[0].outputs, Some(outputs.clone()));
            assert_eq!(plan.payload_bytes, 3);
            assert_eq!(plan.publications, publications);
            assert_eq!(
                plan.terminated,
                if to.is_terminal() {
                    vec![ROOT_ID]
                } else {
                    Vec::new()
                }
            );
        }
    }

    #[test]
    fn regular_tasks_done_of_a_job_that_is_not_running_is_stale() {
        for (state, expected) in [
            (JobState::CleanupReady, StaleStateError::JobNoLongerRunning),
            (
                JobState::Cancelled,
                StaleStateError::JobAlreadyTerminated(JobState::Cancelled),
            ),
        ] {
            let tree = TestTree::new(ROOT_NODE, state);
            let (plan, _) = tree.plan(vec![(
                event(1, TreeEventKind::RegularTasksDone { registers: false }),
                PlannerInput::Empty,
            )]);
            assert!(plan.is_empty());
            assert_eq!(outcome(&plan, 0), expected.to_string());
        }
    }

    #[test]
    fn commit_and_cleanup_done_finish_the_job() {
        let tree = TestTree::new(
            TreeNode {
                has_commit: true,
                ..ROOT_NODE
            },
            JobState::CommitReady,
        );
        let (plan, _) = tree.plan(vec![(
            event(1, TreeEventKind::CommitDone),
            PlannerInput::Empty,
        )]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::CommitReady, JobState::Succeeded)]
        );
        assert_eq!(plan.updates[0].outputs, None);

        let tree = TestTree::new(ROOT_NODE, JobState::CleanupReady);
        let (plan, _) = tree.plan(vec![(
            event(1, TreeEventKind::CleanupDone),
            PlannerInput::Empty,
        )]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::CleanupReady, JobState::Cancelled)]
        );

        let tree = TestTree::new(ROOT_NODE, JobState::Cancelled);
        let (plan, _) = tree.plan(vec![(
            event(1, TreeEventKind::CommitDone),
            PlannerInput::Empty,
        )]);
        assert!(plan.is_empty());
    }

    #[test]
    fn failure_fails_the_job_only_in_its_lane() {
        let tree = TestTree::new(ROOT_NODE, JobState::CommitReady);
        let (plan, _) = tree.plan(vec![failure(1, Lane::Commit, "boom")]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::CommitReady, JobState::Failed)]
        );
        assert_eq!(error_of(&plan, 1), Some("boom"));

        let tree = TestTree::new(ROOT_NODE, JobState::CleanupReady);
        let (plan, _) = tree.plan(vec![failure(1, Lane::Regular, "late")]);
        assert!(plan.is_empty());
        assert_eq!(
            outcome(&plan, 0),
            StaleStateError::JobNoLongerRunning.to_string()
        );
    }

    #[test]
    fn cancel_deactivates_the_root_with_today_s_stale_errors() {
        let with_cleanup = TreeNode {
            has_cleanup: true,
            ..ROOT_NODE
        };
        let (plan, _) = TestTree::new(with_cleanup.clone(), JobState::Running).plan(vec![cancel()]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::Running, JobState::CleanupReady)]
        );
        assert_eq!(plan.publications, vec![Publication::Cleanup(ROOT_ID)]);

        let (plan, _) = TestTree::new(with_cleanup, JobState::Ready).plan(vec![cancel()]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::Ready, JobState::Cancelled)]
        );
        assert_eq!(plan.publications, Vec::new());

        let (plan, _) = TestTree::new(ROOT_NODE, JobState::CommitReady).plan(vec![cancel()]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::CommitReady, JobState::Cancelled)]
        );

        for (state, expected) in [
            (
                JobState::CleanupReady,
                StaleStateError::JobCancellationAlreadyRequested,
            ),
            (JobState::Cancelled, StaleStateError::JobAlreadyCancelled),
            (
                JobState::Succeeded,
                StaleStateError::JobAlreadyTerminated(JobState::Succeeded),
            ),
        ] {
            let (plan, _) = TestTree::new(ROOT_NODE, state).plan(vec![cancel()]);
            assert!(plan.is_empty());
            assert_eq!(outcome(&plan, 0), expected.to_string());
        }
    }

    #[test]
    fn two_failures_of_one_job_in_one_round_fail_it_once() {
        let tree = TestTree::new(ROOT_NODE, JobState::Running);
        let (plan, _) = tree.plan(vec![
            failure(1, Lane::Regular, "first"),
            failure(1, Lane::Regular, "second"),
        ]);
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::Running, JobState::Failed)]
        );
        assert_eq!(error_of(&plan, 1), Some("first"));
        assert_eq!(outcome(&plan, 0), "Applied(Failed)");
        assert_eq!(
            outcome(&plan, 1),
            StaleStateError::JobAlreadyTerminated(JobState::Failed).to_string()
        );
    }

    /// The tree R → P → {A, B}, R → Q, with every job active and B and Q defining a cleanup task.
    fn two_level_tree() -> TestTree {
        let mut tree = TestTree::new(ROOT_NODE, JobState::BlockingOnChildren);
        tree.add(2, 1, JobState::BlockingOnChildren);
        tree.add(3, 2, JobState::Running);
        tree.add(4, 2, JobState::Running).has_cleanup = true;
        tree.add(5, 1, JobState::Running).has_cleanup = true;
        tree.check()
            .expect("the test tree should hold the invariants");
        tree
    }

    #[test]
    fn two_siblings_failing_in_one_round_fail_the_chain_once() {
        let tree = two_level_tree();
        let (plan, _) = tree.plan(vec![
            failure(3, Lane::Regular, "a failed"),
            failure(4, Lane::Regular, "b failed"),
        ]);
        assert_eq!(
            transitions(&plan),
            vec![
                (1, JobState::BlockingOnChildren, JobState::Failed),
                (2, JobState::BlockingOnChildren, JobState::Failed),
                (3, JobState::Running, JobState::Failed),
                (4, JobState::Running, JobState::CleanupReady),
                (5, JobState::Running, JobState::CleanupReady),
            ]
        );
        assert_eq!(outcome(&plan, 0), "Applied(Failed)");
        assert_eq!(
            outcome(&plan, 1),
            StaleStateError::JobNoLongerRunning.to_string()
        );
        assert_eq!(error_of(&plan, 3), Some("a failed"));
        for ancestor in [1, 2] {
            assert_eq!(
                error_of(&plan, ancestor),
                Some("descendant job 3 failed: a failed")
            );
        }
        assert_eq!(
            plan.publications,
            vec![
                Publication::Cleanup(JobId::from(4)),
                Publication::Cleanup(JobId::from(5))
            ]
        );
        assert_eq!(
            plan.apply_order
                .iter()
                .map(|transition| transition.job_id.get())
                .collect::<Vec<_>>(),
            vec![3, 4, 2, 5, 1]
        );
    }

    #[test]
    fn a_propagated_failure_message_is_truncated() {
        let tree = two_level_tree();
        let message = "x".repeat(MAX_PROPAGATED_MESSAGE_CHARS + 10);
        let (plan, _) = tree.plan(vec![failure(3, Lane::Regular, &message)]);
        assert_eq!(error_of(&plan, 3), Some(message.as_str()));
        assert_eq!(
            error_of(&plan, 1),
            Some(
                format!(
                    "descendant job 3 failed: {}",
                    &message[..MAX_PROPAGATED_MESSAGE_CHARS]
                )
                .as_str()
            )
        );
    }

    #[test]
    fn a_failure_message_is_truncated() {
        let tree = two_level_tree();
        let message = "é".repeat(MAX_ERROR_MESSAGE_CHARS + 10);
        let (plan, _) = tree.plan(vec![failure(5, Lane::Regular, &message)]);
        assert_eq!(
            error_of(&plan, 5),
            Some("é".repeat(MAX_ERROR_MESSAGE_CHARS).as_str())
        );
    }

    #[test]
    fn a_cancel_is_planned_first_and_makes_a_registration_and_a_completion_stale() {
        let tree = two_level_tree();
        let (plan, inputs) = tree.plan(vec![
            registration(
                3,
                ChildSetSummary {
                    count: 2,
                    bytes: 10,
                },
            ),
            regular_done(4, vec![vec![4]]),
            cancel(),
        ]);
        assert_eq!(outcome(&plan, 2), "Applied(Cancelled)");
        assert_eq!(
            outcome(&plan, 0),
            StaleStateError::JobAlreadyTerminated(JobState::Cancelled).to_string()
        );
        assert_eq!(
            outcome(&plan, 1),
            StaleStateError::JobNoLongerRunning.to_string()
        );
        assert_eq!(plan.registration, None);
        assert_eq!(
            transitions(&plan),
            vec![
                (1, JobState::BlockingOnChildren, JobState::Cancelled),
                (2, JobState::BlockingOnChildren, JobState::Cancelled),
                (3, JobState::Running, JobState::Cancelled),
                (4, JobState::Running, JobState::CleanupReady),
                (5, JobState::Running, JobState::CleanupReady),
            ]
        );
        assert_eq!(inputs, vec![PlannerInput::Empty; 3]);
    }

    #[test]
    fn a_cancel_and_a_failure_in_one_round_cancel_the_tree_and_drop_the_failure() {
        let tree = TestTree::new(
            TreeNode {
                has_cleanup: true,
                ..ROOT_NODE
            },
            JobState::Running,
        );
        let (plan, _) = tree.plan(vec![failure(1, Lane::Regular, "boom"), cancel()]);
        assert_eq!(outcome(&plan, 1), "Applied(CleanupReady)");
        assert_eq!(
            outcome(&plan, 0),
            StaleStateError::JobNoLongerRunning.to_string()
        );
        assert_eq!(
            transitions(&plan),
            vec![(1, JobState::Running, JobState::CleanupReady)]
        );
        assert_eq!(error_of(&plan, 1), None);
    }

    #[test]
    fn only_the_oldest_registering_parent_registers_and_the_other_keeps_its_input() {
        let tree = two_level_tree();
        let second = registration(
            4,
            ChildSetSummary {
                count: 3,
                bytes: 30,
            },
        );
        let (plan, inputs) = tree.plan(vec![
            registration(
                3,
                ChildSetSummary {
                    count: 2,
                    bytes: 20,
                },
            ),
            second.clone(),
            regular_done(5, vec![vec![5]]),
        ]);
        assert_eq!(plan.deferred, vec![1]);
        assert_eq!(inputs[1], second.1);
        assert_eq!(outcome(&plan, 0), "Applied(BlockingOnChildren)");
        assert_eq!(outcome(&plan, 2), "Applied(Succeeded)");
        assert_eq!(
            plan.registration,
            Some(PlannedRegistration {
                parent_id: JobId::from(3),
                root_id: ROOT_ID,
                resource_group_id: ROOT_NODE.resource_group_id,
                child_depth: 3,
                children: ChildSetSummary {
                    count: 2,
                    bytes: 20,
                },
                event_index: 0,
            })
        );
        assert_eq!(
            plan.counters,
            HashMap::from([(JobId::from(3), 2), (ROOT_ID, 1)])
        );
        assert_eq!(
            transitions(&plan),
            vec![
                (3, JobState::Running, JobState::BlockingOnChildren),
                (5, JobState::Running, JobState::Succeeded),
            ]
        );
        assert_eq!(plan.updates[1].outputs, Some(vec![vec![1]]));
        assert_eq!(plan.payload_bytes, 1 + 20 + 1);
    }

    #[test]
    fn a_child_set_over_the_job_limits_fails_the_job() {
        for children in [
            ChildSetSummary {
                count: 2,
                bytes: MAX_JOB_CHILD_BYTES + 1,
            },
            ChildSetSummary {
                count: MAX_CHILDREN_PER_JOB + 1,
                bytes: 1,
            },
        ] {
            let tree = two_level_tree();
            let (plan, _) = tree.plan(vec![registration(3, children)]);
            assert_eq!(outcome(&plan, 0), "Applied(Failed)");
            assert_eq!(plan.registration, None);
            assert_eq!(error_of(&plan, 3), Some("child-job limit exceeded"));
            assert_eq!(plan.updates[0].outputs, None);
            assert_eq!(
                transitions(&plan),
                vec![
                    (1, JobState::BlockingOnChildren, JobState::Failed),
                    (2, JobState::BlockingOnChildren, JobState::Failed),
                    (3, JobState::Running, JobState::Failed),
                    (4, JobState::Running, JobState::CleanupReady),
                    (5, JobState::Running, JobState::CleanupReady),
                ]
            );
        }
    }

    #[test]
    fn siblings_completing_in_one_round_unblock_their_parent_once() {
        let mut tree = two_level_tree();
        tree.structure
            .nodes
            .get_mut(&JobId::from(2))
            .expect("P is in the tree")
            .has_commit = true;
        let (plan, _) = tree.plan(vec![
            regular_done(3, vec![vec![3]]),
            regular_done(4, vec![vec![4]]),
        ]);
        assert_eq!(outcome(&plan, 0), "Applied(Succeeded)");
        assert_eq!(outcome(&plan, 1), "Applied(Succeeded)");
        assert_eq!(
            transitions(&plan),
            vec![
                (2, JobState::BlockingOnChildren, JobState::CommitReady),
                (3, JobState::Running, JobState::Succeeded),
                (4, JobState::Running, JobState::Succeeded),
            ]
        );
        assert_eq!(plan.counters, HashMap::from([(JobId::from(2), 0)]));
        assert_eq!(plan.publications, vec![Publication::Commit(JobId::from(2))]);

        for job_id in [3, 4] {
            tree.structure.remove(JobId::from(job_id));
            tree.states.remove(&JobId::from(job_id));
        }
        tree.structure.set_active_children(JobId::from(2), 0);
        tree.states.insert(JobId::from(2), JobState::CommitReady);
        tree.check()
            .expect("the unblocked parent should hold the invariants");
        let (plan, _) = tree.plan(vec![(
            event(2, TreeEventKind::CommitDone),
            PlannerInput::Empty,
        )]);
        assert_eq!(
            transitions(&plan),
            vec![(2, JobState::CommitReady, JobState::Succeeded)]
        );
        assert_eq!(plan.counters, HashMap::from([(ROOT_ID, 1)]));
    }

    #[test]
    fn a_completion_cascades_to_the_root() {
        let mut tree = TestTree::new(ROOT_NODE, JobState::BlockingOnChildren);
        tree.add(2, 1, JobState::BlockingOnChildren);
        tree.add(3, 2, JobState::Running);
        let (plan, _) = tree.plan(vec![regular_done(3, vec![vec![3]])]);
        assert_eq!(
            transitions(&plan),
            vec![
                (1, JobState::BlockingOnChildren, JobState::Succeeded),
                (2, JobState::BlockingOnChildren, JobState::Succeeded),
                (3, JobState::Running, JobState::Succeeded),
            ]
        );
        assert_eq!(
            plan.apply_order
                .iter()
                .map(|transition| transition.job_id.get())
                .collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
        assert_eq!(plan.terminated.len(), 3);
    }

    #[test]
    fn a_skipped_commit_ready_job_whose_parent_is_absent_finishes_alone() {
        let mut tree = TestTree::new(ROOT_NODE, JobState::CleanupReady);
        tree.add(2, 1, JobState::BlockingOnChildren);
        tree.add(3, 2, JobState::CommitReady).has_commit = true;
        tree.structure.remove(JobId::from(2));
        tree.states.remove(&JobId::from(2));
        tree.check()
            .expect("a skipped job under an absent parent should hold the invariants");

        let (plan, _) = tree.plan(vec![(
            event(3, TreeEventKind::CommitDone),
            PlannerInput::Empty,
        )]);
        assert_eq!(
            transitions(&plan),
            vec![(3, JobState::CommitReady, JobState::Succeeded)]
        );
        assert_eq!(plan.counters, HashMap::new());

        let (plan, _) = tree.plan(vec![failure(3, Lane::Commit, "commit failed")]);
        assert_eq!(
            transitions(&plan),
            vec![(3, JobState::CommitReady, JobState::Failed)]
        );

        let (plan, _) = tree.plan(vec![(
            event(2, TreeEventKind::CleanupDone),
            PlannerInput::Empty,
        )]);
        assert!(plan.is_empty());
        assert_eq!(outcome(&plan, 0), "Untracked");
    }

    #[test]
    fn a_cancel_skips_a_non_root_commit_ready_job() {
        let mut tree = TestTree::new(ROOT_NODE, JobState::BlockingOnChildren);
        tree.add(2, 1, JobState::CommitReady).has_commit = true;
        tree.add(3, 1, JobState::Running);
        let (plan, _) = tree.plan(vec![cancel()]);
        assert_eq!(
            transitions(&plan),
            vec![
                (1, JobState::BlockingOnChildren, JobState::Cancelled),
                (3, JobState::Running, JobState::Cancelled),
            ]
        );
    }

    #[test]
    fn an_all_stale_round_gives_an_empty_plan() {
        let tree = TestTree::new(ROOT_NODE, JobState::CleanupReady);
        let (plan, _) = tree.plan(vec![
            (event(1, TreeEventKind::Start), PlannerInput::Empty),
            cancel(),
            (
                event(1, TreeEventKind::RegularTasksDone { registers: false }),
                PlannerInput::Empty,
            ),
            (event(1, TreeEventKind::CommitDone), PlannerInput::Empty),
        ]);
        assert!(plan.is_empty());
        assert_eq!(plan.outcomes.len(), 4);
        assert_eq!(plan.publications, Vec::new());
        assert_eq!(plan.apply_order, Vec::new());
    }

    #[test]
    fn an_event_that_would_change_a_job_again_is_deferred_with_its_input() {
        let tree = TestTree::new(
            TreeNode {
                has_commit: true,
                ..ROOT_NODE
            },
            JobState::Running,
        );
        let (plan, inputs) = tree.plan(vec![
            regular_done(1, vec![vec![1]]),
            failure(1, Lane::Commit, "commit failed"),
            (event(1, TreeEventKind::CleanupDone), PlannerInput::Empty),
        ]);
        assert_eq!(outcome(&plan, 0), "Applied(CommitReady)");
        assert_eq!(plan.deferred, vec![1, 2]);
        assert_eq!(plan.outcomes.len(), 1);
        assert_eq!(inputs[0], PlannerInput::Empty);
    }

    #[test]
    fn the_payload_budget_defers_events_but_always_admits_the_first() {
        let mut tree = TestTree::new(ROOT_NODE, JobState::BlockingOnChildren);
        for id in 2..=4 {
            tree.add(id, 1, JobState::Running);
        }
        let outputs_size = usize::try_from(MAX_ROUND_PAYLOAD / 3 + 1).expect("fits a usize");
        let (plan, inputs) = tree.plan(
            (2..=4)
                .map(|id| regular_done(id, vec![vec![0u8; outputs_size]]))
                .collect(),
        );
        assert_eq!(
            plan.outcomes.keys().copied().collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(plan.deferred, vec![2]);
        assert_eq!(plan.payload_bytes, 2 * outputs_size as u64);
        assert!(matches!(
            &inputs[2],
            PlannerInput::RegularTasksDone { outputs, .. } if outputs[0].len() == outputs_size
        ));

        let oversized = usize::try_from(MAX_ROUND_PAYLOAD + 1).expect("fits a usize");
        let (plan, _) = tree.plan(vec![
            (event(1, TreeEventKind::CommitDone), PlannerInput::Empty),
            regular_done(2, vec![vec![0u8; oversized]]),
        ]);
        assert_eq!(plan.deferred, Vec::<usize>::new());
        assert_eq!(plan.payload_bytes, oversized as u64);

        let (plan, _) = tree.plan(vec![
            regular_done(2, vec![vec![0u8; oversized]]),
            failure(3, Lane::Regular, "boom"),
        ]);
        assert_eq!(plan.deferred, Vec::<usize>::new());
        assert_eq!(outcome(&plan, 1), "Applied(Failed)");
    }

    #[test]
    fn a_registration_deferred_for_the_payload_budget_is_planned_in_the_next_round() {
        let tree = two_level_tree();
        let children = ChildSetSummary {
            count: 2,
            bytes: MAX_JOB_CHILD_BYTES,
        };
        let (plan, inputs) = tree.plan(vec![
            regular_done(5, vec![vec![5]]),
            registration(3, children),
        ]);
        assert_eq!(outcome(&plan, 0), "Applied(Succeeded)");
        assert_eq!(plan.deferred, vec![1]);
        assert_eq!(inputs[1], registration(3, children).1);

        let (plan, inputs) = tree.plan(vec![
            registration(3, children),
            regular_done(4, vec![vec![4]]),
        ]);
        assert_eq!(outcome(&plan, 0), "Applied(BlockingOnChildren)");
        assert_eq!(
            plan.registration.map(|registration| registration.parent_id),
            Some(JobId::from(3))
        );
        assert_eq!(plan.deferred, vec![1]);
        assert_eq!(inputs[1], regular_done(4, vec![vec![4]]).1);
        assert_eq!(plan.payload_bytes, MAX_JOB_CHILD_BYTES + 1);

        let (plan, _) = tree.plan(vec![registration(3, children), cancel()]);
        assert_eq!(outcome(&plan, 1), "Applied(Cancelled)");
        assert_eq!(plan.registration, None);

        let small_children = ChildSetSummary {
            count: 2,
            bytes: 1024,
        };
        let (plan, _) = tree.plan(vec![
            registration(3, small_children),
            regular_done(4, vec![vec![4]]),
        ]);
        assert_eq!(plan.deferred, Vec::<usize>::new());
        assert_eq!(outcome(&plan, 1), "Applied(Succeeded)");
    }

    #[test]
    fn events_past_the_round_cap_are_deferred() {
        let tree = TestTree::new(ROOT_NODE, JobState::Running);
        let (plan, _) = tree.plan(
            (0..MAX_EVENTS_PER_ROUND + 3)
                .map(|_| (event(1, TreeEventKind::Start), PlannerInput::Empty))
                .collect(),
        );
        assert_eq!(plan.outcomes.len(), MAX_EVENTS_PER_ROUND);
        assert_eq!(
            plan.deferred,
            (MAX_EVENTS_PER_ROUND..MAX_EVENTS_PER_ROUND + 3).collect::<Vec<_>>()
        );
    }

    #[test]
    fn jobs_to_collect_covers_what_the_planner_reads() {
        let tree = two_level_tree();
        let mut job_ids = tree
            .structure
            .jobs_to_collect([&event(3, TreeEventKind::CommitDone)]);
        job_ids.sort_unstable_by_key(JobId::get);
        assert_eq!(job_ids, vec![ROOT_ID, JobId::from(2), JobId::from(3)]);
        let job_ids = tree.structure.jobs_to_collect([
            &event(5, TreeEventKind::CleanupDone),
            &failure(4, Lane::Regular, "boom").0,
        ]);
        assert_eq!(job_ids.len(), 5);
    }

    #[test]
    fn check_invariants_reports_each_broken_invariant() {
        type Corruption = fn(&mut TestTree);
        let cases: Vec<(&str, Corruption)> = vec![
            ("I1", |tree| {
                tree.states.insert(JobId::from(3), JobState::Ready);
            }),
            ("I2", |tree| {
                tree.structure.set_active_children(JobId::from(2), 3);
            }),
            ("I2", |tree| {
                tree.states.insert(JobId::from(4), JobState::CleanupReady);
            }),
            ("I3", |tree| {
                tree.states.insert(JobId::from(2), JobState::Running);
            }),
            ("I3", |tree| {
                tree.states.insert(JobId::from(2), JobState::CommitReady);
                for job_id in [3, 4] {
                    tree.structure.remove(JobId::from(job_id));
                    tree.states.remove(&JobId::from(job_id));
                }
                tree.add(6, 2, JobState::CommitReady);
            }),
            ("I4", |tree| {
                tree.states.insert(ROOT_ID, JobState::CleanupReady);
            }),
            ("I5", |tree| {
                tree.states.insert(JobId::from(5), JobState::Failed);
            }),
        ];
        for (expected, corrupt) in cases {
            let mut tree = two_level_tree();
            corrupt(&mut tree);
            let violation = tree
                .check()
                .expect_err("the corrupted tree should break an invariant");
            assert_eq!(violation.invariant, expected, "{violation}");
        }
    }

    /// # Returns
    ///
    /// The loaded row of job `id` in `state`, under `parent` (0 for none), without a commit or a
    /// cleanup task.
    fn loaded(id: u64, parent: u64, state: JobState) -> RecoveredRow {
        RecoveredRow {
            id: JobId::from(id),
            parent_id: (parent != 0).then(|| JobId::from(parent)),
            state,
            resource_group_id: ResourceGroupId::from(1),
            has_commit: false,
            has_cleanup: false,
        }
    }

    /// # Returns
    ///
    /// The repair plan of the tree rooted at job 1 rebuilt from `rows`, where the jobs in
    /// `failed_child` have a failed or cancelled child.
    fn repair(rows: &[RecoveredRow], failed_child: &[u64]) -> Plan {
        let structure = TreeStructure::from_rows(ROOT_ID, rows);
        let states = rows.iter().map(|row| (row.id, row.state)).collect();
        let failed_child = failed_child.iter().copied().map(JobId::from).collect();
        plan_repair(&structure, &states, &failed_child)
    }

    #[test]
    fn from_rows_rebuilds_depths_counters_and_flags() {
        let rows = [
            loaded(1, 0, JobState::BlockingOnChildren),
            RecoveredRow {
                has_commit: true,
                ..loaded(2, 1, JobState::BlockingOnChildren)
            },
            loaded(3, 2, JobState::Running),
            RecoveredRow {
                has_cleanup: true,
                ..loaded(4, 2, JobState::CommitReady)
            },
            loaded(5, 1, JobState::Succeeded),
            loaded(6, 5, JobState::CommitReady),
            loaded(7, 9, JobState::Running),
        ];
        let structure = TreeStructure::from_rows(ROOT_ID, &rows);
        assert_eq!(
            structure.sorted_job_ids(),
            [1, 2, 3, 4, 6, 7].map(JobId::from).to_vec()
        );
        let node = |id: u64| {
            let node = structure
                .node(JobId::from(id))
                .expect("the job should be in the tree");
            (
                node.parent.as_ref().map(JobId::get),
                node.depth,
                node.has_commit,
                node.has_cleanup,
                node.active_children,
            )
        };
        assert_eq!(node(1), (None, 0, false, false, 1));
        assert_eq!(node(2), (Some(1), 1, true, false, 2));
        assert_eq!(node(3), (Some(2), 2, false, false, 0));
        assert_eq!(node(4), (Some(2), 2, false, true, 0));
        assert_eq!(node(6), (Some(5), 2, false, false, 0));
        assert_eq!(node(7), (Some(9), 1, false, false, 0));
    }

    #[test]
    fn a_consistent_tree_needs_no_repair() {
        let trees = [
            vec![loaded(1, 0, JobState::Ready)],
            vec![loaded(1, 0, JobState::CleanupReady)],
            vec![
                loaded(1, 0, JobState::BlockingOnChildren),
                loaded(2, 1, JobState::BlockingOnChildren),
                loaded(3, 2, JobState::Running),
                loaded(4, 2, JobState::CommitReady),
                loaded(5, 1, JobState::Running),
            ],
            vec![
                loaded(2, 1, JobState::CommitReady),
                loaded(3, 1, JobState::CleanupReady),
            ],
        ];
        for rows in trees {
            assert_eq!(transitions(&repair(&rows, &[])), [], "{rows:?}");
        }
    }

    #[test]
    fn an_orphan_is_aborted_with_its_descendants() {
        let rows = [
            RecoveredRow {
                has_cleanup: true,
                ..loaded(2, 1, JobState::BlockingOnChildren)
            },
            loaded(3, 2, JobState::Running),
            loaded(4, 2, JobState::CommitReady),
            loaded(5, 0, JobState::Running),
        ];
        let plan = repair(&rows, &[]);
        assert_eq!(
            transitions(&plan),
            [
                (2, JobState::BlockingOnChildren, JobState::CleanupReady),
                (3, JobState::Running, JobState::Cancelled),
                (5, JobState::Running, JobState::Cancelled),
            ]
        );
        assert_eq!(plan.terminated, [3, 5].map(JobId::from).to_vec());

        let rows = [
            loaded(1, 0, JobState::Running),
            loaded(2, 1, JobState::Ready),
        ];
        assert_eq!(
            transitions(&repair(&rows, &[])),
            [(2, JobState::Ready, JobState::Cancelled)]
        );
    }

    #[test]
    fn a_stuck_parent_is_unblocked_and_its_success_climbs() {
        let rows = [
            loaded(1, 0, JobState::BlockingOnChildren),
            loaded(2, 1, JobState::BlockingOnChildren),
            loaded(3, 1, JobState::Running),
        ];
        let plan = repair(&rows, &[]);
        assert_eq!(
            transitions(&plan),
            [(2, JobState::BlockingOnChildren, JobState::Succeeded)]
        );
        assert_eq!(plan.counters, HashMap::from([(ROOT_ID, 1)]));

        let rows = [
            RecoveredRow {
                has_commit: true,
                ..loaded(1, 0, JobState::BlockingOnChildren)
            },
            loaded(2, 1, JobState::BlockingOnChildren),
            loaded(3, 2, JobState::BlockingOnChildren),
        ];
        let plan = repair(&rows, &[]);
        assert_eq!(
            transitions(&plan),
            [
                (1, JobState::BlockingOnChildren, JobState::CommitReady),
                (2, JobState::BlockingOnChildren, JobState::Succeeded),
                (3, JobState::BlockingOnChildren, JobState::Succeeded),
            ]
        );
        assert_eq!(plan.publications, [Publication::Commit(ROOT_ID)]);
    }

    #[test]
    fn a_blocking_parent_with_a_failed_child_fails_and_aborts_the_tree() {
        let rows = [
            loaded(1, 0, JobState::BlockingOnChildren),
            RecoveredRow {
                has_cleanup: true,
                ..loaded(2, 1, JobState::Running)
            },
            loaded(3, 1, JobState::BlockingOnChildren),
            loaded(4, 3, JobState::Running),
        ];
        let plan = repair(&rows, &[3]);
        assert_eq!(
            transitions(&plan),
            [
                (1, JobState::BlockingOnChildren, JobState::Failed),
                (2, JobState::Running, JobState::CleanupReady),
                (3, JobState::BlockingOnChildren, JobState::Failed),
                (4, JobState::Running, JobState::Cancelled),
            ]
        );
        assert_eq!(
            error_of(&plan, 3),
            Some("tree repair: a child job failed or was cancelled")
        );
        assert_eq!(
            error_of(&plan, 1),
            Some("descendant job 3 failed: tree repair: a child job failed or was cancelled")
        );

        let rows = [
            loaded(1, 0, JobState::BlockingOnChildren),
            loaded(2, 1, JobState::CleanupReady),
            loaded(3, 1, JobState::Running),
        ];
        assert_eq!(
            transitions(&repair(&rows, &[])),
            [
                (1, JobState::BlockingOnChildren, JobState::Failed),
                (3, JobState::Running, JobState::Cancelled),
            ]
        );
    }

    #[test]
    fn a_tree_that_repair_cannot_fix_fails_as_a_whole() {
        let rows = [
            loaded(1, 0, JobState::BlockingOnChildren),
            loaded(2, 1, JobState::Ready),
            loaded(3, 1, JobState::Running),
        ];
        let plan = repair(&rows, &[]);
        assert_eq!(
            transitions(&plan),
            [
                (1, JobState::BlockingOnChildren, JobState::Failed),
                (2, JobState::Ready, JobState::Cancelled),
                (3, JobState::Running, JobState::Failed),
            ]
        );
        assert_eq!(
            error_of(&plan, 3),
            Some("tree invariant violation: invariant I1 broken at job 2: a non-root job is Ready")
        );

        let rows = [
            loaded(1, 0, JobState::BlockingOnChildren),
            loaded(2, 3, JobState::Running),
            loaded(3, 1, JobState::BlockingOnChildren),
        ];
        let plan = repair(&rows, &[]);
        assert_eq!(
            transitions(&plan),
            [
                (1, JobState::BlockingOnChildren, JobState::Failed),
                (2, JobState::Running, JobState::Failed),
                (3, JobState::BlockingOnChildren, JobState::Failed),
            ]
        );
        assert_eq!(
            error_of(&plan, 1),
            Some("tree invariant violation: invariant I1 broken at job 2: parent link to 3")
        );
    }
}

#[cfg(test)]
mod property;
