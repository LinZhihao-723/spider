//! Root cancels racing other events of a job tree, made deterministic through the test hooks: a
//! leaf's registration of children (T2), a leaf's failure (T3), and a commit of a `CommitReady`
//! non-root or root (T7).

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use spider_core::job::JobState;
use spider_core::task::DataTypeDescriptor;
use spider_core::task::ExecutionPolicy;
use spider_core::task::TaskDescriptor;
use spider_core::task::TaskIndex;
use spider_core::task::TdlContext;
use spider_core::task::TerminationTaskDescriptor;
use spider_core::task::ValueTypeDescriptor;
use spider_core::types::id::JobId;
use spider_core::types::id::TaskId;
use spider_core::types::id::TaskInstanceId;
use spider_core::types::io::TaskGraphInput;
use spider_core::types::io::TaskGraphInputBuilder;
use spider_core::types::io::TaskOutput;
use spider_core::types::io::TaskOutputsSerializer;
use spider_storage::cache::error::CacheError;
use spider_storage::cache::error::StaleStateError;
use spider_storage::job_submission::ChildJobSubmission;
use spider_storage::state::StorageServerError;
use spider_storage::task_instance_pool::TaskInstancePoolConfig;
use spider_storage::test_hooks::ArmedHook;
use spider_storage::test_hooks::EventKindTag;
use spider_storage::test_hooks::HookAction;
use spider_storage::test_hooks::HookCount;
use spider_storage::test_hooks::HookFilter;
use spider_storage::test_hooks::HookPoint;
use sqlx::MySqlPool;
use tokio::task::JoinHandle;

use super::job_tree_infra::TestEnv;
use super::task_graph_builder::SubmittedTaskGraph;
use super::task_graph_builder::compress_job_inputs;
use super::task_graph_builder::compress_task_graph;

/// The result of a job RPC.
type JobResult = Result<JobState, StorageServerError>;

/// The shape of a test job.
#[derive(Clone, Copy, Debug)]
struct JobShape {
    num_tasks: usize,
    max_num_retry: u32,
    with_commit: bool,
    with_cleanup: bool,
}

const ONE_TASK: JobShape = JobShape {
    num_tasks: 1,
    max_num_retry: 0,
    with_commit: false,
    with_cleanup: false,
};

/// The number of children that T2's registering job reports.
const NUM_GRANDCHILDREN: usize = 3;

/// A lane of the inbound queue.
#[derive(Clone, Copy, Debug)]
enum Lane {
    Task,
    Commit,
    Cleanup,
}

/// Which jobs of a test tree have a cleanup task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cleanups {
    /// No job: every deactivated job becomes `Cancelled`, so the tree is quiescent once the race
    /// is decided and its committer exits.
    None,

    /// Every job, except where a test says otherwise: the raced job stays in the tree as
    /// `CleanupReady`, and the committer stays alive.
    All,

    /// Only the jobs that are neither the root nor on the raced job's path: the raced job leaves
    /// the tree while the committer stays alive for the others.
    BystandersOnly,

    /// Only the root: every other job leaves the tree while the committer stays alive for the
    /// root.
    RootOnly,
}

/// Observers of the committer of one tree: its rounds, its database transactions, and its cache
/// applies, in total and per job.
struct TreeProbes {
    rounds: ArmedHook,
    commits: ArmedHook,
    all_applies: ArmedHook,
    applies: HashMap<JobId, ArmedHook>,
}

impl TreeProbes {
    /// # Returns
    ///
    /// Probes of the tree rooted at `root_id` that also observe the cache applies of each of
    /// `job_ids`.
    fn arm(env: &TestEnv, root_id: JobId, job_ids: &[JobId]) -> Self {
        Self {
            rounds: observe(
                env,
                HookPoint::CommitterBeforePlan,
                HookFilter::root(root_id),
            ),
            commits: observe(
                env,
                HookPoint::CommitterAfterCommit,
                HookFilter::root(root_id),
            ),
            all_applies: observe(
                env,
                HookPoint::CommitterAfterApplyJob,
                HookFilter::root(root_id),
            ),
            applies: job_ids
                .iter()
                .map(|job_id| {
                    (
                        *job_id,
                        observe(
                            env,
                            HookPoint::CommitterAfterApplyJob,
                            job_filter(root_id, *job_id),
                        ),
                    )
                })
                .collect(),
        }
    }

    /// Asserts the number of rounds, the events of the last one in planning order, and the number
    /// of database transactions.
    fn assert_rounds(
        &self,
        num_rounds: usize,
        last_round: &[EventKindTag],
        num_commits: usize,
        what: &str,
    ) {
        assert_eq!(self.rounds.hits(), num_rounds, "{what}: number of rounds");
        assert_eq!(
            self.rounds
                .last_context()
                .expect("a round should have been observed")
                .event_kinds,
            last_round,
            "{what}: the events of the last round"
        );
        assert_eq!(
            self.commits.hits(),
            num_commits,
            "{what}: database transactions"
        );
    }

    /// Asserts the number of cache applies of each job of `expected`, and the total number of
    /// cache applies of the tree.
    ///
    /// # Panics
    ///
    /// Panics if the probes do not observe a job of `expected`.
    fn assert_applies(&self, expected: &[(JobId, usize)], total: usize, what: &str) {
        for (job_id, num_applies) in expected {
            assert_eq!(
                self.applies
                    .get(job_id)
                    .expect("the probes should observe the job")
                    .hits(),
                *num_applies,
                "{what}: applies of job {job_id}"
            );
        }
        assert_eq!(
            self.all_applies.hits(),
            total,
            "{what}: applies of the tree"
        );
    }
}

/// Two RPCs whose events race in one tree: the first RPC's round is paused at a hook point until
/// the second RPC's event has been submitted.
struct Race<FirstType, SecondType> {
    paused: ArmedHook,
    first: JoinHandle<FirstType>,
    second: JoinHandle<SecondType>,
}

impl<FirstType: Send + 'static, SecondType: Send + 'static> Race<FirstType, SecondType> {
    /// Runs `first` until its round of the tree rooted at `root_id` is paused at `point`, then
    /// runs `second` until its event, matched by `second_event`, is submitted:
    ///
    /// * At [`HookPoint::CommitterAfterReceive`], both events are planned in one round.
    /// * At [`HookPoint::CommitterBeforeDbTransaction`], `second`'s event is queued while `first`'s
    ///   round is in flight, and is planned after it.
    /// * At [`HookPoint::CommitterAfterCommit`], `second` runs while `first`'s round has committed
    ///   but is not applied to the cache yet, and its event is planned after that round.
    ///
    /// # Returns
    ///
    /// The race, with `first`'s round still paused.
    async fn start(
        env: &TestEnv,
        root_id: JobId,
        point: HookPoint,
        first: impl Future<Output = FirstType> + Send + 'static,
        second: impl Future<Output = SecondType> + Send + 'static,
        second_event: HookFilter,
    ) -> Self {
        let paused = pause(env, point, HookFilter::root(root_id));
        let first = tokio::spawn(first);
        reached(&paused, 1).await;
        // Armed only now, so that the first event's own submission is not counted.
        let submitted = observe(env, HookPoint::EventSubmitted, second_event);
        let second = tokio::spawn(second);
        reached(&submitted, 1).await;
        Self {
            paused,
            first,
            second,
        }
    }

    /// Releases the first RPC's round.
    ///
    /// # Returns
    ///
    /// The results of the first and the second RPC.
    ///
    /// # Panics
    ///
    /// Panics if either RPC panics.
    async fn finish(self) -> (FirstType, SecondType) {
        self.paused.release();
        let first = self.first.await.expect("the first RPC should not panic");
        let second = self.second.await.expect("the second RPC should not panic");
        assert_eq!(self.paused.hits(), 1);
        (first, second)
    }
}

/// The rows of a tree at the moment a committer round of it committed: the round is paused at
/// [`HookPoint::CommitterAfterCommit`] while the rows are read, then released.
struct CommitSnapshot {
    task: JoinHandle<Vec<(JobId, JobState)>>,
}

impl CommitSnapshot {
    /// Arms a snapshot of the tree rooted at `root_id` at the commit of its first round that holds
    /// an event of `kind`.
    ///
    /// # Returns
    ///
    /// The armed snapshot.
    fn arm(env: &TestEnv, root_id: JobId, kind: EventKindTag) -> Self {
        let hook = pause(
            env,
            HookPoint::CommitterAfterCommit,
            HookFilter {
                root_id: Some(root_id),
                job_id: None,
                event_kind: Some(kind),
            },
        );
        let pool = env.pool.clone();
        let task = tokio::spawn(async move {
            reached(&hook, 1).await;
            let rows = tree_rows(&pool, root_id).await;
            hook.release();
            rows
        });
        Self { task }
    }

    /// # Returns
    ///
    /// The rows of the tree at the commit, by ascending job ID.
    ///
    /// # Panics
    ///
    /// Panics if the round never committed.
    async fn rows(self) -> Vec<(JobId, JobState)> {
        self.task
            .await
            .expect("the snapshot should be taken at the commit")
    }
}

/// # Returns
///
/// The number of iterations of each race, from `JOB_TREE_RACE_ITERATIONS` (default 50).
///
/// # Panics
///
/// Panics if the variable is set but is not a number.
fn race_iterations() -> usize {
    std::env::var("JOB_TREE_RACE_ITERATIONS").map_or(50, |value| {
        value
            .parse()
            .expect("`JOB_TREE_RACE_ITERATIONS` should be a number")
    })
}

/// Starts a [`TestEnv`] whose task instance pool never considers the test's execution manager,
/// which sends no heartbeats, dead while a test runs, so that it never removes the test's task
/// instances or re-enqueues their tasks.
///
/// # Returns
///
/// The started environment.
async fn start_env() -> TestEnv {
    TestEnv::start_with_task_instance_pool(TaskInstancePoolConfig {
        execution_manager_stale_cutoff_sec: 24 * 60 * 60,
        ..TaskInstancePoolConfig::default()
    })
    .await
}

/// # Returns
///
/// The state that a cancel or an abort moves a running or blocked job to.
const fn abort_target(has_cleanup: bool) -> JobState {
    if has_cleanup {
        JobState::CleanupReady
    } else {
        JobState::Cancelled
    }
}

/// # Returns
///
/// The stale error that an event of a job's regular lane gets when it is planned against the job's
/// `state` after a cancel or an abort.
const fn regular_stale(state: JobState) -> StaleStateError {
    if state.is_terminal() {
        StaleStateError::JobAlreadyTerminated(state)
    } else {
        StaleStateError::JobNoLongerRunning
    }
}

/// # Returns
///
/// The stale error that a second cancel of a root in `state` gets.
const fn second_cancel_stale(state: JobState) -> StaleStateError {
    match state {
        JobState::CleanupReady => StaleStateError::JobCancellationAlreadyRequested,
        JobState::Cancelled => StaleStateError::JobAlreadyCancelled,
        state => StaleStateError::JobAlreadyTerminated(state),
    }
}

/// Builds a job of `shape` whose independent tasks each take and return one byte-typed value.
///
/// # Returns
///
/// The task graph and its inputs.
///
/// # Panics
///
/// Panics if the graph fails to build.
fn build_job(shape: JobShape) -> (SubmittedTaskGraph, TaskGraphInput) {
    let execution_policy = Some(ExecutionPolicy {
        max_num_retry: shape.max_num_retry,
        ..ExecutionPolicy::default()
    });
    let termination_task = |name: &str| TerminationTaskDescriptor {
        tdl_context: TdlContext {
            package: "test".to_owned(),
            task_func: name.to_owned(),
        },
        execution_policy: execution_policy.clone(),
    };
    let mut graph = SubmittedTaskGraph::new(
        shape.with_commit.then(|| termination_task("commit")),
        shape.with_cleanup.then(|| termination_task("cleanup")),
    )
    .expect("task graph creation should succeed");
    let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
    let mut inputs = TaskGraphInputBuilder::new();
    for task_index in 0..shape.num_tasks {
        graph
            .insert_task(TaskDescriptor {
                tdl_context: TdlContext {
                    package: "test".to_owned(),
                    task_func: format!("t{task_index}"),
                },
                execution_policy: execution_policy.clone(),
                inputs: vec![bytes_type.clone()],
                outputs: vec![bytes_type.clone()],
                input_sources: None,
            })
            .expect("task insertion should succeed");
        inputs
            .append_task_input(&[0u8; 4])
            .expect("task input appending should succeed");
    }
    (graph, inputs.build())
}

/// # Returns
///
/// The submission of a child job of `shape`.
///
/// # Panics
///
/// Panics if the job fails to compress.
fn child(shape: JobShape) -> ChildJobSubmission {
    let (graph, inputs) = build_job(shape);
    ChildJobSubmission {
        compressed_serialized_task_graph: compress_task_graph(&graph)
            .expect("task graph compression should succeed"),
        compressed_serialized_job_inputs: compress_job_inputs(&inputs)
            .expect("input compression should succeed"),
    }
}

/// # Returns
///
/// A task output value that is unique for `tag`.
fn value(tag: u8) -> Vec<u8> {
    vec![tag, 0x5a]
}

/// # Returns
///
/// The serialized outputs of a task whose only output is `value`.
///
/// # Panics
///
/// Panics if the serialization fails.
fn serialized_output(value: &[u8]) -> Vec<u8> {
    TaskOutputsSerializer::from_tuple(&(value.to_vec(),))
        .expect("output serialization should succeed")
}

/// # Returns
///
/// The outputs that the storage service stores for a one-task job whose task reported `value`.
///
/// # Panics
///
/// Panics if the deserialization fails.
fn one_task_outputs(value: &[u8]) -> Vec<TaskOutput> {
    TaskOutputsSerializer::deserialize(&serialized_output(value))
        .expect("output deserialization should succeed")
}

/// # Returns
///
/// The filter of the hits of the tree rooted at `root_id` that concern `job_id`.
const fn job_filter(root_id: JobId, job_id: JobId) -> HookFilter {
    HookFilter {
        root_id: Some(root_id),
        job_id: Some(job_id),
        event_kind: None,
    }
}

/// # Returns
///
/// The filter of the events of `kind` that `job_id` submits in the tree rooted at `root_id`.
const fn event_filter(root_id: JobId, job_id: JobId, kind: EventKindTag) -> HookFilter {
    HookFilter {
        root_id: Some(root_id),
        job_id: Some(job_id),
        event_kind: Some(kind),
    }
}

/// # Returns
///
/// An observer of every hit of `point` that `filter` matches.
fn observe(env: &TestEnv, point: HookPoint, filter: HookFilter) -> ArmedHook {
    env.hooks
        .arm(point, filter, HookAction::Observe, HookCount::Always)
}

/// # Returns
///
/// A pause of the first hit of `point` that `filter` matches.
fn pause(env: &TestEnv, point: HookPoint, filter: HookFilter) -> ArmedHook {
    env.hooks
        .arm(point, filter, HookAction::Pause, HookCount::Once)
}

/// Waits until `hook` has fired `num_hits` times.
///
/// # Panics
///
/// Panics if the hook has not fired that often within 30 seconds, which means that the code under
/// test is stuck.
async fn reached(hook: &ArmedHook, num_hits: usize) {
    tokio::time::timeout(Duration::from_secs(30), hook.reached(num_hits))
        .await
        .expect("the hook should be reached");
}

/// # Returns
///
/// The entries that one poll of `lane` returns, at most `max_entries` within `wait`.
///
/// # Panics
///
/// Panics if the poll fails.
async fn poll_lane(
    env: &TestEnv,
    lane: Lane,
    max_entries: usize,
    wait: Duration,
) -> Vec<(JobId, TaskId)> {
    let service = &env.service;
    match lane {
        Lane::Task => service
            .poll_ready_tasks(max_entries, wait)
            .await
            .expect("the task lane poll should succeed")
            .into_iter()
            .map(|entry| (entry.job_id, TaskId::Index(entry.task_kind)))
            .collect(),
        Lane::Commit => service
            .poll_commit_ready_tasks(max_entries, wait)
            .await
            .expect("the commit lane poll should succeed")
            .into_iter()
            .map(|entry| (entry.job_id, TaskId::Commit))
            .collect(),
        Lane::Cleanup => service
            .poll_cleanup_ready_tasks(max_entries, wait)
            .await
            .expect("the cleanup lane poll should succeed")
            .into_iter()
            .map(|entry| (entry.job_id, TaskId::Cleanup))
            .collect(),
    }
}

/// Polls `lane` until it has returned exactly the entries of `expected`, in any order.
///
/// # Panics
///
/// Panics if the lane returns an entry that `expected` does not hold, one more time than
/// `expected` holds it, or if the entries do not arrive within 10 seconds.
async fn drain_lane(env: &TestEnv, lane: Lane, expected: &[(JobId, TaskId)]) {
    let mut remaining: HashMap<(JobId, TaskId), usize> = HashMap::new();
    for entry in expected {
        *remaining.entry(*entry).or_default() += 1;
    }
    let mut num_remaining = expected.len();
    tokio::time::timeout(Duration::from_secs(10), async {
        while num_remaining > 0 {
            for entry in poll_lane(env, lane, num_remaining, Duration::from_secs(1)).await {
                let Some(count) = remaining.get_mut(&entry).filter(|count| **count > 0) else {
                    panic!("unexpected entry {entry:?} in the {lane:?} lane");
                };
                *count -= 1;
                num_remaining -= 1;
            }
        }
    })
    .await
    .expect("the expected lane entries should be published");
}

/// Asserts that no lane holds an entry.
async fn assert_lanes_empty(env: &TestEnv) {
    for lane in [Lane::Task, Lane::Commit, Lane::Cleanup] {
        assert_eq!(
            poll_lane(env, lane, 1, Duration::from_millis(100)).await,
            Vec::new(),
            "the {lane:?} lane should be empty"
        );
    }
}

/// # Returns
///
/// The result of creating an instance of `task_id` of `job_id`: the instance's ID on success.
fn create_instance(
    env: &TestEnv,
    job_id: JobId,
    task_id: TaskId,
) -> impl Future<Output = Result<TaskInstanceId, StorageServerError>> + Send + use<> {
    let service = env.service.clone();
    let execution_manager_id = env.execution_manager_id;
    async move {
        service
            .create_task_instance(service.session_id(), job_id, task_id, execution_manager_id)
            .await
            .map(|context| context.task_instance_id)
    }
}

/// # Returns
///
/// The ID of a new instance of `task_id` of `job_id`.
///
/// # Panics
///
/// Panics if the instance creation fails.
async fn instance(env: &TestEnv, job_id: JobId, task_id: TaskId) -> TaskInstanceId {
    create_instance(env, job_id, task_id)
        .await
        .expect("the instance creation should succeed")
}

/// # Returns
///
/// A future that reports the success of `task_instance_id` of task `task_index` of `job_id`, with
/// the output `value` and `children`.
fn succeed(
    env: &TestEnv,
    job_id: JobId,
    task_index: TaskIndex,
    task_instance_id: TaskInstanceId,
    value: &[u8],
    children: Vec<ChildJobSubmission>,
) -> impl Future<Output = JobResult> + Send + use<> {
    let service = env.service.clone();
    let serialized_outputs = serialized_output(value);
    async move {
        service
            .succeed_task_instance(
                service.session_id(),
                job_id,
                task_instance_id,
                task_index,
                serialized_outputs,
                children,
            )
            .await
    }
}

/// # Returns
///
/// A future that reports the failure of `task_instance_id` of `task_id` of `job_id` with
/// `message`.
fn fail(
    env: &TestEnv,
    job_id: JobId,
    task_id: TaskId,
    task_instance_id: TaskInstanceId,
    message: &str,
) -> impl Future<Output = JobResult> + Send + use<> {
    let service = env.service.clone();
    let message = message.to_owned();
    async move {
        service
            .fail_task_instance(
                service.session_id(),
                job_id,
                task_instance_id,
                task_id,
                message,
            )
            .await
    }
}

/// # Returns
///
/// A future that reports the success of `task_instance_id` of `job_id`'s commit task.
fn succeed_commit(
    env: &TestEnv,
    job_id: JobId,
    task_instance_id: TaskInstanceId,
) -> impl Future<Output = JobResult> + Send + use<> {
    let service = env.service.clone();
    async move {
        service
            .succeed_commit_task_instance(service.session_id(), job_id, task_instance_id)
            .await
    }
}

/// # Returns
///
/// A future that cancels `job_id`.
fn cancel(env: &TestEnv, job_id: JobId) -> impl Future<Output = JobResult> + Send + use<> {
    let service = env.service.clone();
    async move { service.cancel_job(job_id).await }
}

/// Registers and starts a root job of `shape`, and drains its ready tasks.
///
/// # Returns
///
/// The job ID.
///
/// # Panics
///
/// Panics if the registration or the start fails, or the job's tasks are not published exactly
/// once.
async fn start_root(env: &TestEnv, shape: JobShape) -> JobId {
    let (graph, inputs) = build_job(shape);
    let job_id = env.register_job(&graph, &inputs).await;
    env.service
        .start_job(job_id)
        .await
        .expect("the job should start");
    let ready: Vec<_> = (0..shape.num_tasks)
        .map(|task_index| (job_id, TaskId::Index(task_index)))
        .collect();
    drain_lane(env, Lane::Task, &ready).await;
    job_id
}

/// Runs a new instance of the only task of `parent_id` to success with one child per shape, and
/// drains the children's ready tasks.
///
/// # Returns
///
/// The IDs of the registered children, by ascending ID, which is the order of `shapes`.
///
/// # Panics
///
/// Panics if the report fails, does not block the parent on its children, or the children's tasks
/// are not published exactly once.
async fn register_children(env: &TestEnv, parent_id: JobId, shapes: &[JobShape]) -> Vec<JobId> {
    let task_instance_id = instance(env, parent_id, TaskId::Index(0)).await;
    let children = shapes.iter().copied().map(child).collect();
    assert_eq!(
        succeed(env, parent_id, 0, task_instance_id, &value(0x10), children)
            .await
            .expect("the registering report should succeed"),
        JobState::BlockingOnChildren
    );
    let child_ids = children_of(env, parent_id).await;
    assert_eq!(child_ids.len(), shapes.len());
    let mut ready = Vec::new();
    for (child_id, shape) in child_ids.iter().zip(shapes) {
        ready.extend((0..shape.num_tasks).map(|task_index| (*child_id, TaskId::Index(task_index))));
    }
    drain_lane(env, Lane::Task, &ready).await;
    child_ids
}

/// Runs a new instance of the only task of `job_id` to success with the output `value`.
///
/// # Returns
///
/// The job state after the report.
///
/// # Panics
///
/// Panics if the report fails.
async fn run_task(env: &TestEnv, job_id: JobId, value: &[u8]) -> JobState {
    let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
    succeed(env, job_id, 0, task_instance_id, value, Vec::new())
        .await
        .expect("the report should succeed")
}

/// Drains the published cleanup tasks of `job_ids` and runs a new instance of each to success.
///
/// # Panics
///
/// Panics if the cleanup tasks are not published exactly once each, or a report does not cancel
/// its job.
async fn run_cleanups(env: &TestEnv, job_ids: &[JobId]) {
    let expected: Vec<_> = job_ids
        .iter()
        .map(|job_id| (*job_id, TaskId::Cleanup))
        .collect();
    drain_lane(env, Lane::Cleanup, &expected).await;
    for job_id in job_ids {
        let task_instance_id = instance(env, *job_id, TaskId::Cleanup).await;
        assert_eq!(
            env.service
                .succeed_cleanup_task_instance(env.service.session_id(), *job_id, task_instance_id)
                .await
                .expect("the cleanup report should succeed"),
            JobState::Cancelled,
            "the cleanup of job {job_id}"
        );
    }
}

/// # Returns
///
/// The IDs of the rows whose parent is `parent_id`, by ascending ID.
///
/// # Panics
///
/// Panics if the query fails.
async fn children_of(env: &TestEnv, parent_id: JobId) -> Vec<JobId> {
    sqlx::query_scalar::<_, JobId>("SELECT `id` FROM `jobs` WHERE `parent_id` = ? ORDER BY `id`;")
        .bind(parent_id)
        .fetch_all(&env.pool)
        .await
        .expect("the child query should succeed")
}

/// # Returns
///
/// The IDs and states of the rows of the tree rooted at `root_id`, by ascending ID.
///
/// # Panics
///
/// Panics if the query fails.
async fn tree_rows(pool: &MySqlPool, root_id: JobId) -> Vec<(JobId, JobState)> {
    sqlx::query_as::<_, (JobId, JobState)>(
        "SELECT `id`, `state` FROM `jobs` WHERE `id` = ? OR `root_id` = ? ORDER BY `id`;",
    )
    .bind(root_id)
    .bind(root_id)
    .fetch_all(pool)
    .await
    .expect("the tree query should succeed")
}

/// # Returns
///
/// The error messages stored in the rows of the tree rooted at `root_id`, by ascending job ID.
///
/// # Panics
///
/// Panics if the query fails.
async fn tree_errors(env: &TestEnv, root_id: JobId) -> Vec<(JobId, String)> {
    sqlx::query_as::<_, (JobId, String)>(
        "SELECT `id`, `error_message` FROM `jobs` WHERE (`id` = ? OR `root_id` = ?) AND \
         `error_message` IS NOT NULL ORDER BY `id`;",
    )
    .bind(root_id)
    .bind(root_id)
    .fetch_all(&env.pool)
    .await
    .expect("the error query should succeed")
}

/// Asserts that the rows of the tree rooted at `root_id` are exactly the jobs of `expected`, in
/// their states.
async fn assert_tree_rows(
    env: &TestEnv,
    root_id: JobId,
    expected: &HashMap<JobId, JobState>,
    what: &str,
) {
    assert_rows(&tree_rows(&env.pool, root_id).await, expected, what);
}

/// Asserts that `rows` are exactly the jobs of `expected`, in their states.
fn assert_rows(rows: &[(JobId, JobState)], expected: &HashMap<JobId, JobState>, what: &str) {
    let mut expected: Vec<(JobId, JobState)> = expected
        .iter()
        .map(|(job_id, state)| (*job_id, *state))
        .collect();
    expected.sort_by_key(|(job_id, _)| job_id.get());
    assert_eq!(rows, expected, "{what}: the rows of the tree");
}

/// Asserts that every job of `expected` has its state in the cache and in its row, that the row's
/// end time is set exactly when the state is terminal, and that the service has not shut down
/// fatally.
async fn assert_states(env: &TestEnv, expected: &HashMap<JobId, JobState>, what: &str) {
    for (job_id, state) in expected {
        assert_eq!(
            env.cached_job_state(*job_id).await,
            Some(*state),
            "{what}: cached state of job {job_id}"
        );
        let (row_state, has_ended) = sqlx::query_as::<_, (JobState, i64)>(
            "SELECT `state`, `ended_at` IS NOT NULL FROM `jobs` WHERE `id` = ?;",
        )
        .bind(job_id)
        .fetch_one(&env.pool)
        .await
        .expect("the row state query should succeed");
        assert_eq!(row_state, *state, "{what}: row state of job {job_id}");
        assert_eq!(
            has_ended != 0,
            state.is_terminal(),
            "{what}: end time of job {job_id} in state {state:?}"
        );
    }
    assert_not_fatal(env, what);
}

/// Asserts that the service has not shut down fatally.
fn assert_not_fatal(env: &TestEnv, what: &str) {
    assert!(
        !env.is_fatal(),
        "{what}: the service should not have shut down fatally"
    );
}

/// Asserts that the tree rooted at `root_id` has no non-terminal row.
async fn assert_quiescent(env: &TestEnv, root_id: JobId) {
    let non_terminal = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM `jobs` WHERE (`id` = ? OR `root_id` = ?) AND `state` NOT IN (?, ?, \
         ?);",
    )
    .bind(root_id)
    .bind(root_id)
    .bind(JobState::Succeeded)
    .bind(JobState::Failed)
    .bind(JobState::Cancelled)
    .fetch_one(&env.pool)
    .await
    .expect("the quiescence query should succeed");
    assert_eq!(non_terminal, 0, "the tree of {root_id} should be quiescent");
}

/// Asserts that `result` is the stale error `expected`.
fn assert_stale<ValueType: std::fmt::Debug>(
    result: &Result<ValueType, StorageServerError>,
    expected: &StaleStateError,
    what: &str,
) {
    match result {
        Err(StorageServerError::Cache(CacheError::StaleState(error)))
            if error.to_string() == expected.to_string() => {}
        other => panic!("{what}: expected the stale error `{expected}`, got {other:?}"),
    }
}

/// Asserts that `result` is the job state `expected`.
fn assert_job_state(result: &JobResult, expected: JobState, what: &str) {
    match result {
        Ok(state) if *state == expected => {}
        other => panic!("{what}: expected `Ok({expected:?})`, got {other:?}"),
    }
}

/// Asserts what the root of a decided race shows to the client: `GetJobState` returns `state`, and
/// one more cancel is stale.
async fn assert_root_after_race(env: &TestEnv, root_id: JobId, state: JobState, what: &str) {
    assert_eq!(
        env.service
            .get_job_state(root_id)
            .await
            .expect("the root's state should be readable"),
        state,
        "{what}: `GetJobState` of the root"
    );
    assert_stale(
        &cancel(env, root_id).await,
        &second_cancel_stale(state),
        &format!("{what}: one more cancel"),
    );
}

/// The job whose task registers children in T2.
#[derive(Clone, Copy, Debug)]
enum Registrar {
    /// The root, before it has any child.
    Root,

    /// A leaf `L` in the tree `R → {L, S}`.
    Leaf,
}

/// How T2's registration and cancel interleave.
#[derive(Clone, Copy, Debug)]
enum RegistrationRace {
    /// Both in one round; the registration arrives first.
    SameRoundRegistrationFirst,

    /// Both in one round; the cancel arrives first.
    SameRoundCancelFirst,

    /// The registration is submitted while the cancel's round is in flight, before its database
    /// transaction.
    CancelInFlight,

    /// The registration's phase A runs while the cancel's round has committed but is not applied.
    CancelInCommitWindow,

    /// The registration's phase A holds the registering job's read guard while the cancel commits,
    /// so that the cancel's apply waits for it.
    PhaseAAcrossCancelCommit,

    /// The registration's phase A has dropped its guard and is about to submit when the cancel is
    /// answered.
    PhaseABeforeSubmitAcrossCancel,

    /// The registration is reported after the cancel was answered.
    AfterCancelReply,

    /// The cancel is submitted while the registration's round is in flight, before its database
    /// transaction.
    RegistrationInFlight,

    /// The cancel is submitted while the registration's round has committed, inserting the
    /// children, but is not applied.
    RegistrationInCommitWindow,

    /// The cancel is submitted after the registration was answered.
    AfterRegistrationReply,
}

impl RegistrationRace {
    const ALL: [Self; 10] = [
        Self::SameRoundRegistrationFirst,
        Self::SameRoundCancelFirst,
        Self::CancelInFlight,
        Self::CancelInCommitWindow,
        Self::PhaseAAcrossCancelCommit,
        Self::PhaseABeforeSubmitAcrossCancel,
        Self::AfterCancelReply,
        Self::RegistrationInFlight,
        Self::RegistrationInCommitWindow,
        Self::AfterRegistrationReply,
    ];

    /// # Returns
    ///
    /// Whether the cancel is planned before the registration.
    const fn cancel_first(self) -> bool {
        !matches!(
            self,
            Self::RegistrationInFlight
                | Self::RegistrationInCommitWindow
                | Self::AfterRegistrationReply
        )
    }
}

/// The jobs of a T2 tree.
struct RegistrationTree {
    root_id: JobId,
    registrar_id: JobId,
    sibling_id: Option<JobId>,
    has_cleanup: HashMap<JobId, bool>,
    grandchild_shapes: Vec<JobShape>,
}

impl RegistrationTree {
    /// Starts a tree whose registering job is `registrar`, with `cleanups`. With
    /// [`Cleanups::All`], every job but the second grandchild has a cleanup task; with
    /// [`Cleanups::BystandersOnly`], only the sibling has one.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, registrar: Registrar, cleanups: Cleanups) -> Self {
        let shape = |has_cleanup: bool| JobShape {
            with_cleanup: has_cleanup,
            ..ONE_TASK
        };
        let all = cleanups == Cleanups::All;
        let root_id = start_root(env, shape(all)).await;
        let mut has_cleanup = HashMap::from([(root_id, all)]);
        let (registrar_id, sibling_id) = match registrar {
            Registrar::Root => (root_id, None),
            Registrar::Leaf => {
                let sibling_has_cleanup = cleanups != Cleanups::None;
                let children =
                    register_children(env, root_id, &[shape(all), shape(sibling_has_cleanup)])
                        .await;
                has_cleanup.insert(children[0], all);
                has_cleanup.insert(children[1], sibling_has_cleanup);
                (children[0], Some(children[1]))
            }
        };
        Self {
            root_id,
            registrar_id,
            sibling_id,
            has_cleanup,
            grandchild_shapes: (0..NUM_GRANDCHILDREN)
                .map(|index| shape(all && index != 1))
                .collect(),
        }
    }

    /// # Returns
    ///
    /// The jobs of the tree before the registration.
    fn job_ids(&self) -> Vec<JobId> {
        let mut job_ids = vec![self.root_id];
        if self.registrar_id != self.root_id {
            job_ids.push(self.registrar_id);
        }
        job_ids.extend(self.sibling_id);
        job_ids
    }

    /// # Returns
    ///
    /// The state that the cancel moves `job_id`, a job of the tree before the registration, to.
    fn target(&self, job_id: JobId) -> JobState {
        abort_target(self.has_cleanup[&job_id])
    }

    /// # Returns
    ///
    /// Whether the tree's committer stays alive after the cancel, because a job of the tree is
    /// `CleanupReady`.
    fn outlives_cancel(&self) -> bool {
        self.has_cleanup.values().any(|has_cleanup| *has_cleanup)
    }

    /// # Returns
    ///
    /// The states of the jobs of the tree before the race.
    fn before_race(&self) -> HashMap<JobId, JobState> {
        self.job_ids()
            .into_iter()
            .map(|job_id| {
                let state = if job_id == self.root_id && self.registrar_id != self.root_id {
                    JobState::BlockingOnChildren
                } else {
                    JobState::Running
                };
                (job_id, state)
            })
            .collect()
    }

    /// # Returns
    ///
    /// The states that the cancel moves the jobs of the tree before the registration to.
    fn cancelled(&self) -> HashMap<JobId, JobState> {
        self.job_ids()
            .into_iter()
            .map(|job_id| (job_id, self.target(job_id)))
            .collect()
    }
}

/// Observers of a T2 race: the tree's committer, the registering job's submitted
/// `RegularTasksDone` events, and the rows at the cancel's commit.
struct RegistrationProbes {
    tree: TreeProbes,
    registrations: ArmedHook,
    snapshot: CommitSnapshot,
}

impl RegistrationProbes {
    /// # Returns
    ///
    /// The probes of `tree`.
    fn arm(env: &TestEnv, tree: &RegistrationTree) -> Self {
        Self {
            tree: TreeProbes::arm(env, tree.root_id, &tree.job_ids()),
            registrations: observe(
                env,
                HookPoint::EventSubmitted,
                event_filter(
                    tree.root_id,
                    tree.registrar_id,
                    EventKindTag::RegularTasksDone,
                ),
            ),
            snapshot: CommitSnapshot::arm(env, tree.root_id, EventKindTag::CancelRoot),
        }
    }

    /// Asserts the submitted registrations, the rounds, the database transactions and the cache
    /// applies of `tree`'s committer in `race`.
    fn assert_committer(&self, tree: &RegistrationTree, race: RegistrationRace, what: &str) {
        let outlives_cancel = tree.outlives_cancel();
        let num_registrations = match race {
            RegistrationRace::AfterCancelReply => Some(0),
            RegistrationRace::PhaseABeforeSubmitAcrossCancel if !outlives_cancel => Some(0),
            // The registration is submitted while the cancel's apply finishes, so it may or may
            // not reach the committer before the committer of the quiescent tree exits.
            RegistrationRace::PhaseAAcrossCancelCommit if !outlives_cancel => None,
            _ => Some(1),
        };
        let hits = self.registrations.hits();
        assert!(
            num_registrations.map_or(hits <= 1, |expected| hits == expected),
            "{what}: {hits} submitted registrations, expected {num_registrations:?}"
        );

        let cancel_round = [EventKindTag::CancelRoot];
        match race {
            RegistrationRace::SameRoundRegistrationFirst
            | RegistrationRace::SameRoundCancelFirst => self.tree.assert_rounds(
                1,
                &[EventKindTag::CancelRoot, EventKindTag::RegularTasksDone],
                1,
                what,
            ),
            RegistrationRace::CancelInFlight
            | RegistrationRace::CancelInCommitWindow
            | RegistrationRace::PhaseAAcrossCancelCommit
            | RegistrationRace::PhaseABeforeSubmitAcrossCancel
                if outlives_cancel =>
            {
                self.tree
                    .assert_rounds(2, &[EventKindTag::RegularTasksDone], 1, what);
            }
            RegistrationRace::CancelInFlight
            | RegistrationRace::CancelInCommitWindow
            | RegistrationRace::PhaseAAcrossCancelCommit
            | RegistrationRace::PhaseABeforeSubmitAcrossCancel
            | RegistrationRace::AfterCancelReply => {
                self.tree.assert_rounds(1, &cancel_round, 1, what);
            }
            RegistrationRace::RegistrationInFlight
            | RegistrationRace::RegistrationInCommitWindow
            | RegistrationRace::AfterRegistrationReply => {
                self.tree.assert_rounds(2, &cancel_round, 2, what);
            }
        }

        let job_ids = tree.job_ids();
        let registers = !race.cancel_first();
        let applies: Vec<(JobId, usize)> = job_ids
            .iter()
            .map(|job_id| {
                let num_applies = if registers && *job_id == tree.registrar_id {
                    2
                } else {
                    1
                };
                (*job_id, num_applies)
            })
            .collect();
        let total = if registers {
            job_ids.len() + 1 + NUM_GRANDCHILDREN
        } else {
            job_ids.len()
        };
        self.tree.assert_applies(&applies, total, what);
    }
}

/// T2: a root cancel races the registration of a leaf's (or the root's) children. A cancel planned
/// first inserts no child; a registration planned first inserts the children, which the cancel then
/// deactivates in its own database transaction. No child is ever live under a closed tree.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_root_cancel_racing_a_registration_never_leaves_a_live_child() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for (registrar, cleanups) in [
            (Registrar::Leaf, Cleanups::None),
            (Registrar::Leaf, Cleanups::All),
            (Registrar::Leaf, Cleanups::BystandersOnly),
            (Registrar::Root, Cleanups::None),
            (Registrar::Root, Cleanups::All),
        ] {
            for race in RegistrationRace::ALL {
                assert_cancel_races_registration(&env, registrar, cleanups, race).await?;
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Races the cancel of a T2 tree with the registration of its registering job's children as `race`
/// says, and asserts the outcome.
///
/// # Errors
///
/// Returns an error if the cancel's RPC fails.
async fn assert_cancel_races_registration(
    env: &TestEnv,
    registrar: Registrar,
    cleanups: Cleanups,
    race: RegistrationRace,
) -> anyhow::Result<()> {
    let what = format!("{registrar:?}, {cleanups:?}, {race:?}");
    let tree = RegistrationTree::start(env, registrar, cleanups).await;
    let task_instance_id = instance(env, tree.registrar_id, TaskId::Index(0)).await;
    let probes = RegistrationProbes::arm(env, &tree);

    let (report, cancelled) = race_registration(env, &tree, race, task_instance_id, &what).await;

    let root_state = tree.target(tree.root_id);
    assert_job_state(&cancelled, root_state, &format!("{what}: the cancel"));
    cancelled?;
    let after_cancel = assert_registration(env, &tree, race, &report, &what).await;
    probes.assert_committer(&tree, race, &what);
    assert_rows(
        &probes.snapshot.rows().await,
        &after_cancel,
        &format!("{what}: at the cancel's commit"),
    );
    finish_tree(env, tree.root_id, &after_cancel, &[], &what).await;
    Ok(())
}

/// Runs the registration of `tree`'s children and the cancel of its root as `race` says.
///
/// # Returns
///
/// The results of the registering report and of the cancel.
async fn race_registration(
    env: &TestEnv,
    tree: &RegistrationTree,
    race: RegistrationRace,
    task_instance_id: TaskInstanceId,
    what: &str,
) -> (JobResult, JobResult) {
    let root_id = tree.root_id;
    let registrar_id = tree.registrar_id;
    let report = succeed(
        env,
        registrar_id,
        0,
        task_instance_id,
        &value(0x20),
        tree.grandchild_shapes.iter().copied().map(child).collect(),
    );
    let cancelled = cancel(env, root_id);
    let registration_event = event_filter(root_id, registrar_id, EventKindTag::RegularTasksDone);
    let cancel_event = event_filter(root_id, root_id, EventKindTag::CancelRoot);
    match race {
        RegistrationRace::SameRoundRegistrationFirst => {
            Race::start(
                env,
                root_id,
                HookPoint::CommitterAfterReceive,
                report,
                cancelled,
                cancel_event,
            )
            .await
            .finish()
            .await
        }
        RegistrationRace::SameRoundCancelFirst
        | RegistrationRace::CancelInFlight
        | RegistrationRace::CancelInCommitWindow => {
            let point = match race {
                RegistrationRace::SameRoundCancelFirst => HookPoint::CommitterAfterReceive,
                RegistrationRace::CancelInFlight => HookPoint::CommitterBeforeDbTransaction,
                _ => HookPoint::CommitterAfterCommit,
            };
            let running =
                Race::start(env, root_id, point, cancelled, report, registration_event).await;
            if matches!(race, RegistrationRace::CancelInCommitWindow) {
                assert_cancel_committed_not_applied(env, tree, what).await;
            }
            let (cancelled, report) = running.finish().await;
            (report, cancelled)
        }
        RegistrationRace::PhaseAAcrossCancelCommit => {
            phase_a_across_cancel_commit(env, tree, report, cancelled, what).await
        }
        RegistrationRace::PhaseABeforeSubmitAcrossCancel => {
            let held = pause(
                env,
                HookPoint::PhaseABeforeSubmit,
                job_filter(root_id, registrar_id),
            );
            let report = tokio::spawn(report);
            reached(&held, 1).await;
            let cancelled = cancelled.await;
            held.release();
            let report = report.await.expect("the report should not panic");
            (report, cancelled)
        }
        RegistrationRace::AfterCancelReply => {
            let cancelled = cancelled.await;
            (report.await, cancelled)
        }
        RegistrationRace::RegistrationInFlight => {
            Race::start(
                env,
                root_id,
                HookPoint::CommitterBeforeDbTransaction,
                report,
                cancelled,
                cancel_event,
            )
            .await
            .finish()
            .await
        }
        RegistrationRace::RegistrationInCommitWindow => {
            let running = Race::start(
                env,
                root_id,
                HookPoint::CommitterAfterCommit,
                report,
                cancelled,
                cancel_event,
            )
            .await;
            assert_registration_committed_not_applied(env, tree, what).await;
            running.finish().await
        }
        RegistrationRace::AfterRegistrationReply => {
            let report = report.await;
            (report, cancelled.await)
        }
    }
}

/// Runs `report` until its phase A, holding the registering job's read guard, is paused right
/// after the task's success, then runs `cancelled` until its round of `tree` has committed, and
/// releases the phase A.
///
/// # Returns
///
/// The results of `report` and of `cancelled`.
///
/// # Panics
///
/// Panics if the cancel finishes while the phase A is paused, or if either RPC panics.
async fn phase_a_across_cancel_commit(
    env: &TestEnv,
    tree: &RegistrationTree,
    report: impl Future<Output = JobResult> + Send + 'static,
    cancelled: impl Future<Output = JobResult> + Send + 'static,
    what: &str,
) -> (JobResult, JobResult) {
    let held = pause(
        env,
        HookPoint::PhaseAAfterTcbFlip,
        job_filter(tree.root_id, tree.registrar_id),
    );
    let committed = observe(
        env,
        HookPoint::CommitterAfterCommit,
        event_filter(tree.root_id, tree.root_id, EventKindTag::CancelRoot),
    );
    let report = tokio::spawn(report);
    reached(&held, 1).await;
    let cancelled = tokio::spawn(cancelled);
    reached(&committed, 1).await;
    // The cancel's apply needs the registering job's write lock, against which the paused phase A
    // holds a read guard. A read of that job's cached state would queue behind the waiting writer,
    // so only the rows are read here.
    assert_tree_rows(env, tree.root_id, &tree.cancelled(), what).await;
    assert!(
        !cancelled.is_finished(),
        "{what}: the cancel should wait for the paused phase A"
    );
    held.release();
    let report = report.await.expect("the report should not panic");
    let cancelled = cancelled.await.expect("the cancel should not panic");
    (report, cancelled)
}

/// Asserts the state of `tree` while the cancel's round is paused between its commit and its
/// apply, after the registration's phase A passed: the database is ahead of the cache, whose jobs
/// are still in their states before the race while their rows are deactivated.
async fn assert_cancel_committed_not_applied(env: &TestEnv, tree: &RegistrationTree, what: &str) {
    assert_tree_rows(env, tree.root_id, &tree.cancelled(), what).await;
    for (job_id, state) in tree.before_race() {
        assert_eq!(
            env.cached_job_state(job_id).await,
            Some(state),
            "{what}: cached state of job {job_id} in the cancel's commit window"
        );
    }
}

/// Asserts the state of `tree` while the registration's round is paused between its commit and
/// its apply: the children's rows exist and are `Running` under the `BlockingOnChildren`
/// registering job, but the cache does not know them, and still has the registering job
/// `Running`.
async fn assert_registration_committed_not_applied(
    env: &TestEnv,
    tree: &RegistrationTree,
    what: &str,
) {
    let grandchild_ids = children_of(env, tree.registrar_id).await;
    assert_eq!(
        grandchild_ids.len(),
        NUM_GRANDCHILDREN,
        "{what}: the children inserted by the registration's commit"
    );
    let mut committed = tree.before_race();
    committed.insert(tree.registrar_id, JobState::BlockingOnChildren);
    for grandchild_id in &grandchild_ids {
        committed.insert(*grandchild_id, JobState::Running);
        assert_eq!(
            env.cached_job_state(*grandchild_id).await,
            None,
            "{what}: a child that is committed but not yet bound should not be cached"
        );
    }
    assert_tree_rows(env, tree.root_id, &committed, what).await;
    assert_eq!(
        env.cached_job_state(tree.registrar_id).await,
        Some(JobState::Running),
        "{what}: the registering job before the apply"
    );
}

/// Asserts the registering report's result and the children it left in `tree`, and that the
/// published tasks of registered children can no longer get an instance.
///
/// # Returns
///
/// The states of every job of the tree after the cancel.
async fn assert_registration(
    env: &TestEnv,
    tree: &RegistrationTree,
    race: RegistrationRace,
    report: &JobResult,
    what: &str,
) -> HashMap<JobId, JobState> {
    let grandchild_ids = children_of(env, tree.registrar_id).await;
    let mut after_cancel = tree.cancelled();
    if race.cancel_first() {
        let expected = if matches!(race, RegistrationRace::AfterCancelReply) {
            StaleStateError::JobNoLongerRunning
        } else {
            regular_stale(tree.target(tree.registrar_id))
        };
        assert_stale(report, &expected, &format!("{what}: the registration"));
        assert_eq!(
            grandchild_ids,
            Vec::new(),
            "{what}: a registration planned after the cancel should insert no child"
        );
        return after_cancel;
    }
    assert_job_state(
        report,
        JobState::BlockingOnChildren,
        &format!("{what}: the registration"),
    );
    assert_eq!(
        grandchild_ids.len(),
        NUM_GRANDCHILDREN,
        "{what}: the registered children"
    );
    for (grandchild_id, shape) in grandchild_ids.iter().zip(&tree.grandchild_shapes) {
        after_cancel.insert(*grandchild_id, abort_target(shape.with_cleanup));
    }
    let ready: Vec<_> = grandchild_ids
        .iter()
        .map(|grandchild_id| (*grandchild_id, TaskId::Index(0)))
        .collect();
    drain_lane(env, Lane::Task, &ready).await;
    for grandchild_id in &grandchild_ids {
        assert_stale(
            &create_instance(env, *grandchild_id, TaskId::Index(0)).await,
            &StaleStateError::JobNoLongerRunning,
            &format!("{what}: an instance of a cancelled child's published task"),
        );
    }
    after_cancel
}

/// Finishes a tree whose race is decided and whose jobs are in the states of `expected`: asserts
/// those states in the cache and in the rows, that the rows are exactly `expected`'s jobs, that
/// the only error messages are `expected_errors`, and what the root shows to the client; then runs
/// the cleanup task of every `CleanupReady` job and asserts that the tree is quiescent.
async fn finish_tree(
    env: &TestEnv,
    root_id: JobId,
    expected: &HashMap<JobId, JobState>,
    expected_errors: &[(JobId, String)],
    what: &str,
) {
    assert_states(env, expected, what).await;
    assert_tree_rows(env, root_id, expected, what).await;
    assert_eq!(
        tree_errors(env, root_id).await,
        expected_errors,
        "{what}: the error messages of the tree"
    );
    assert_root_after_race(env, root_id, expected[&root_id], what).await;

    let mut cleanup_ready: Vec<JobId> = expected
        .iter()
        .filter(|(_, state)| **state == JobState::CleanupReady)
        .map(|(job_id, _)| *job_id)
        .collect();
    cleanup_ready.sort_by_key(JobId::get);
    run_cleanups(env, &cleanup_ready).await;
    let finished: HashMap<JobId, JobState> = expected
        .iter()
        .map(|(job_id, state)| {
            let state = if *state == JobState::CleanupReady {
                JobState::Cancelled
            } else {
                *state
            };
            (*job_id, state)
        })
        .collect();
    assert_states(env, &finished, what).await;
    assert_quiescent(env, root_id).await;
}

/// How T2's second variant interleaves the registrations of two sibling leaves with the root's
/// cancel.
#[derive(Clone, Copy, Debug)]
enum TwoRegistrationRace {
    /// The first registration, the second registration and the cancel arrive in this order before
    /// one round. The round takes the cancel and the first registration, and the second
    /// registration waits for the next round.
    SameRound,

    /// The second registration and the cancel are submitted while the first registration's round
    /// is in flight, before its database transaction.
    FirstInFlight,

    /// The second registration's phase A runs, and the cancel is submitted, while the first
    /// registration's round has committed but is not applied.
    FirstInCommitWindow,
}

/// The jobs of a tree `R → {L, S}` whose two leaves both register children.
struct TwoRegistrationTree {
    root_id: JobId,
    leaf_ids: [JobId; 2],
    root_has_cleanup: bool,
    leaf_has_cleanup: bool,
    grandchild_shapes: Vec<JobShape>,
}

impl TwoRegistrationTree {
    /// Starts a tree with `cleanups`: with [`Cleanups::All`], every job but the second grandchild
    /// of each leaf has a cleanup task; with [`Cleanups::RootOnly`], only the root has one.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, cleanups: Cleanups) -> Self {
        let all = cleanups == Cleanups::All;
        let root_has_cleanup = cleanups != Cleanups::None;
        let shape = |has_cleanup: bool| JobShape {
            with_cleanup: has_cleanup,
            ..ONE_TASK
        };
        let root_id = start_root(env, shape(root_has_cleanup)).await;
        let [first, second] = register_children(env, root_id, &[shape(all), shape(all)]).await[..]
        else {
            panic!("the root should have two children");
        };
        Self {
            root_id,
            leaf_ids: [first, second],
            root_has_cleanup,
            leaf_has_cleanup: all,
            grandchild_shapes: (0..NUM_GRANDCHILDREN)
                .map(|index| shape(all && index != 1))
                .collect(),
        }
    }

    /// # Returns
    ///
    /// The states that the cancel moves the root and the leaves to.
    fn cancelled(&self) -> HashMap<JobId, JobState> {
        let leaf = abort_target(self.leaf_has_cleanup);
        HashMap::from([
            (self.root_id, abort_target(self.root_has_cleanup)),
            (self.leaf_ids[0], leaf),
            (self.leaf_ids[1], leaf),
        ])
    }

    /// # Returns
    ///
    /// Whether the tree's committer stays alive after the cancel.
    const fn outlives_cancel(&self) -> bool {
        self.root_has_cleanup
    }

    /// # Returns
    ///
    /// A future that reports the success of `leaf_id`'s only task, through `task_instance_id`,
    /// with the tree's grandchildren.
    fn registration(
        &self,
        env: &TestEnv,
        leaf_id: JobId,
        task_instance_id: TaskInstanceId,
    ) -> impl Future<Output = JobResult> + Send + use<> {
        succeed(
            env,
            leaf_id,
            0,
            task_instance_id,
            &value(0x21),
            self.grandchild_shapes.iter().copied().map(child).collect(),
        )
    }
}

/// T2 with two registering leaves: the oldest registration is planned last in its round and the
/// other one waits for the next round. A cancel that is planned before either inserts no child; a
/// cancel planned after the first one deactivates its children and makes the second one stale.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_root_cancel_racing_two_registrations_never_leaves_a_live_child() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for cleanups in [Cleanups::None, Cleanups::All, Cleanups::RootOnly] {
            for race in [
                TwoRegistrationRace::SameRound,
                TwoRegistrationRace::FirstInFlight,
                TwoRegistrationRace::FirstInCommitWindow,
            ] {
                assert_cancel_races_two_registrations(&env, cleanups, race).await?;
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Races the cancel of a tree with the registrations of both its leaves as `race` says, and
/// asserts the outcome.
///
/// # Errors
///
/// Returns an error if the cancel's RPC fails.
async fn assert_cancel_races_two_registrations(
    env: &TestEnv,
    cleanups: Cleanups,
    race: TwoRegistrationRace,
) -> anyhow::Result<()> {
    let what = format!("two registrations, {cleanups:?}, {race:?}");
    let tree = TwoRegistrationTree::start(env, cleanups).await;
    let root_id = tree.root_id;
    let [first_id, second_id] = tree.leaf_ids;
    let first_instance = instance(env, first_id, TaskId::Index(0)).await;
    let second_instance = instance(env, second_id, TaskId::Index(0)).await;
    let probes = TreeProbes::arm(env, root_id, &[root_id, first_id, second_id]);
    let snapshot = CommitSnapshot::arm(env, root_id, EventKindTag::CancelRoot);

    let (first, second, cancelled) =
        race_two_registrations(env, &tree, race, [first_instance, second_instance]).await;

    let mut after_cancel = tree.cancelled();
    assert_job_state(
        &cancelled,
        after_cancel[&root_id],
        &format!("{what}: the cancel"),
    );
    cancelled?;
    let leaf_stale = regular_stale(after_cancel[&first_id]);
    assert_stale(
        &second,
        &leaf_stale,
        &format!("{what}: the second registration"),
    );
    assert_eq!(
        children_of(env, second_id).await,
        Vec::new(),
        "{what}: the second registration should insert no child"
    );
    let first_children = children_of(env, first_id).await;
    let registers = !matches!(race, TwoRegistrationRace::SameRound);
    if registers {
        assert_job_state(
            &first,
            JobState::BlockingOnChildren,
            &format!("{what}: the first registration"),
        );
        assert_eq!(first_children.len(), NUM_GRANDCHILDREN, "{what}");
        for (grandchild_id, shape) in first_children.iter().zip(&tree.grandchild_shapes) {
            after_cancel.insert(*grandchild_id, abort_target(shape.with_cleanup));
        }
        let ready: Vec<_> = first_children
            .iter()
            .map(|grandchild_id| (*grandchild_id, TaskId::Index(0)))
            .collect();
        drain_lane(env, Lane::Task, &ready).await;
        probes.assert_rounds(
            2,
            &[EventKindTag::CancelRoot, EventKindTag::RegularTasksDone],
            2,
            &what,
        );
        probes.assert_applies(
            &[(root_id, 1), (first_id, 2), (second_id, 1)],
            3 + 1 + NUM_GRANDCHILDREN,
            &what,
        );
    } else {
        assert_stale(
            &first,
            &leaf_stale,
            &format!("{what}: the first registration"),
        );
        assert_eq!(first_children, Vec::new(), "{what}");
        if tree.outlives_cancel() {
            probes.assert_rounds(2, &[EventKindTag::RegularTasksDone], 1, &what);
        } else {
            probes.assert_rounds(
                1,
                &[EventKindTag::CancelRoot, EventKindTag::RegularTasksDone],
                1,
                &what,
            );
        }
        probes.assert_applies(&[(root_id, 1), (first_id, 1), (second_id, 1)], 3, &what);
    }
    assert_rows(
        &snapshot.rows().await,
        &after_cancel,
        &format!("{what}: at the cancel's commit"),
    );
    finish_tree(env, root_id, &after_cancel, &[], &what).await;
    Ok(())
}

/// Runs the registrations of both leaves of `tree`, through their instances `task_instance_ids`,
/// and the cancel of its root as `race` says.
///
/// # Returns
///
/// The results of the first and the second registration, and of the cancel.
///
/// # Panics
///
/// Panics if an RPC panics.
async fn race_two_registrations(
    env: &TestEnv,
    tree: &TwoRegistrationTree,
    race: TwoRegistrationRace,
    task_instance_ids: [TaskInstanceId; 2],
) -> (JobResult, JobResult, JobResult) {
    let root_id = tree.root_id;
    let [first_id, second_id] = tree.leaf_ids;
    let [first_instance, second_instance] = task_instance_ids;
    let paused = pause(
        env,
        match race {
            TwoRegistrationRace::SameRound => HookPoint::CommitterAfterReceive,
            TwoRegistrationRace::FirstInFlight => HookPoint::CommitterBeforeDbTransaction,
            TwoRegistrationRace::FirstInCommitWindow => HookPoint::CommitterAfterCommit,
        },
        HookFilter::root(root_id),
    );
    let first = tokio::spawn(tree.registration(env, first_id, first_instance));
    reached(&paused, 1).await;
    let second_submitted = observe(
        env,
        HookPoint::EventSubmitted,
        event_filter(root_id, second_id, EventKindTag::RegularTasksDone),
    );
    let second = tokio::spawn(tree.registration(env, second_id, second_instance));
    reached(&second_submitted, 1).await;
    let cancel_submitted = observe(
        env,
        HookPoint::EventSubmitted,
        event_filter(root_id, root_id, EventKindTag::CancelRoot),
    );
    let cancelled = tokio::spawn(cancel(env, root_id));
    reached(&cancel_submitted, 1).await;
    paused.release();
    let first = first
        .await
        .expect("the first registration should not panic");
    let second = second
        .await
        .expect("the second registration should not panic");
    let cancelled = cancelled.await.expect("the cancel should not panic");
    (first, second, cancelled)
}

/// How T3's leaf failure and root cancel interleave.
#[derive(Clone, Copy, Debug)]
enum FailureRace {
    /// Both in one round; the failure arrives first.
    SameRoundFailureFirst,

    /// Both in one round; the cancel arrives first.
    SameRoundCancelFirst,

    /// The failure is submitted while the cancel's round is in flight, before its database
    /// transaction.
    CancelInFlight,

    /// The failure's phase A runs while the cancel's round has committed but is not applied.
    CancelInCommitWindow,

    /// The failure is reported after the cancel was answered.
    AfterCancelReply,

    /// The cancel is submitted while the failure's round is in flight, before its database
    /// transaction.
    FailureInFlight,

    /// The cancel is submitted while the failure's round has committed but is not applied.
    FailureInCommitWindow,

    /// The cancel is submitted after the failure was answered.
    AfterFailureReply,
}

impl FailureRace {
    const ALL: [Self; 8] = [
        Self::SameRoundFailureFirst,
        Self::SameRoundCancelFirst,
        Self::CancelInFlight,
        Self::CancelInCommitWindow,
        Self::AfterCancelReply,
        Self::FailureInFlight,
        Self::FailureInCommitWindow,
        Self::AfterFailureReply,
    ];

    /// # Returns
    ///
    /// Whether the cancel is planned before the failure.
    const fn cancel_first(self) -> bool {
        !matches!(
            self,
            Self::FailureInFlight | Self::FailureInCommitWindow | Self::AfterFailureReply
        )
    }
}

/// The jobs of a T3 tree `R → {A → {L, S}, Q}`, where the leaf `L` fails.
struct FailureTree {
    root_id: JobId,
    middle_id: JobId,
    leaf_id: JobId,
    sibling_id: JobId,
    uncle_id: JobId,
    has_cleanup: HashMap<JobId, bool>,
}

impl FailureTree {
    /// Starts a tree with `cleanups`. With [`Cleanups::BystandersOnly`], `S` and `Q` have a cleanup
    /// task.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, cleanups: Cleanups) -> Self {
        let shape = |has_cleanup: bool| JobShape {
            with_cleanup: has_cleanup,
            ..ONE_TASK
        };
        let all = cleanups == Cleanups::All;
        let bystander = cleanups != Cleanups::None;
        let root_id = start_root(env, shape(all)).await;
        let [middle_id, uncle_id] =
            register_children(env, root_id, &[shape(all), shape(bystander)]).await[..]
        else {
            panic!("the root should have two children");
        };
        let [leaf_id, sibling_id] =
            register_children(env, middle_id, &[shape(all), shape(bystander)]).await[..]
        else {
            panic!("the middle job should have two children");
        };
        Self {
            root_id,
            middle_id,
            leaf_id,
            sibling_id,
            uncle_id,
            has_cleanup: HashMap::from([
                (root_id, all),
                (middle_id, all),
                (leaf_id, all),
                (sibling_id, bystander),
                (uncle_id, bystander),
            ]),
        }
    }

    /// # Returns
    ///
    /// The state that a cancel or an abort moves `job_id` to.
    fn target(&self, job_id: JobId) -> JobState {
        abort_target(self.has_cleanup[&job_id])
    }

    /// # Returns
    ///
    /// The states of the tree's jobs after the cancel.
    fn cancelled(&self) -> HashMap<JobId, JobState> {
        self.has_cleanup
            .keys()
            .map(|job_id| (*job_id, self.target(*job_id)))
            .collect()
    }

    /// # Returns
    ///
    /// The states of the tree's jobs after the leaf's failure.
    fn failed(&self) -> HashMap<JobId, JobState> {
        HashMap::from([
            (self.root_id, JobState::Failed),
            (self.middle_id, JobState::Failed),
            (self.leaf_id, JobState::Failed),
            (self.sibling_id, self.target(self.sibling_id)),
            (self.uncle_id, self.target(self.uncle_id)),
        ])
    }

    /// # Returns
    ///
    /// The error messages of the tree's jobs after the leaf's failure, by ascending job ID.
    fn failure_errors(&self) -> Vec<(JobId, String)> {
        let propagated = format!("descendant job {} failed: leaf failure", self.leaf_id);
        let mut errors = vec![
            (self.root_id, propagated.clone()),
            (self.middle_id, propagated),
            (self.leaf_id, "leaf failure".to_owned()),
        ];
        errors.sort_by_key(|(job_id, _)| job_id.get());
        errors
    }

    /// # Returns
    ///
    /// Whether the tree's committer stays alive after the winning event of `race`, because a job
    /// of the tree is `CleanupReady`.
    fn outlives(&self, race: FailureRace) -> bool {
        let states = if race.cancel_first() {
            self.cancelled()
        } else {
            self.failed()
        };
        states
            .values()
            .any(|state| *state == JobState::CleanupReady)
    }
}

/// T3: a root cancel races a leaf's failure. In one tree state transaction the cancel wins: the
/// whole tree is deactivated, every cleanup runs, and the failure is stale and stored nowhere.
/// Across tree state transactions, the event planned first wins.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_root_cancel_racing_a_leaf_failure_lets_the_first_planned_win() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for cleanups in [Cleanups::None, Cleanups::All, Cleanups::BystandersOnly] {
            for race in FailureRace::ALL {
                assert_cancel_races_failure(&env, cleanups, race).await?;
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Races the cancel of a T3 tree with its leaf's failure as `race` says, and asserts the outcome.
///
/// # Errors
///
/// Returns an error if the winning RPC fails.
async fn assert_cancel_races_failure(
    env: &TestEnv,
    cleanups: Cleanups,
    race: FailureRace,
) -> anyhow::Result<()> {
    let what = format!("{cleanups:?}, {race:?}");
    let tree = FailureTree::start(env, cleanups).await;
    let root_id = tree.root_id;
    let leaf_id = tree.leaf_id;
    let task_instance_id = instance(env, leaf_id, TaskId::Index(0)).await;
    let job_ids: Vec<JobId> = tree.has_cleanup.keys().copied().collect();
    let probes = TreeProbes::arm(env, root_id, &job_ids);
    let failure_event = event_filter(root_id, leaf_id, EventKindTag::TerminalFailure);
    let cancel_event = event_filter(root_id, root_id, EventKindTag::CancelRoot);
    let failures = observe(env, HookPoint::EventSubmitted, failure_event);
    let cancels = observe(env, HookPoint::EventSubmitted, cancel_event);
    let snapshot = CommitSnapshot::arm(
        env,
        root_id,
        if race.cancel_first() {
            EventKindTag::CancelRoot
        } else {
            EventKindTag::TerminalFailure
        },
    );

    let (failure, cancelled) = race_failure(env, &tree, race, task_instance_id).await;

    let outlives = tree.outlives(race);
    let (expected, expected_errors) = if race.cancel_first() {
        assert_job_state(
            &cancelled,
            tree.target(root_id),
            &format!("{what}: the cancel"),
        );
        let expected_stale = if matches!(race, FailureRace::AfterCancelReply) {
            StaleStateError::JobNoLongerRunning
        } else {
            regular_stale(tree.target(leaf_id))
        };
        assert_stale(&failure, &expected_stale, &format!("{what}: the failure"));
        cancelled?;
        (tree.cancelled(), Vec::new())
    } else {
        assert_job_state(&failure, JobState::Failed, &format!("{what}: the failure"));
        assert_stale(
            &cancelled,
            &StaleStateError::JobAlreadyTerminated(JobState::Failed),
            &format!("{what}: the cancel"),
        );
        failure?;
        (tree.failed(), tree.failure_errors())
    };

    assert_eq!(
        failures.hits(),
        usize::from(!matches!(race, FailureRace::AfterCancelReply)),
        "{what}: submitted failures"
    );
    assert_eq!(
        cancels.hits(),
        usize::from(!matches!(race, FailureRace::AfterFailureReply) || outlives),
        "{what}: submitted cancels"
    );
    let (num_rounds, last_round): (usize, &[EventKindTag]) = match race {
        FailureRace::SameRoundFailureFirst | FailureRace::SameRoundCancelFirst => (
            1,
            &[EventKindTag::CancelRoot, EventKindTag::TerminalFailure],
        ),
        FailureRace::CancelInFlight | FailureRace::CancelInCommitWindow if outlives => {
            (2, &[EventKindTag::TerminalFailure])
        }
        FailureRace::CancelInFlight
        | FailureRace::CancelInCommitWindow
        | FailureRace::AfterCancelReply => (1, &[EventKindTag::CancelRoot]),
        FailureRace::FailureInFlight
        | FailureRace::FailureInCommitWindow
        | FailureRace::AfterFailureReply
            if outlives =>
        {
            (2, &[EventKindTag::CancelRoot])
        }
        FailureRace::FailureInFlight
        | FailureRace::FailureInCommitWindow
        | FailureRace::AfterFailureReply => (1, &[EventKindTag::TerminalFailure]),
    };
    probes.assert_rounds(num_rounds, last_round, 1, &what);
    let applies: Vec<(JobId, usize)> = job_ids.iter().map(|job_id| (*job_id, 1)).collect();
    probes.assert_applies(&applies, job_ids.len(), &what);
    assert_rows(
        &snapshot.rows().await,
        &expected,
        &format!("{what}: at the winner's commit"),
    );
    finish_tree(env, root_id, &expected, &expected_errors, &what).await;
    Ok(())
}

/// Runs the failure of `tree`'s leaf, through its instance `task_instance_id`, and the cancel of
/// its root as `race` says.
///
/// # Returns
///
/// The results of the failure's report and of the cancel.
async fn race_failure(
    env: &TestEnv,
    tree: &FailureTree,
    race: FailureRace,
    task_instance_id: TaskInstanceId,
) -> (JobResult, JobResult) {
    let root_id = tree.root_id;
    let leaf_id = tree.leaf_id;
    let failure_event = event_filter(root_id, leaf_id, EventKindTag::TerminalFailure);
    let cancel_event = event_filter(root_id, root_id, EventKindTag::CancelRoot);
    let failure = fail(
        env,
        leaf_id,
        TaskId::Index(0),
        task_instance_id,
        "leaf failure",
    );
    let cancelled = cancel(env, root_id);
    match race {
        FailureRace::SameRoundFailureFirst
        | FailureRace::FailureInFlight
        | FailureRace::FailureInCommitWindow => {
            let point = match race {
                FailureRace::SameRoundFailureFirst => HookPoint::CommitterAfterReceive,
                FailureRace::FailureInFlight => HookPoint::CommitterBeforeDbTransaction,
                _ => HookPoint::CommitterAfterCommit,
            };
            Race::start(env, root_id, point, failure, cancelled, cancel_event)
                .await
                .finish()
                .await
        }
        FailureRace::SameRoundCancelFirst
        | FailureRace::CancelInFlight
        | FailureRace::CancelInCommitWindow => {
            let point = match race {
                FailureRace::SameRoundCancelFirst => HookPoint::CommitterAfterReceive,
                FailureRace::CancelInFlight => HookPoint::CommitterBeforeDbTransaction,
                _ => HookPoint::CommitterAfterCommit,
            };
            let (cancelled, failure) =
                Race::start(env, root_id, point, cancelled, failure, failure_event)
                    .await
                    .finish()
                    .await;
            (failure, cancelled)
        }
        FailureRace::AfterCancelReply => {
            let cancelled = cancelled.await;
            (failure.await, cancelled)
        }
        FailureRace::AfterFailureReply => {
            let failure = failure.await;
            (failure, cancelled.await)
        }
    }
}

/// How T7's cancel of the root and the commit of a `CommitReady` job interleave.
#[derive(Clone, Copy, Debug)]
enum CommitRace {
    /// The commit task's instance is registered, then the cancel is answered, then the commit
    /// succeeds.
    CancelThenCommit,

    /// The cancel is answered, then the commit task gets an instance, which succeeds.
    CancelThenNewInstanceCommit,

    /// The commit task's instance is registered, then the cancel is answered, then the commit
    /// fails for good.
    CancelThenCommitFailure,

    /// The commit's success and the cancel in one round; the commit arrives first.
    SameRoundCommitFirst,

    /// The commit's success and the cancel in one round; the cancel arrives first.
    SameRoundCancelFirst,

    /// The commit's failure and the cancel in one round; the failure arrives first.
    SameRoundCommitFailure,

    /// The commit's success is submitted while the cancel's round is in flight, before its
    /// database transaction.
    CancelInFlight,

    /// The commit's success is submitted while the cancel's round has committed but is not
    /// applied.
    CancelInCommitWindow,

    /// The cancel is submitted while the commit's round is in flight, before its database
    /// transaction.
    CommitInFlight,

    /// The cancel is submitted after the commit's success was answered.
    AfterCommitReply,
}

impl CommitRace {
    const ALL: [Self; 10] = [
        Self::CancelThenCommit,
        Self::CancelThenNewInstanceCommit,
        Self::CancelThenCommitFailure,
        Self::SameRoundCommitFirst,
        Self::SameRoundCancelFirst,
        Self::SameRoundCommitFailure,
        Self::CancelInFlight,
        Self::CancelInCommitWindow,
        Self::CommitInFlight,
        Self::AfterCommitReply,
    ];

    /// # Returns
    ///
    /// Whether the commit is planned before the cancel.
    const fn commit_first(self) -> bool {
        matches!(self, Self::CommitInFlight | Self::AfterCommitReply)
    }

    /// # Returns
    ///
    /// Whether the commit task fails instead of succeeding.
    const fn commit_fails(self) -> bool {
        matches!(
            self,
            Self::CancelThenCommitFailure | Self::SameRoundCommitFailure
        )
    }

    /// # Returns
    ///
    /// Whether the cancel is answered before the commit is reported.
    const fn cancel_answered_first(self) -> bool {
        matches!(
            self,
            Self::CancelThenCommit
                | Self::CancelThenNewInstanceCommit
                | Self::CancelThenCommitFailure
        )
    }
}

/// The jobs of a T7 tree `R → {P → C, Q}`, where `P` has a commit task and is `CommitReady`, and
/// the uncle `Q` is optional.
struct CommitTree {
    root_id: JobId,
    parent_id: JobId,
    child_id: JobId,
    uncle_id: Option<JobId>,
    has_cleanup: bool,
}

impl CommitTree {
    /// Starts a tree whose jobs, `C` excepted, all have a cleanup task if `has_cleanup`, and runs
    /// `C` so that `P` is `CommitReady`, then drains `P`'s commit task.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, has_cleanup: bool, with_uncle: bool) -> Self {
        let shape = JobShape {
            with_cleanup: has_cleanup,
            ..ONE_TASK
        };
        let root_id = start_root(env, shape).await;
        let parent_shape = JobShape {
            with_commit: true,
            ..shape
        };
        let shapes: &[JobShape] = if with_uncle {
            &[parent_shape, shape]
        } else {
            &[parent_shape]
        };
        let children = register_children(env, root_id, shapes).await;
        let parent_id = children[0];
        let child_id = register_children(env, parent_id, &[ONE_TASK]).await[0];
        assert_eq!(
            run_task(env, child_id, &value(0x30)).await,
            JobState::Succeeded
        );
        drain_lane(env, Lane::Commit, &[(parent_id, TaskId::Commit)]).await;
        Self {
            root_id,
            parent_id,
            child_id,
            uncle_id: children.get(1).copied(),
            has_cleanup,
        }
    }

    /// # Returns
    ///
    /// The jobs whose cache applies a T7 race can change.
    fn job_ids(&self) -> Vec<JobId> {
        let mut job_ids = vec![self.root_id, self.parent_id, self.child_id];
        job_ids.extend(self.uncle_id);
        job_ids
    }

    /// # Returns
    ///
    /// The states of the tree's jobs once `race` is decided, with `parent` as `P`'s state.
    fn states(&self, race: CommitRace, parent: JobState) -> HashMap<JobId, JobState> {
        let target = abort_target(self.has_cleanup);
        let root = if race.commit_first() && self.uncle_id.is_none() {
            JobState::Succeeded
        } else {
            target
        };
        let mut states = HashMap::from([
            (self.root_id, root),
            (self.parent_id, parent),
            (self.child_id, JobState::Succeeded),
        ]);
        states.extend(self.uncle_id.map(|uncle_id| (uncle_id, target)));
        states
    }
}

/// T7, non-root: a cancel skips a `CommitReady` non-root job, which then finishes its commit
/// (successfully or not) without propagating to its cancelled parent; a commit planned before the
/// cancel climbs as usual.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_cancel_skips_a_commit_ready_non_root_job_which_then_finishes() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for has_cleanup in [false, true] {
            for race in CommitRace::ALL {
                assert_cancel_skips_commit_ready_job(&env, has_cleanup, true, race).await?;
                if race.commit_first() {
                    assert_cancel_skips_commit_ready_job(&env, has_cleanup, false, race).await?;
                }
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Races the cancel of a T7 tree with the commit of its `CommitReady` non-root job as `race` says,
/// and asserts the outcome.
///
/// # Errors
///
/// Returns an error if the commit's RPC fails.
async fn assert_cancel_skips_commit_ready_job(
    env: &TestEnv,
    has_cleanup: bool,
    with_uncle: bool,
    race: CommitRace,
) -> anyhow::Result<()> {
    let what = format!("cleanup: {has_cleanup}, uncle: {with_uncle}, {race:?}");
    let tree = CommitTree::start(env, has_cleanup, with_uncle).await;
    let parent_id = tree.parent_id;
    let parent_final = if race.commit_fails() {
        JobState::Failed
    } else {
        JobState::Succeeded
    };
    let cancel_commits = !race.commit_first() || with_uncle;
    let probes = TreeProbes::arm(env, tree.root_id, &tree.job_ids());
    let snapshot =
        cancel_commits.then(|| CommitSnapshot::arm(env, tree.root_id, EventKindTag::CancelRoot));

    let (commit, cancelled) = race_commit(env, &tree, race, &what).await;

    assert_job_state(&commit, parent_final, &format!("{what}: the commit"));
    commit?;
    let expected = tree.states(race, parent_final);
    if cancel_commits {
        assert_job_state(
            &cancelled,
            abort_target(has_cleanup),
            &format!("{what}: the cancel"),
        );
    } else {
        assert_stale(
            &cancelled,
            &StaleStateError::JobAlreadyTerminated(JobState::Succeeded),
            &format!("{what}: the cancel"),
        );
    }

    let commit_kind = if race.commit_fails() {
        EventKindTag::TerminalFailure
    } else {
        EventKindTag::CommitDone
    };
    let (num_rounds, last_round) = match race {
        CommitRace::SameRoundCommitFirst
        | CommitRace::SameRoundCancelFirst
        | CommitRace::SameRoundCommitFailure => (1, vec![EventKindTag::CancelRoot, commit_kind]),
        CommitRace::CommitInFlight | CommitRace::AfterCommitReply if with_uncle => {
            (2, vec![EventKindTag::CancelRoot])
        }
        CommitRace::CommitInFlight | CommitRace::AfterCommitReply => (1, vec![commit_kind]),
        _ => (2, vec![commit_kind]),
    };
    probes.assert_rounds(num_rounds, &last_round, num_rounds, &what);
    let mut applies = vec![(tree.root_id, 1), (parent_id, 1), (tree.child_id, 0)];
    applies.extend(tree.uncle_id.map(|uncle_id| (uncle_id, 1)));
    probes.assert_applies(&applies, 2 + usize::from(with_uncle), &what);
    if let Some(snapshot) = snapshot {
        let parent_at_cancel = match race {
            CommitRace::SameRoundCommitFirst
            | CommitRace::SameRoundCancelFirst
            | CommitRace::SameRoundCommitFailure => parent_final,
            race if race.commit_first() => JobState::Succeeded,
            _ => JobState::CommitReady,
        };
        assert_rows(
            &snapshot.rows().await,
            &tree.states(race, parent_at_cancel),
            &format!("{what}: at the cancel's commit"),
        );
    }
    let errors = if race.commit_fails() {
        vec![(parent_id, "commit failure".to_owned())]
    } else {
        Vec::new()
    };
    finish_tree(env, tree.root_id, &expected, &errors, &what).await;
    Ok(())
}

/// Runs the commit of `tree`'s `CommitReady` non-root job and the cancel of its root as `race`
/// says.
///
/// # Returns
///
/// The results of the commit's report and of the cancel.
async fn race_commit(
    env: &TestEnv,
    tree: &CommitTree,
    race: CommitRace,
    what: &str,
) -> (JobResult, JobResult) {
    let root_id = tree.root_id;
    let parent_id = tree.parent_id;
    let cancelled = cancel(env, root_id);
    let cancel_event = event_filter(root_id, root_id, EventKindTag::CancelRoot);
    if race.cancel_answered_first() {
        let commit_instance = if matches!(race, CommitRace::CancelThenNewInstanceCommit) {
            None
        } else {
            Some(instance(env, parent_id, TaskId::Commit).await)
        };
        let cancelled = cancelled.await;
        assert_states(
            env,
            &tree.states(race, JobState::CommitReady),
            &format!("{what}: after the cancel"),
        )
        .await;
        let commit_instance = match commit_instance {
            Some(commit_instance) => commit_instance,
            None => instance(env, parent_id, TaskId::Commit).await,
        };
        let commit = if race.commit_fails() {
            fail(
                env,
                parent_id,
                TaskId::Commit,
                commit_instance,
                "commit failure",
            )
            .await
        } else {
            succeed_commit(env, parent_id, commit_instance).await
        };
        return (commit, cancelled);
    }
    let commit_instance = instance(env, parent_id, TaskId::Commit).await;
    let commit = succeed_commit(env, parent_id, commit_instance);
    let commit_event = event_filter(root_id, parent_id, EventKindTag::CommitDone);
    match race {
        CommitRace::SameRoundCommitFirst | CommitRace::CommitInFlight => {
            let point = if matches!(race, CommitRace::SameRoundCommitFirst) {
                HookPoint::CommitterAfterReceive
            } else {
                HookPoint::CommitterBeforeDbTransaction
            };
            Race::start(env, root_id, point, commit, cancelled, cancel_event)
                .await
                .finish()
                .await
        }
        CommitRace::SameRoundCommitFailure => {
            let failure = fail(
                env,
                parent_id,
                TaskId::Commit,
                commit_instance,
                "commit failure",
            );
            Race::start(
                env,
                root_id,
                HookPoint::CommitterAfterReceive,
                failure,
                cancelled,
                cancel_event,
            )
            .await
            .finish()
            .await
        }
        CommitRace::SameRoundCancelFirst
        | CommitRace::CancelInFlight
        | CommitRace::CancelInCommitWindow => {
            let point = match race {
                CommitRace::SameRoundCancelFirst => HookPoint::CommitterAfterReceive,
                CommitRace::CancelInFlight => HookPoint::CommitterBeforeDbTransaction,
                _ => HookPoint::CommitterAfterCommit,
            };
            let (cancelled, commit) =
                Race::start(env, root_id, point, cancelled, commit, commit_event)
                    .await
                    .finish()
                    .await;
            (commit, cancelled)
        }
        _ => {
            let commit = commit.await;
            (commit, cancelled.await)
        }
    }
}

/// How T7's cancel of a `CommitReady` root and the root's commit interleave.
#[derive(Clone, Copy, Debug)]
enum RootCommitRace {
    /// The cancel is answered, then the commit task's registered instance succeeds.
    CancelThenLateCommit,

    /// The cancel is answered, then the commit task's registered instance fails for good.
    CancelThenLateCommitFailure,

    /// The cancel is answered, then the commit task's published entry asks for an instance.
    CancelThenNewInstance,

    /// The commit's success and the cancel in one round; the commit arrives first.
    SameRoundCommitFirst,

    /// The commit's success and the cancel in one round; the cancel arrives first.
    SameRoundCancelFirst,

    /// The commit's success is submitted while the cancel's round is in flight, before its
    /// database transaction.
    CancelInFlight,

    /// The cancel is submitted while the commit's round is in flight, before its database
    /// transaction.
    CommitInFlight,

    /// The cancel is submitted after the commit's success was answered.
    AfterCommitReply,
}

impl RootCommitRace {
    const ALL: [Self; 8] = [
        Self::CancelThenLateCommit,
        Self::CancelThenLateCommitFailure,
        Self::CancelThenNewInstance,
        Self::SameRoundCommitFirst,
        Self::SameRoundCancelFirst,
        Self::CancelInFlight,
        Self::CommitInFlight,
        Self::AfterCommitReply,
    ];

    /// # Returns
    ///
    /// Whether the commit is planned before the cancel.
    const fn commit_first(self) -> bool {
        matches!(self, Self::CommitInFlight | Self::AfterCommitReply)
    }
}

/// T7, root: a cancel of a `CommitReady` root deactivates it, to `CleanupReady` or, through the
/// `CommitReady → Cancelled` edge, to `Cancelled`; the root's commit is then stale. A commit
/// planned before the cancel succeeds the root, and the cancel is stale.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_cancel_deactivates_a_commit_ready_root() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for has_cleanup in [false, true] {
            for race in RootCommitRace::ALL {
                assert_cancel_deactivates_commit_ready_root(&env, has_cleanup, race).await?;
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Races the cancel of a `CommitReady` root with its commit as `race` says, and asserts the
/// outcome.
///
/// # Errors
///
/// Returns an error if the winning RPC fails, or if the root's outputs cannot be read after a
/// successful commit.
async fn assert_cancel_deactivates_commit_ready_root(
    env: &TestEnv,
    has_cleanup: bool,
    race: RootCommitRace,
) -> anyhow::Result<()> {
    let what = format!("cleanup: {has_cleanup}, {race:?}");
    let root_value = value(0x40);
    let root_id = start_root(
        env,
        JobShape {
            with_commit: true,
            with_cleanup: has_cleanup,
            ..ONE_TASK
        },
    )
    .await;
    assert_eq!(
        run_task(env, root_id, &root_value).await,
        JobState::CommitReady
    );
    drain_lane(env, Lane::Commit, &[(root_id, TaskId::Commit)]).await;
    let target = abort_target(has_cleanup);
    let probes = TreeProbes::arm(env, root_id, &[root_id]);
    let snapshot =
        (!race.commit_first()).then(|| CommitSnapshot::arm(env, root_id, EventKindTag::CancelRoot));

    let (commit, cancelled) = race_root_commit(env, root_id, race).await;

    let root_state = if race.commit_first() {
        assert_job_state(&commit, JobState::Succeeded, &format!("{what}: the commit"));
        assert_stale(
            &cancelled,
            &StaleStateError::JobAlreadyTerminated(JobState::Succeeded),
            &format!("{what}: the cancel"),
        );
        commit?;
        assert_eq!(
            env.service.get_job_outputs(root_id).await?,
            one_task_outputs(&root_value),
            "{what}: the root's outputs"
        );
        JobState::Succeeded
    } else {
        assert_job_state(&cancelled, target, &format!("{what}: the cancel"));
        let expected_stale = match race {
            RootCommitRace::CancelThenLateCommit
            | RootCommitRace::CancelThenLateCommitFailure
            | RootCommitRace::CancelThenNewInstance => StaleStateError::JobNoLongerCommitReady,
            _ if target.is_terminal() => StaleStateError::JobAlreadyTerminated(target),
            _ => StaleStateError::JobNoLongerCommitReady,
        };
        assert_stale(&commit, &expected_stale, &format!("{what}: the commit"));
        cancelled?;
        target
    };

    let (num_rounds, last_round): (usize, &[EventKindTag]) = match race {
        RootCommitRace::SameRoundCommitFirst | RootCommitRace::SameRoundCancelFirst => {
            (1, &[EventKindTag::CancelRoot, EventKindTag::CommitDone])
        }
        RootCommitRace::CancelInFlight if has_cleanup => (2, &[EventKindTag::CommitDone]),
        RootCommitRace::CommitInFlight | RootCommitRace::AfterCommitReply => {
            (1, &[EventKindTag::CommitDone])
        }
        _ => (1, &[EventKindTag::CancelRoot]),
    };
    probes.assert_rounds(num_rounds, last_round, 1, &what);
    probes.assert_applies(&[(root_id, 1)], 1, &what);
    let expected = HashMap::from([(root_id, root_state)]);
    if let Some(snapshot) = snapshot {
        assert_rows(
            &snapshot.rows().await,
            &expected,
            &format!("{what}: at the cancel's commit"),
        );
    }
    finish_tree(env, root_id, &expected, &[], &what).await;
    Ok(())
}

/// Runs the commit of the `CommitReady` root `root_id` and its cancel as `race` says.
///
/// # Returns
///
/// The results of the commit's report (or, for [`RootCommitRace::CancelThenNewInstance`], of the
/// commit task's instance creation) and of the cancel.
async fn race_root_commit(
    env: &TestEnv,
    root_id: JobId,
    race: RootCommitRace,
) -> (JobResult, JobResult) {
    let cancelled = cancel(env, root_id);
    if matches!(race, RootCommitRace::CancelThenNewInstance) {
        let cancelled = cancelled.await;
        let created = create_instance(env, root_id, TaskId::Commit)
            .await
            .map(|_| JobState::CommitReady);
        return (created, cancelled);
    }
    let commit_instance = instance(env, root_id, TaskId::Commit).await;
    let commit = succeed_commit(env, root_id, commit_instance);
    let cancel_event = event_filter(root_id, root_id, EventKindTag::CancelRoot);
    let commit_event = event_filter(root_id, root_id, EventKindTag::CommitDone);
    match race {
        RootCommitRace::CancelThenLateCommit => {
            let cancelled = cancelled.await;
            (commit.await, cancelled)
        }
        RootCommitRace::CancelThenLateCommitFailure => {
            let cancelled = cancelled.await;
            let failure = fail(
                env,
                root_id,
                TaskId::Commit,
                commit_instance,
                "late commit failure",
            );
            (failure.await, cancelled)
        }
        RootCommitRace::SameRoundCommitFirst | RootCommitRace::CommitInFlight => {
            let point = if matches!(race, RootCommitRace::SameRoundCommitFirst) {
                HookPoint::CommitterAfterReceive
            } else {
                HookPoint::CommitterBeforeDbTransaction
            };
            Race::start(env, root_id, point, commit, cancelled, cancel_event)
                .await
                .finish()
                .await
        }
        RootCommitRace::SameRoundCancelFirst | RootCommitRace::CancelInFlight => {
            let point = if matches!(race, RootCommitRace::SameRoundCancelFirst) {
                HookPoint::CommitterAfterReceive
            } else {
                HookPoint::CommitterBeforeDbTransaction
            };
            let (cancelled, commit) =
                Race::start(env, root_id, point, cancelled, commit, commit_event)
                    .await
                    .finish()
                    .await;
            (commit, cancelled)
        }
        _ => {
            let commit = commit.await;
            (commit, cancelled.await)
        }
    }
}
