//! Completion and failure races on jobs and job trees, made deterministic through the test hooks:
//! two tasks failing one job (T1), siblings completing together (T4), a straggler failing a task
//! that already succeeded (T5), and a parent with a commit task (T6).

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use spider_core::job::JobState;
use spider_core::task::DataTypeDescriptor;
use spider_core::task::ExecutionPolicy;
use spider_core::task::TaskDescriptor;
use spider_core::task::TaskIndex;
use spider_core::task::TaskState;
use spider_core::task::TdlContext;
use spider_core::task::TerminationTaskDescriptor;
use spider_core::task::ValueTypeDescriptor;
use spider_core::types::id::JobId;
use spider_core::types::id::TaskId;
use spider_core::types::id::TaskInstanceId;
use spider_core::types::io::SerializedTaskOutputs;
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
    max_num_instances: u32,
    max_num_retry: u32,
    with_commit: bool,
    with_cleanup: bool,
}

const ONE_TASK: JobShape = JobShape {
    num_tasks: 1,
    max_num_instances: 1,
    max_num_retry: 0,
    with_commit: false,
    with_cleanup: false,
};

/// A lane of the inbound queue.
#[derive(Clone, Copy, Debug)]
enum Lane {
    Task,
    Commit,
    Cleanup,
}

/// Observers of the committer of one tree: its rounds, its database transactions, and the cache
/// applies of some of its jobs.
struct TreeProbes {
    rounds: ArmedHook,
    commits: ArmedHook,
    applies: HashMap<JobId, ArmedHook>,
}

impl TreeProbes {
    /// # Returns
    ///
    /// Probes of the tree rooted at `root_id` that observe the cache applies of `job_ids`.
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

    /// # Returns
    ///
    /// The number of times the committer has applied a transition of `job_id` to the cache.
    ///
    /// # Panics
    ///
    /// Panics if the probes do not observe `job_id`.
    fn applies(&self, job_id: JobId) -> usize {
        self.applies
            .get(&job_id)
            .expect("the probes should observe the job")
            .hits()
    }

    /// Asserts that the committer planned exactly one round, holding the events of `kinds` in this
    /// order.
    fn assert_one_round(&self, kinds: &[EventKindTag], what: &str) {
        assert_eq!(self.rounds.hits(), 1, "{what}: number of rounds");
        assert_eq!(
            self.rounds
                .last_context()
                .expect("a round should have been observed")
                .event_kinds,
            kinds,
            "{what}: the events of the round"
        );
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
        max_num_instances: shape.max_num_instances,
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
/// A task output value that is unique for `tag` and `index`.
///
/// # Panics
///
/// Panics if `index` does not fit in 16 bits.
fn value(tag: u8, index: usize) -> Vec<u8> {
    let index = u16::try_from(index).expect("test indices should fit in 16 bits");
    let [low, high] = index.to_le_bytes();
    vec![tag, low, high, 0x5a]
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
/// The outputs that the storage service stores for the tasks that reported `values`, in order.
///
/// # Panics
///
/// Panics if the deserialization fails.
fn task_outputs<'value_lifetime>(
    values: impl IntoIterator<Item = &'value_lifetime Vec<u8>>,
) -> Vec<TaskOutput> {
    values
        .into_iter()
        .flat_map(|value| {
            TaskOutputsSerializer::deserialize(&serialized_output(value))
                .expect("output deserialization should succeed")
        })
        .collect()
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
/// The ID of a new instance of `task_id` of `job_id`.
///
/// # Panics
///
/// Panics if the instance creation fails.
async fn instance(env: &TestEnv, job_id: JobId, task_id: TaskId) -> TaskInstanceId {
    env.service
        .create_task_instance(
            env.service.session_id(),
            job_id,
            task_id,
            env.execution_manager_id,
        )
        .await
        .expect("the instance creation should succeed")
        .task_instance_id
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
/// A future that reports the failure of `task_instance_id` of task `task_index` of `job_id` with
/// `message`.
fn fail(
    env: &TestEnv,
    job_id: JobId,
    task_index: TaskIndex,
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
                TaskId::Index(task_index),
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

/// Runs a new instance of the only task of `parent_id` to success with the output `value` and one
/// child per shape, and drains the children's ready tasks.
///
/// # Returns
///
/// The IDs of the registered children, by ascending ID, which is the order of `shapes`.
///
/// # Panics
///
/// Panics if the report fails, does not block the parent on its children, or the children's tasks
/// are not published exactly once.
async fn register_children(
    env: &TestEnv,
    parent_id: JobId,
    value: &[u8],
    shapes: &[JobShape],
) -> Vec<JobId> {
    let task_instance_id = instance(env, parent_id, TaskId::Index(0)).await;
    let children = shapes.iter().copied().map(child).collect();
    assert_eq!(
        succeed(env, parent_id, 0, task_instance_id, value, children)
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

/// Runs a new instance of task `task_index` of `job_id` to success with the output `value`.
///
/// # Returns
///
/// The job state after the report.
///
/// # Panics
///
/// Panics if the report fails.
async fn run_task(env: &TestEnv, job_id: JobId, task_index: TaskIndex, value: &[u8]) -> JobState {
    let task_instance_id = instance(env, job_id, TaskId::Index(task_index)).await;
    succeed(env, job_id, task_index, task_instance_id, value, Vec::new())
        .await
        .expect("the report should succeed")
}

/// Drains the published cleanup task of `job_id` and runs a new instance of it to success.
///
/// # Returns
///
/// The job state after the report.
///
/// # Panics
///
/// Panics if the cleanup task is not published exactly once or the report fails.
async fn run_cleanup(env: &TestEnv, job_id: JobId) -> JobState {
    drain_lane(env, Lane::Cleanup, &[(job_id, TaskId::Cleanup)]).await;
    let task_instance_id = instance(env, job_id, TaskId::Cleanup).await;
    env.service
        .succeed_cleanup_task_instance(env.service.session_id(), job_id, task_instance_id)
        .await
        .expect("the cleanup report should succeed")
}

/// Drains the published commit task of `job_id` and registers an instance of it.
///
/// # Returns
///
/// A tuple on success, containing:
///
/// * The ID of the instance.
/// * The outputs that the instance's execution context hands to the commit task.
///
/// # Panics
///
/// Panics if the commit task is not published exactly once, the instance creation fails, or its
/// context does not hold task outputs.
async fn commit_instance(env: &TestEnv, job_id: JobId) -> (TaskInstanceId, Vec<TaskOutput>) {
    drain_lane(env, Lane::Commit, &[(job_id, TaskId::Commit)]).await;
    let context = env
        .service
        .create_task_instance(
            env.service.session_id(),
            job_id,
            TaskId::Commit,
            env.execution_manager_id,
        )
        .await
        .expect("the commit instance creation should succeed");
    let outputs = SerializedTaskOutputs::deserialize_from_raw(&context.serialized_task_io)
        .expect("the commit task's context should hold task outputs");
    (context.task_instance_id, outputs)
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
/// The outputs stored in `job_id`'s row, if any.
///
/// # Panics
///
/// Panics if the query or the deserialization fails.
async fn row_outputs(env: &TestEnv, job_id: JobId) -> Option<Vec<TaskOutput>> {
    sqlx::query_scalar::<_, Option<Vec<u8>>>(
        "SELECT `serialized_job_outputs` FROM `jobs` WHERE `id` = ?;",
    )
    .bind(job_id)
    .fetch_one(&env.pool)
    .await
    .expect("the outputs query should succeed")
    .map(|raw| {
        SerializedTaskOutputs::deserialize_from_raw(&raw)
            .expect("the stored outputs should deserialize")
    })
}

/// # Returns
///
/// The outputs of the succeeded job `job_id` in the job cache.
///
/// # Panics
///
/// Panics if the job is not cached or its outputs cannot be read.
async fn cached_outputs(env: &TestEnv, job_id: JobId) -> Vec<TaskOutput> {
    env.service
        .tree_env()
        .job_cache
        .get(job_id)
        .await
        .expect("the job should be cached")
        .get_outputs()
        .await
        .expect("the cached outputs should be readable")
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

/// Asserts that every job of `expected` has its state in the cache and in its row, that the row's
/// end time is set exactly when the state is terminal, and that the service has not shut down
/// fatally.
async fn assert_states(env: &TestEnv, expected: &[(JobId, JobState)]) {
    for (job_id, state) in expected {
        assert_eq!(
            env.cached_job_state(*job_id).await,
            Some(*state),
            "cached state of job {job_id}"
        );
        let (row_state, has_ended) = sqlx::query_as::<_, (JobState, i64)>(
            "SELECT `state`, `ended_at` IS NOT NULL FROM `jobs` WHERE `id` = ?;",
        )
        .bind(job_id)
        .fetch_one(&env.pool)
        .await
        .expect("the row state query should succeed");
        assert_eq!(row_state, *state, "row state of job {job_id}");
        assert_eq!(
            has_ended != 0,
            state.is_terminal(),
            "end time of job {job_id} in state {state:?}"
        );
    }
    assert!(
        !env.is_fatal(),
        "the service should not have shut down fatally"
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
fn assert_stale(result: &JobResult, expected: &StaleStateError, what: &str) {
    match result {
        Err(StorageServerError::Cache(CacheError::StaleState(error)))
            if error.to_string() == expected.to_string() => {}
        other => panic!("{what}: expected the stale error `{expected}`, got {other:?}"),
    }
}

/// Runs `first` and then `second` so that `second`'s event, matched by `second_event`, is submitted
/// while `first`'s round of the tree rooted at `root_id` is paused at `point`:
///
/// * At [`HookPoint::CommitterAfterReceive`], both events are planned in one round.
/// * At [`HookPoint::CommitterBeforeDbTransaction`], `second`'s event is queued while `first`'s
///   round is in flight, and is planned after it.
/// * At [`HookPoint::CommitterAfterCommit`], `second` runs while `first`'s round has committed but
///   is not applied to the cache yet, and its event is planned after that round.
///
/// # Type Parameters
///
/// * `FirstType` - The type of the future that submits the first event.
/// * `SecondType` - The type of the future that submits the second event.
///
/// # Returns
///
/// The results of `first` and `second`.
///
/// # Panics
///
/// Panics if either future panics.
async fn race<
    FirstType: Future<Output: Send + 'static> + Send + 'static,
    SecondType: Future<Output: Send + 'static> + Send + 'static,
>(
    env: &TestEnv,
    root_id: JobId,
    point: HookPoint,
    first: FirstType,
    second: SecondType,
    second_event: HookFilter,
) -> (FirstType::Output, SecondType::Output) {
    let paused = pause(env, point, HookFilter::root(root_id));
    let first = tokio::spawn(first);
    reached(&paused, 1).await;
    // Armed only now, so that the first event's own submission is not counted.
    let submitted = observe(env, HookPoint::EventSubmitted, second_event);
    let second = tokio::spawn(second);
    reached(&submitted, 1).await;
    paused.release();
    let first = first.await.expect("the first RPC should not panic");
    let second = second.await.expect("the second RPC should not panic");
    assert_eq!(paused.hits(), 1);
    (first, second)
}

/// The job whose two tasks fail in T1.
#[derive(Clone, Copy, Debug)]
enum FailingJob {
    /// A plain root job.
    Root,

    /// A leaf `J` in the tree `R → A → {J, S}`, where the sibling `S` has a cleanup task.
    Leaf,
}

/// When T1's second failure reaches the committer.
#[derive(Clone, Copy, Debug)]
enum SecondFailure {
    /// In the round of the first failure.
    SameRound,

    /// Queued while the first failure's round is in flight, before its database transaction.
    QueuedBehind,

    /// Through phase A while the first failure's round has committed but is not applied yet.
    InCommitWindow,

    /// After the first failure has been answered.
    AfterReply,
}

/// The jobs of a T1 tree.
struct FailureTree {
    root_id: JobId,
    job_id: JobId,
    ancestor_ids: Vec<JobId>,
    sibling_id: Option<JobId>,
}

impl FailureTree {
    /// Starts a tree whose two-task job is `failing_job`.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, failing_job: FailingJob) -> Self {
        let two_tasks = JobShape {
            num_tasks: 2,
            ..ONE_TASK
        };
        match failing_job {
            FailingJob::Root => {
                let job_id = start_root(env, two_tasks).await;
                Self {
                    root_id: job_id,
                    job_id,
                    ancestor_ids: Vec::new(),
                    sibling_id: None,
                }
            }
            FailingJob::Leaf => {
                let root_id = start_root(env, ONE_TASK).await;
                let middle_id =
                    register_children(env, root_id, &value(0xa0, 0), &[ONE_TASK]).await[0];
                let with_cleanup = JobShape {
                    with_cleanup: true,
                    ..ONE_TASK
                };
                let leaf_ids =
                    register_children(env, middle_id, &value(0xa1, 0), &[two_tasks, with_cleanup])
                        .await;
                Self {
                    root_id,
                    job_id: leaf_ids[0],
                    ancestor_ids: vec![root_id, middle_id],
                    sibling_id: Some(leaf_ids[1]),
                }
            }
        }
    }

    /// # Returns
    ///
    /// Every job of the tree.
    fn job_ids(&self) -> Vec<JobId> {
        let mut job_ids = self.ancestor_ids.clone();
        job_ids.push(self.job_id);
        job_ids.extend(self.sibling_id);
        job_ids
    }
}

/// T1: two different tasks of one job fail, in the same tree state transaction and in different
/// ones.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn two_tasks_failing_one_job_fail_it_once() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for failing_job in [FailingJob::Root, FailingJob::Leaf] {
            for second_failure in [
                SecondFailure::SameRound,
                SecondFailure::QueuedBehind,
                SecondFailure::InCommitWindow,
                SecondFailure::AfterReply,
            ] {
                assert_two_failures_fail_the_job_once(&env, failing_job, second_failure).await?;
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Fails tasks 0 and 1 of a two-task job of `failing_job`, the second as `second_failure` says.
///
/// # Errors
///
/// Returns an error if the first failure's report or the sibling's cleanup report fails.
async fn assert_two_failures_fail_the_job_once(
    env: &TestEnv,
    failing_job: FailingJob,
    second_failure: SecondFailure,
) -> anyhow::Result<()> {
    let what = format!("{failing_job:?}, second failure {second_failure:?}");
    let tree = FailureTree::start(env, failing_job).await;
    let first_instance = instance(env, tree.job_id, TaskId::Index(0)).await;
    let second_instance = instance(env, tree.job_id, TaskId::Index(1)).await;
    let probes = TreeProbes::arm(env, tree.root_id, &tree.job_ids());
    let failure_events = observe(
        env,
        HookPoint::EventSubmitted,
        event_filter(tree.root_id, tree.job_id, EventKindTag::TerminalFailure),
    );

    let first = fail(env, tree.job_id, 0, first_instance, "first failure");
    let second = fail(env, tree.job_id, 1, second_instance, "second failure");
    let point = match second_failure {
        SecondFailure::SameRound => Some(HookPoint::CommitterAfterReceive),
        SecondFailure::QueuedBehind => Some(HookPoint::CommitterBeforeDbTransaction),
        SecondFailure::InCommitWindow => Some(HookPoint::CommitterAfterCommit),
        SecondFailure::AfterReply => None,
    };
    let (first, second) = if let Some(point) = point {
        let failure_filter = event_filter(tree.root_id, tree.job_id, EventKindTag::TerminalFailure);
        race(env, tree.root_id, point, first, second, failure_filter).await
    } else {
        let first = first.await;
        (first, second.await)
    };

    assert_eq!(first?, JobState::Failed, "{what}");
    let (expected_stale, num_failure_events) = if point.is_some() {
        (StaleStateError::JobAlreadyTerminated(JobState::Failed), 2)
    } else {
        (StaleStateError::JobNoLongerRunning, 1)
    };
    assert_stale(
        &second,
        &expected_stale,
        &format!("{what}: the second failure"),
    );
    assert_eq!(
        failure_events.hits(),
        num_failure_events,
        "{what}: failure events"
    );
    match (failing_job, second_failure) {
        (_, SecondFailure::SameRound) => probes.assert_one_round(
            &[EventKindTag::TerminalFailure, EventKindTag::TerminalFailure],
            &what,
        ),
        // The aborted sibling keeps the leaf's tree, and so its committer, alive for the queued
        // second failure. A root is quiescent after the first failure, so its committer exits and
        // answers the queued event without planning it.
        (FailingJob::Leaf, SecondFailure::QueuedBehind | SecondFailure::InCommitWindow) => {
            assert_eq!(probes.rounds.hits(), 2, "{what}: number of rounds");
        }
        _ => assert_eq!(probes.rounds.hits(), 1, "{what}: number of rounds"),
    }
    assert_eq!(probes.commits.hits(), 1, "{what}: database transactions");
    for job_id in tree.job_ids() {
        assert_eq!(probes.applies(job_id), 1, "{what}: applies of job {job_id}");
    }
    assert_failed_tree(env, &tree, &what).await
}

/// Asserts that `tree`'s job failed with the first failure's message, that its ancestors failed
/// for it and its sibling was aborted, then runs the sibling's cleanup.
///
/// # Errors
///
/// Returns an error if the sibling's cleanup report fails.
async fn assert_failed_tree(env: &TestEnv, tree: &FailureTree, what: &str) -> anyhow::Result<()> {
    let job_id = tree.job_id;
    let mut expected_errors = vec![(job_id, "first failure".to_owned())];
    let mut expected_states = vec![(job_id, JobState::Failed)];
    for ancestor_id in &tree.ancestor_ids {
        expected_errors.push((
            *ancestor_id,
            format!("descendant job {job_id} failed: first failure"),
        ));
        expected_states.push((*ancestor_id, JobState::Failed));
    }
    expected_errors.sort_by_key(|(id, _)| id.get());
    assert_eq!(
        tree_errors(env, tree.root_id).await,
        expected_errors,
        "{what}"
    );
    expected_states.extend(
        tree.sibling_id
            .map(|sibling_id| (sibling_id, JobState::CleanupReady)),
    );
    assert_states(env, &expected_states).await;
    if let Some(sibling_id) = tree.sibling_id {
        assert_eq!(
            run_cleanup(env, sibling_id).await,
            JobState::Cancelled,
            "{what}"
        );
        assert_states(env, &[(sibling_id, JobState::Cancelled)]).await;
    }
    assert_quiescent(env, tree.root_id).await;
    Ok(())
}

/// How T4 lands the last reports of the siblings.
#[derive(Clone, Copy, Debug)]
enum SiblingRounds {
    /// Every report in the round of the first one.
    One,

    /// Each report in its own round, one after the other.
    Each,

    /// Every report at once, in whatever rounds the committer forms.
    Free,
}

/// Where T4's parent is.
#[derive(Clone, Copy, Debug)]
enum ParentPosition {
    /// The parent is the root.
    Root,

    /// The parent is the only child of a one-task root.
    UnderRoot,
}

/// The jobs of a T4 tree: a parent, with or without a commit task, blocking on one-task children.
struct SiblingTree {
    root_id: JobId,
    parent_id: JobId,
    parent_value: Vec<u8>,
    child_ids: Vec<JobId>,
    child_values: Vec<Vec<u8>>,
    position: ParentPosition,
    parent_has_commit: bool,
}

impl SiblingTree {
    /// Starts a tree whose parent is at `position` and has `num_children` children.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(
        env: &TestEnv,
        num_children: usize,
        position: ParentPosition,
        parent_has_commit: bool,
    ) -> Self {
        let parent_shape = JobShape {
            with_commit: parent_has_commit,
            ..ONE_TASK
        };
        let (root_id, parent_id) = match position {
            ParentPosition::Root => {
                let parent_id = start_root(env, parent_shape).await;
                (parent_id, parent_id)
            }
            ParentPosition::UnderRoot => {
                let root_id = start_root(env, ONE_TASK).await;
                let parent_id =
                    register_children(env, root_id, &value(0xb0, 0), &[parent_shape]).await[0];
                (root_id, parent_id)
            }
        };
        let parent_value = value(0xb1, 0);
        let child_ids =
            register_children(env, parent_id, &parent_value, &vec![ONE_TASK; num_children]).await;
        Self {
            root_id,
            parent_id,
            parent_value,
            child_ids,
            child_values: (0..num_children).map(|index| value(0xc0, index)).collect(),
            position,
            parent_has_commit,
        }
    }

    /// # Returns
    ///
    /// The state the parent advances to when its last child succeeds.
    const fn advanced_parent_state(&self) -> JobState {
        if self.parent_has_commit {
            JobState::CommitReady
        } else {
            JobState::Succeeded
        }
    }
}

/// T4: sibling jobs complete at the same time: their parent advances exactly once, its counter
/// reaches 0 once, and when it advances every sibling's outputs are in its row and in the cache.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn sibling_jobs_completing_together_advance_their_parent_once() -> anyhow::Result<()> {
    let env = start_env().await;
    for iteration in 0..race_iterations() {
        for num_children in [2, 8, 64] {
            // 64 children cost about 30 times as much as 2; they run on every tenth iteration.
            if num_children == 64 && iteration % 10 != 0 {
                continue;
            }
            for rounds in [SiblingRounds::One, SiblingRounds::Each, SiblingRounds::Free] {
                for position in [ParentPosition::Root, ParentPosition::UnderRoot] {
                    for parent_has_commit in [false, true] {
                        let tree =
                            SiblingTree::start(&env, num_children, position, parent_has_commit)
                                .await;
                        assert_siblings_advance_their_parent_once(&env, &tree, rounds).await?;
                    }
                }
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Reports the successes of `tree`'s children as `rounds` says.
///
/// # Errors
///
/// Returns an error if any report fails.
async fn assert_siblings_advance_their_parent_once(
    env: &TestEnv,
    tree: &SiblingTree,
    rounds: SiblingRounds,
) -> anyhow::Result<()> {
    let num_children = tree.child_ids.len();
    let what = format!(
        "{num_children} children in {rounds:?} rounds, parent at {:?}, commit: {}",
        tree.position, tree.parent_has_commit
    );
    let mut reports = Vec::with_capacity(num_children);
    for (child_id, child_value) in tree.child_ids.iter().zip(&tree.child_values) {
        let task_instance_id = instance(env, *child_id, TaskId::Index(0)).await;
        reports.push(succeed(
            env,
            *child_id,
            0,
            task_instance_id,
            child_value,
            Vec::new(),
        ));
    }
    let probes = TreeProbes::arm(env, tree.root_id, &[tree.root_id, tree.parent_id]);
    let advance = pause(
        env,
        HookPoint::CommitterAfterApplyJob,
        job_filter(tree.root_id, tree.parent_id),
    );
    let (results, ()) = tokio::join!(
        report_sibling_successes(env, tree, reports, rounds, &what),
        assert_children_visible_at_advance(env, tree, &advance, &what)
    );
    for result in results {
        assert_eq!(result?, JobState::Succeeded, "{what}");
    }
    assert_eq!(
        probes.applies(tree.parent_id),
        1,
        "{what}: the parent advances once"
    );
    match rounds {
        SiblingRounds::One => {
            probes.assert_one_round(&vec![EventKindTag::RegularTasksDone; num_children], &what);
            assert_eq!(probes.commits.hits(), 1, "{what}: database transactions");
        }
        SiblingRounds::Each => {
            assert_eq!(
                probes.commits.hits(),
                num_children,
                "{what}: database transactions"
            );
        }
        SiblingRounds::Free => {
            assert!(
                (1..=num_children).contains(&probes.commits.hits()),
                "{what}: {} database transactions",
                probes.commits.hits()
            );
        }
    }
    assert_after_siblings(env, tree, &probes, &what).await;
    if tree.parent_has_commit {
        commit_sibling_parent(env, tree, &probes, &what).await?;
    }
    assert_quiescent(env, tree.root_id).await;
    Ok(())
}

/// Submits `reports`, the successes of `tree`'s children in child order, as `rounds` says. Between
/// two reports of [`SiblingRounds::Each`], asserts that the parent still blocks on its children.
///
/// # Type Parameters
///
/// * `ReportType` - The type of the futures that report the successes.
///
/// # Returns
///
/// The results of the reports, in child order.
async fn report_sibling_successes<ReportType: Future<Output = JobResult> + Send + 'static>(
    env: &TestEnv,
    tree: &SiblingTree,
    mut reports: Vec<ReportType>,
    rounds: SiblingRounds,
    what: &str,
) -> Vec<JobResult> {
    let num_reports = reports.len();
    let mut handles = Vec::with_capacity(num_reports);
    match rounds {
        SiblingRounds::One => {
            let paused = pause(
                env,
                HookPoint::CommitterAfterReceive,
                HookFilter::root(tree.root_id),
            );
            handles.push(tokio::spawn(reports.remove(0)));
            reached(&paused, 1).await;
            let submitted = observe(
                env,
                HookPoint::EventSubmitted,
                HookFilter {
                    root_id: Some(tree.root_id),
                    job_id: None,
                    event_kind: Some(EventKindTag::RegularTasksDone),
                },
            );
            handles.extend(reports.into_iter().map(tokio::spawn));
            reached(&submitted, num_reports - 1).await;
            paused.release();
        }
        SiblingRounds::Each => {
            let mut results = Vec::with_capacity(num_reports);
            for (index, report) in reports.into_iter().enumerate() {
                results.push(
                    tokio::spawn(report)
                        .await
                        .expect("a report should not panic"),
                );
                if index + 1 < num_reports {
                    assert_eq!(
                        env.cached_job_state(tree.parent_id).await,
                        Some(JobState::BlockingOnChildren),
                        "{what}: the cached parent after {} reports",
                        index + 1
                    );
                    assert_eq!(
                        env.row_state(tree.parent_id).await,
                        JobState::BlockingOnChildren,
                        "{what}: the parent's row after {} reports",
                        index + 1
                    );
                }
            }
            return results;
        }
        SiblingRounds::Free => handles.extend(reports.into_iter().map(tokio::spawn)),
    }
    let mut results = Vec::with_capacity(num_reports);
    for handle in handles {
        results.push(handle.await.expect("a report should not panic"));
    }
    results
}

/// Waits until the committer, paused by `advance`, has applied the advance of `tree`'s parent to
/// the cache, asserts that every child is `Succeeded` with its outputs in its row and in the
/// cache, and that the parent advanced in its row and in the cache, then releases the committer.
async fn assert_children_visible_at_advance(
    env: &TestEnv,
    tree: &SiblingTree,
    advance: &ArmedHook,
    what: &str,
) {
    reached(advance, 1).await;
    for (child_id, child_value) in tree.child_ids.iter().zip(&tree.child_values) {
        let expected = task_outputs([child_value]);
        assert_eq!(
            env.row_state(*child_id).await,
            JobState::Succeeded,
            "{what}: child {child_id} when the parent advances"
        );
        assert_eq!(
            row_outputs(env, *child_id).await,
            Some(expected.clone()),
            "{what}: stored outputs of child {child_id} when the parent advances"
        );
        assert_eq!(
            env.cached_job_state(*child_id).await,
            Some(JobState::Succeeded),
            "{what}: cached child {child_id} when the parent advances"
        );
        assert_eq!(
            cached_outputs(env, *child_id).await,
            expected,
            "{what}: cached outputs of child {child_id} when the parent advances"
        );
    }
    let parent_state = tree.advanced_parent_state();
    assert_eq!(env.row_state(tree.parent_id).await, parent_state, "{what}");
    assert_eq!(
        env.cached_job_state(tree.parent_id).await,
        Some(parent_state),
        "{what}"
    );
    advance.release();
}

/// Asserts the states of `tree` after its children succeeded, the root's applies, and that the
/// parent's own outputs are unchanged.
async fn assert_after_siblings(env: &TestEnv, tree: &SiblingTree, probes: &TreeProbes, what: &str) {
    let mut expected = vec![(tree.parent_id, tree.advanced_parent_state())];
    expected.extend(tree.child_ids.iter().map(|id| (*id, JobState::Succeeded)));
    if matches!(tree.position, ParentPosition::UnderRoot) {
        let (root_state, root_applies) = if tree.parent_has_commit {
            (JobState::BlockingOnChildren, 0)
        } else {
            (JobState::Succeeded, 1)
        };
        expected.push((tree.root_id, root_state));
        assert_eq!(
            probes.applies(tree.root_id),
            root_applies,
            "{what}: root applies"
        );
    }
    assert_states(env, &expected).await;
    assert_eq!(
        row_outputs(env, tree.parent_id).await,
        Some(task_outputs([&tree.parent_value])),
        "{what}: the parent's own outputs"
    );
}

/// Runs the commit task of `tree`'s commit-ready parent, which must be handed only the parent's
/// outputs, and asserts that the parent and its root succeed.
///
/// # Errors
///
/// Returns an error if the commit report fails.
async fn commit_sibling_parent(
    env: &TestEnv,
    tree: &SiblingTree,
    probes: &TreeProbes,
    what: &str,
) -> anyhow::Result<()> {
    let (task_instance_id, commit_inputs) = commit_instance(env, tree.parent_id).await;
    assert_eq!(
        commit_inputs,
        task_outputs([&tree.parent_value]),
        "{what}: the commit task's inputs"
    );
    assert_eq!(
        succeed_commit(env, tree.parent_id, task_instance_id).await?,
        JobState::Succeeded,
        "{what}"
    );
    let mut expected = vec![(tree.parent_id, JobState::Succeeded)];
    if matches!(tree.position, ParentPosition::UnderRoot) {
        expected.push((tree.root_id, JobState::Succeeded));
        assert_eq!(probes.applies(tree.root_id), 1, "{what}: root applies");
    }
    assert_states(env, &expected).await;
    assert_eq!(probes.applies(tree.parent_id), 2, "{what}: parent applies");
    Ok(())
}

/// T4 within one job: the last tasks of a job pass the task-level success together, and race on
/// the job's count of incomplete tasks. Exactly one of them submits the job's completion, and the
/// job's outputs hold every task's output.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn sibling_tasks_completing_together_complete_their_job_once() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for num_tasks in [2, 8] {
            for position in [ParentPosition::Root, ParentPosition::UnderRoot] {
                assert_sibling_tasks_complete_their_job_once(&env, num_tasks, position).await?;
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Holds every task of a job of `num_tasks` tasks at `position` between its task-level success and
/// the job's count decrement, then lets them all decrement at once.
///
/// # Errors
///
/// Returns an error if any report fails.
async fn assert_sibling_tasks_complete_their_job_once(
    env: &TestEnv,
    num_tasks: usize,
    position: ParentPosition,
) -> anyhow::Result<()> {
    let what = format!("{num_tasks} tasks, job at {position:?}");
    let shape = JobShape {
        num_tasks,
        ..ONE_TASK
    };
    let (root_id, job_id) = match position {
        ParentPosition::Root => {
            let job_id = start_root(env, shape).await;
            (job_id, job_id)
        }
        ParentPosition::UnderRoot => {
            let root_id = start_root(env, ONE_TASK).await;
            let job_id = register_children(env, root_id, &value(0xd0, 0), &[shape]).await[0];
            (root_id, job_id)
        }
    };
    let task_values: Vec<_> = (0..num_tasks).map(|index| value(0xd1, index)).collect();
    let mut reports = Vec::with_capacity(num_tasks);
    for (task_index, task_value) in task_values.iter().enumerate() {
        let task_instance_id = instance(env, job_id, TaskId::Index(task_index)).await;
        reports.push(succeed(
            env,
            job_id,
            task_index,
            task_instance_id,
            task_value,
            Vec::new(),
        ));
    }
    let flips = env.hooks.arm(
        HookPoint::PhaseAAfterTcbFlip,
        job_filter(root_id, job_id),
        HookAction::Pause,
        HookCount::Times(num_tasks),
    );
    let done_events = observe(
        env,
        HookPoint::EventSubmitted,
        event_filter(root_id, job_id, EventKindTag::RegularTasksDone),
    );
    let probes = TreeProbes::arm(env, root_id, &[root_id]);

    let handles: Vec<_> = reports.into_iter().map(tokio::spawn).collect();
    reached(&flips, num_tasks).await;
    assert_eq!(
        done_events.hits(),
        0,
        "{what}: no completion before the decrements"
    );
    assert_eq!(
        env.cached_job_state(job_id).await,
        Some(JobState::Running),
        "{what}"
    );
    for _ in 0..num_tasks {
        flips.release();
    }
    let mut results = Vec::with_capacity(num_tasks);
    for handle in handles {
        results.push(handle.await.expect("a report should not panic")?);
    }
    let num_succeeded = results
        .iter()
        .filter(|state| **state == JobState::Succeeded)
        .count();
    let num_running = results
        .iter()
        .filter(|state| **state == JobState::Running)
        .count();
    assert_eq!(
        (num_running, num_succeeded),
        (num_tasks - 1, 1),
        "{what}: report results {results:?}"
    );
    assert_eq!(done_events.hits(), 1, "{what}: completion events");

    let expected_outputs = task_outputs(&task_values);
    assert_eq!(
        row_outputs(env, job_id).await,
        Some(expected_outputs.clone()),
        "{what}"
    );
    assert_eq!(
        cached_outputs(env, job_id).await,
        expected_outputs,
        "{what}"
    );
    assert_states(
        env,
        &[
            (job_id, JobState::Succeeded),
            (root_id, JobState::Succeeded),
        ],
    )
    .await;
    assert_eq!(probes.applies(root_id), 1, "{what}: root applies");
    assert_quiescent(env, root_id).await;
    Ok(())
}

/// The job of T5's straggler.
#[derive(Clone, Copy, Debug)]
enum StragglerJob {
    /// A one-task root job, which the success completes.
    Root,

    /// A one-task leaf with a one-task sibling: the success completes the leaf but not its tree.
    Leaf,

    /// A two-task root job, which the success of task 0 does not complete.
    TwoTaskRoot,
}

/// The instance that fails in T5.
#[derive(Clone, Copy, Debug)]
enum Straggler {
    /// The instance that succeeded.
    SameInstance,

    /// A second instance of the task.
    OtherInstance,
}

/// When T5's straggler fails, relative to the success.
#[derive(Clone, Copy, Debug)]
enum StragglerWindow {
    /// After the success's task-level flip, before it decrements the job's count.
    AfterTcbFlip,

    /// While the success's round is paused before its database transaction.
    BeforeDbTransaction,

    /// While the success's round has committed but is not applied to the cache yet.
    AfterCommit,

    /// After the success was answered.
    AfterApply,
}

/// A T5 job with the instance that succeeds task 0 and the instance that fails it.
struct StragglerCase {
    job: StragglerJob,
    root_id: JobId,
    job_id: JobId,
    sibling_id: Option<JobId>,
    winner: TaskInstanceId,
    straggler: TaskInstanceId,
}

impl StragglerCase {
    /// Starts a job of `job` with tasks that may run two instances and retry `max_num_retry` times,
    /// and registers the instances.
    ///
    /// # Returns
    ///
    /// The case.
    async fn start(
        env: &TestEnv,
        job: StragglerJob,
        straggler: Straggler,
        max_num_retry: u32,
    ) -> Self {
        let shape = JobShape {
            num_tasks: if matches!(job, StragglerJob::TwoTaskRoot) {
                2
            } else {
                1
            },
            max_num_instances: 2,
            max_num_retry,
            ..ONE_TASK
        };
        let (root_id, job_id, sibling_id) = match job {
            StragglerJob::Root | StragglerJob::TwoTaskRoot => {
                let job_id = start_root(env, shape).await;
                (job_id, job_id, None)
            }
            StragglerJob::Leaf => {
                let root_id = start_root(env, ONE_TASK).await;
                let child_ids =
                    register_children(env, root_id, &value(0xe0, 0), &[shape, ONE_TASK]).await;
                (root_id, child_ids[0], Some(child_ids[1]))
            }
        };
        let winner = instance(env, job_id, TaskId::Index(0)).await;
        let straggler = match straggler {
            Straggler::SameInstance => winner,
            Straggler::OtherInstance => instance(env, job_id, TaskId::Index(0)).await,
        };
        Self {
            job,
            root_id,
            job_id,
            sibling_id,
            winner,
            straggler,
        }
    }

    /// # Returns
    ///
    /// The state of the job after task 0 succeeded.
    const fn state_after_success(&self) -> JobState {
        if matches!(self.job, StragglerJob::TwoTaskRoot) {
            JobState::Running
        } else {
            JobState::Succeeded
        }
    }
}

/// T5: a straggler fails a task after another report succeeded it, before and after the success is
/// applied, from the same and from another instance.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_straggler_failure_after_a_success_is_stale() -> anyhow::Result<()> {
    let env = start_env().await;
    for iteration in 0..race_iterations() {
        let max_num_retry = u32::try_from(iteration % 2)?;
        for job in [
            StragglerJob::Root,
            StragglerJob::Leaf,
            StragglerJob::TwoTaskRoot,
        ] {
            for straggler in [Straggler::SameInstance, Straggler::OtherInstance] {
                for window in [
                    StragglerWindow::AfterTcbFlip,
                    StragglerWindow::BeforeDbTransaction,
                    StragglerWindow::AfterCommit,
                    StragglerWindow::AfterApply,
                ] {
                    // Task 0's success alone does not complete a two-task job, so no round exists
                    // to pause.
                    if matches!(job, StragglerJob::TwoTaskRoot)
                        && matches!(
                            window,
                            StragglerWindow::BeforeDbTransaction | StragglerWindow::AfterCommit
                        )
                    {
                        continue;
                    }
                    let case = StragglerCase::start(&env, job, straggler, max_num_retry).await;
                    let what =
                        format!("{job:?}, {straggler:?}, {window:?}, retries: {max_num_retry}");
                    assert_straggler_failure_is_stale(&env, &case, straggler, window, &what)
                        .await?;
                }
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Succeeds task 0 of `case`'s job, fails it from `straggler` in `window`, and asserts that the
/// failure is stale and changes nothing.
///
/// # Errors
///
/// Returns an error if the success or a later report fails.
async fn assert_straggler_failure_is_stale(
    env: &TestEnv,
    case: &StragglerCase,
    straggler: Straggler,
    window: StragglerWindow,
    what: &str,
) -> anyhow::Result<()> {
    let failure_events = observe(
        env,
        HookPoint::EventSubmitted,
        HookFilter {
            root_id: Some(case.root_id),
            job_id: None,
            event_kind: Some(EventKindTag::TerminalFailure),
        },
    );
    let winner_value = value(0xe1, 0);
    let (success, failure) = succeed_then_fail(env, case, window, &winner_value, what).await;

    assert_eq!(success?, case.state_after_success(), "{what}: the success");
    let expected_stale = match (window, case.job, straggler) {
        (StragglerWindow::AfterApply, StragglerJob::Root | StragglerJob::Leaf, _) => {
            StaleStateError::JobNoLongerRunning
        }
        (_, _, Straggler::SameInstance) => StaleStateError::InvalidTaskInstanceId,
        (_, _, Straggler::OtherInstance) => {
            StaleStateError::TaskAlreadyTerminated(TaskState::Succeeded)
        }
    };
    assert_stale(&failure, &expected_stale, &format!("{what}: the straggler"));
    assert_eq!(failure_events.hits(), 0, "{what}: failure events");
    assert_eq!(tree_errors(env, case.root_id).await, Vec::new(), "{what}");
    let mut expected = vec![(case.job_id, case.state_after_success())];
    if let Some(sibling_id) = case.sibling_id {
        expected.push((case.root_id, JobState::BlockingOnChildren));
        expected.push((sibling_id, JobState::Running));
    }
    assert_states(env, &expected).await;
    finish_straggler_job(env, case, &winner_value, what).await;
    Ok(())
}

/// Reports the success of `case`'s winner with `winner_value`, then the failure of its straggler
/// in `window`.
///
/// # Returns
///
/// The results of the success and of the failure.
///
/// # Panics
///
/// Panics if the straggler is not answered within 30 seconds while the success is paused.
async fn succeed_then_fail(
    env: &TestEnv,
    case: &StragglerCase,
    window: StragglerWindow,
    winner_value: &[u8],
    what: &str,
) -> (JobResult, JobResult) {
    let success = succeed(env, case.job_id, 0, case.winner, winner_value, Vec::new());
    let failure = fail(env, case.job_id, 0, case.straggler, "straggler");
    let (point, filter) = match window {
        StragglerWindow::AfterTcbFlip => (
            HookPoint::PhaseAAfterTcbFlip,
            job_filter(case.root_id, case.job_id),
        ),
        StragglerWindow::BeforeDbTransaction => (
            HookPoint::CommitterBeforeDbTransaction,
            HookFilter::root(case.root_id),
        ),
        StragglerWindow::AfterCommit => (
            HookPoint::CommitterAfterCommit,
            HookFilter::root(case.root_id),
        ),
        StragglerWindow::AfterApply => {
            let success = success.await;
            return (success, failure.await);
        }
    };
    let paused = pause(env, point, filter);
    let success = tokio::spawn(success);
    reached(&paused, 1).await;
    if matches!(window, StragglerWindow::AfterCommit) {
        assert_eq!(
            env.row_state(case.job_id).await,
            JobState::Succeeded,
            "{what}: the row in the commit window"
        );
        assert_eq!(
            env.cached_job_state(case.job_id).await,
            Some(JobState::Running),
            "{what}: the cache in the commit window"
        );
    }
    let failure = tokio::time::timeout(Duration::from_secs(30), failure)
        .await
        .expect("the straggler should be answered while the success is paused");
    paused.release();
    (
        success.await.expect("the success should not panic"),
        failure,
    )
}

/// Runs the rest of `case`'s tree to success and asserts the job's outputs: `winner_value` from
/// task 0, followed by task 1's output for a two-task job.
async fn finish_straggler_job(
    env: &TestEnv,
    case: &StragglerCase,
    winner_value: &[u8],
    what: &str,
) {
    let mut expected_outputs = task_outputs([&winner_value.to_vec()]);
    match case.job {
        StragglerJob::Root => {}
        StragglerJob::Leaf => {
            let sibling_id = case.sibling_id.expect("a leaf has a sibling");
            assert_eq!(
                run_task(env, sibling_id, 0, &value(0xe2, 0)).await,
                JobState::Succeeded,
                "{what}"
            );
            assert_states(
                env,
                &[
                    (case.root_id, JobState::Succeeded),
                    (sibling_id, JobState::Succeeded),
                ],
            )
            .await;
        }
        StragglerJob::TwoTaskRoot => {
            let last_value = value(0xe1, 1);
            assert_eq!(
                run_task(env, case.job_id, 1, &last_value).await,
                JobState::Succeeded,
                "{what}"
            );
            expected_outputs.extend(task_outputs([&last_value]));
            assert_states(env, &[(case.job_id, JobState::Succeeded)]).await;
        }
    }
    assert_eq!(
        row_outputs(env, case.job_id).await,
        Some(expected_outputs.clone()),
        "{what}"
    );
    assert_eq!(
        cached_outputs(env, case.job_id).await,
        expected_outputs,
        "{what}"
    );
    assert_quiescent(env, case.root_id).await;
}

/// When T6's sibling `Q` of the commit parent `P` completes.
#[derive(Clone, Copy, Debug)]
enum SiblingTiming {
    /// Before `P`'s children complete.
    BeforeChildren,

    /// In the round of `P`'s last child.
    WithLastChild,

    /// After `P` is commit-ready, before its commit task reports.
    BeforeCommit,

    /// In the round of `P`'s commit success.
    WithCommit,

    /// Queued while `P`'s commit success's round is in flight.
    AfterCommit,
}

/// The jobs of a T6 tree: a root `R` whose children are `P`, with a commit task and blocking on
/// its own children, and `Q`, a one-task job.
struct CommitTree {
    root_id: JobId,
    root_value: Vec<u8>,
    parent_id: JobId,
    parent_value: Vec<u8>,
    sibling_id: JobId,
    sibling_instance: TaskInstanceId,
    child_ids: Vec<JobId>,
    child_values: Vec<Vec<u8>>,
    child_instances: Vec<TaskInstanceId>,
    root_has_commit: bool,
}

impl CommitTree {
    const NUM_CHILDREN: usize = 3;

    /// Starts the tree, with a commit task on the root if `root_has_commit`, and registers an
    /// instance of `Q`'s task and of each of `P`'s children's tasks.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, root_has_commit: bool) -> Self {
        let with_commit = JobShape {
            with_commit: true,
            ..ONE_TASK
        };
        let root_value = value(0xf0, 0);
        let parent_value = value(0xf1, 0);
        let root_id = start_root(
            env,
            JobShape {
                with_commit: root_has_commit,
                ..ONE_TASK
            },
        )
        .await;
        let root_children =
            register_children(env, root_id, &root_value, &[with_commit, ONE_TASK]).await;
        let (parent_id, sibling_id) = (root_children[0], root_children[1]);
        let child_ids = register_children(
            env,
            parent_id,
            &parent_value,
            &[ONE_TASK; Self::NUM_CHILDREN],
        )
        .await;
        let mut child_instances = Vec::with_capacity(Self::NUM_CHILDREN);
        for child_id in &child_ids {
            child_instances.push(instance(env, *child_id, TaskId::Index(0)).await);
        }
        Self {
            root_id,
            root_value,
            parent_id,
            parent_value,
            sibling_id,
            sibling_instance: instance(env, sibling_id, TaskId::Index(0)).await,
            child_ids,
            child_values: (0..Self::NUM_CHILDREN)
                .map(|index| value(0xf3, index))
                .collect(),
            child_instances,
            root_has_commit,
        }
    }

    /// # Returns
    ///
    /// The output value of `Q`'s task.
    fn sibling_value() -> Vec<u8> {
        value(0xf2, 0)
    }

    /// # Returns
    ///
    /// A future that reports the success of `Q`'s task.
    fn sibling_report(&self, env: &TestEnv) -> impl Future<Output = JobResult> + Send + use<> {
        succeed(
            env,
            self.sibling_id,
            0,
            self.sibling_instance,
            &Self::sibling_value(),
            Vec::new(),
        )
    }

    /// # Returns
    ///
    /// A future that reports the success of the task of `P`'s child `index`.
    fn child_report(
        &self,
        env: &TestEnv,
        index: usize,
    ) -> impl Future<Output = JobResult> + Send + use<> {
        succeed(
            env,
            self.child_ids[index],
            0,
            self.child_instances[index],
            &self.child_values[index],
            Vec::new(),
        )
    }
}

/// T6: a parent `P` with a commit task, under a root `R` with a second child `Q`: `P`'s children
/// complete, `P` becomes commit-ready with its commit task published once and handed only `P`'s
/// outputs, the commit succeeds, and the success climbs to `R`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_parent_with_a_commit_task_commits_then_climbs() -> anyhow::Result<()> {
    let env = start_env().await;
    for _ in 0..race_iterations() {
        for sibling_timing in [
            SiblingTiming::BeforeChildren,
            SiblingTiming::WithLastChild,
            SiblingTiming::BeforeCommit,
            SiblingTiming::WithCommit,
            SiblingTiming::AfterCommit,
        ] {
            for root_has_commit in [false, true] {
                let what = format!("Q {sibling_timing:?}, root commit: {root_has_commit}");
                let tree = CommitTree::start(&env, root_has_commit).await;
                let probes = TreeProbes::arm(
                    &env,
                    tree.root_id,
                    &[tree.root_id, tree.parent_id, tree.sibling_id],
                );
                complete_commit_parent_children(&env, &tree, &probes, sibling_timing, &what)
                    .await?;
                commit_parent(&env, &tree, sibling_timing, &what).await?;
                assert_commit_tree_climbed(&env, &tree, &probes, &what).await?;
            }
        }
    }
    assert_lanes_empty(&env).await;
    env.finish().await;
    Ok(())
}

/// Completes `P`'s children, with `Q` before them or in the round of the last one as
/// `sibling_timing` says, and asserts that `P` is commit-ready with its commit task published
/// once, its children's and its own outputs stored, while `R` still blocks.
///
/// # Errors
///
/// Returns an error if any report fails.
async fn complete_commit_parent_children(
    env: &TestEnv,
    tree: &CommitTree,
    probes: &TreeProbes,
    sibling_timing: SiblingTiming,
    what: &str,
) -> anyhow::Result<()> {
    if matches!(sibling_timing, SiblingTiming::BeforeChildren) {
        assert_eq!(
            tree.sibling_report(env).await?,
            JobState::Succeeded,
            "{what}"
        );
        assert_states(
            env,
            &[
                (tree.sibling_id, JobState::Succeeded),
                (tree.root_id, JobState::BlockingOnChildren),
            ],
        )
        .await;
    }
    let last_index = CommitTree::NUM_CHILDREN - 1;
    for index in 0..last_index {
        assert_eq!(
            tree.child_report(env, index).await?,
            JobState::Succeeded,
            "{what}"
        );
        assert_states(env, &[(tree.parent_id, JobState::BlockingOnChildren)]).await;
    }
    if matches!(sibling_timing, SiblingTiming::WithLastChild) {
        let (last_child, sibling) = race(
            env,
            tree.root_id,
            HookPoint::CommitterAfterReceive,
            tree.child_report(env, last_index),
            tree.sibling_report(env),
            event_filter(
                tree.root_id,
                tree.sibling_id,
                EventKindTag::RegularTasksDone,
            ),
        )
        .await;
        assert_eq!(last_child?, JobState::Succeeded, "{what}");
        assert_eq!(sibling?, JobState::Succeeded, "{what}");
        assert_eq!(
            probes
                .rounds
                .last_context()
                .map(|context| context.event_kinds),
            Some(vec![
                EventKindTag::RegularTasksDone,
                EventKindTag::RegularTasksDone
            ]),
            "{what}: the last child's round"
        );
    } else {
        assert_eq!(
            tree.child_report(env, last_index).await?,
            JobState::Succeeded,
            "{what}"
        );
    }

    let mut expected: Vec<_> = tree
        .child_ids
        .iter()
        .map(|id| (*id, JobState::Succeeded))
        .collect();
    expected.push((tree.parent_id, JobState::CommitReady));
    expected.push((tree.root_id, JobState::BlockingOnChildren));
    assert_states(env, &expected).await;
    for (child_id, child_value) in tree.child_ids.iter().zip(&tree.child_values) {
        assert_eq!(
            row_outputs(env, *child_id).await,
            Some(task_outputs([child_value])),
            "{what}: outputs of child {child_id}"
        );
    }
    assert_eq!(
        row_outputs(env, tree.parent_id).await,
        Some(task_outputs([&tree.parent_value])),
        "{what}: the parent's outputs"
    );
    assert_eq!(probes.applies(tree.parent_id), 1, "{what}: parent applies");
    assert_eq!(probes.applies(tree.root_id), 0, "{what}: root applies");
    Ok(())
}

/// Runs `P`'s commit task, which must be handed only `P`'s outputs, with `Q` before it, in its
/// round, or queued behind its round as `sibling_timing` says, unless `Q` already completed.
///
/// # Errors
///
/// Returns an error if any report fails.
async fn commit_parent(
    env: &TestEnv,
    tree: &CommitTree,
    sibling_timing: SiblingTiming,
    what: &str,
) -> anyhow::Result<()> {
    let (task_instance_id, commit_inputs) = commit_instance(env, tree.parent_id).await;
    assert_eq!(
        commit_inputs,
        task_outputs([&tree.parent_value]),
        "{what}: the parent's commit task sees only the parent's outputs"
    );
    if matches!(sibling_timing, SiblingTiming::BeforeCommit) {
        assert_eq!(
            tree.sibling_report(env).await?,
            JobState::Succeeded,
            "{what}"
        );
        assert_states(
            env,
            &[
                (tree.sibling_id, JobState::Succeeded),
                (tree.parent_id, JobState::CommitReady),
                (tree.root_id, JobState::BlockingOnChildren),
            ],
        )
        .await;
    }
    let commit = succeed_commit(env, tree.parent_id, task_instance_id);
    let point = match sibling_timing {
        SiblingTiming::WithCommit => HookPoint::CommitterAfterReceive,
        SiblingTiming::AfterCommit => HookPoint::CommitterBeforeDbTransaction,
        SiblingTiming::BeforeChildren
        | SiblingTiming::WithLastChild
        | SiblingTiming::BeforeCommit => {
            assert_eq!(commit.await?, JobState::Succeeded, "{what}");
            return Ok(());
        }
    };
    let rounds = observe(
        env,
        HookPoint::CommitterBeforePlan,
        HookFilter::root(tree.root_id),
    );
    let (committed, sibling) = race(
        env,
        tree.root_id,
        point,
        commit,
        tree.sibling_report(env),
        event_filter(
            tree.root_id,
            tree.sibling_id,
            EventKindTag::RegularTasksDone,
        ),
    )
    .await;
    assert_eq!(committed?, JobState::Succeeded, "{what}");
    assert_eq!(sibling?, JobState::Succeeded, "{what}");
    if point == HookPoint::CommitterAfterReceive {
        assert_eq!(rounds.hits(), 1, "{what}: number of rounds");
        assert_eq!(
            rounds.last_context().map(|context| context.event_kinds),
            Some(vec![
                EventKindTag::CommitDone,
                EventKindTag::RegularTasksDone
            ]),
            "{what}: the commit's round"
        );
    } else {
        assert_eq!(rounds.hits(), 2, "{what}: number of rounds");
    }
    Ok(())
}

/// Asserts that `P` and `Q` succeeded and that the success climbed to `R` once, then runs `R`'s
/// commit task if it has one, which must be handed only `R`'s outputs.
///
/// # Errors
///
/// Returns an error if the root's commit report or its outputs read fails.
async fn assert_commit_tree_climbed(
    env: &TestEnv,
    tree: &CommitTree,
    probes: &TreeProbes,
    what: &str,
) -> anyhow::Result<()> {
    let root_state = if tree.root_has_commit {
        JobState::CommitReady
    } else {
        JobState::Succeeded
    };
    assert_states(
        env,
        &[
            (tree.parent_id, JobState::Succeeded),
            (tree.sibling_id, JobState::Succeeded),
            (tree.root_id, root_state),
        ],
    )
    .await;
    assert_eq!(probes.applies(tree.parent_id), 2, "{what}: parent applies");
    assert_eq!(
        probes.applies(tree.sibling_id),
        1,
        "{what}: sibling applies"
    );
    assert_eq!(probes.applies(tree.root_id), 1, "{what}: root applies");
    let parent_outputs = task_outputs([&tree.parent_value]);
    assert_eq!(
        cached_outputs(env, tree.parent_id).await,
        parent_outputs,
        "{what}"
    );
    assert_eq!(
        row_outputs(env, tree.parent_id).await,
        Some(parent_outputs),
        "{what}"
    );
    assert_eq!(
        row_outputs(env, tree.sibling_id).await,
        Some(task_outputs([&CommitTree::sibling_value()])),
        "{what}"
    );
    if tree.root_has_commit {
        let (task_instance_id, commit_inputs) = commit_instance(env, tree.root_id).await;
        assert_eq!(
            commit_inputs,
            task_outputs([&tree.root_value]),
            "{what}: the root's commit task sees only the root's outputs"
        );
        assert_eq!(
            succeed_commit(env, tree.root_id, task_instance_id).await?,
            JobState::Succeeded,
            "{what}"
        );
        assert_states(env, &[(tree.root_id, JobState::Succeeded)]).await;
        assert_eq!(probes.applies(tree.root_id), 2, "{what}: root applies");
    }
    assert_eq!(
        env.service.get_job_outputs(tree.root_id).await?,
        task_outputs([&tree.root_value]),
        "{what}"
    );
    assert_eq!(tree_errors(env, tree.root_id).await, Vec::new(), "{what}");
    assert_quiescent(env, tree.root_id).await;
    Ok(())
}
