//! Faults around job trees, made deterministic through the test hooks: storage restarts at the
//! committer's crash points (T8), RPC futures dropped while their work is in flight (T9), and task
//! reports over the limits (T10).

use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::net::IpAddr;
use std::pin::Pin;
use std::time::Duration;

use spider_core::job::JobState;
use spider_core::task::DataTypeDescriptor;
use spider_core::task::ExecutionPolicy;
use spider_core::task::TaskDescriptor;
use spider_core::task::TaskIndex;
use spider_core::task::TaskInputOutputIndex;
use spider_core::task::TdlContext;
use spider_core::task::TerminationTaskDescriptor;
use spider_core::task::ValueTypeDescriptor;
use spider_core::types::id::ExecutionManagerId;
use spider_core::types::id::JobId;
use spider_core::types::id::SessionId;
use spider_core::types::id::TaskId;
use spider_core::types::id::TaskInstanceId;
use spider_core::types::io::SerializedTaskOutputs;
use spider_core::types::io::TaskGraphInput;
use spider_core::types::io::TaskGraphInputBuilder;
use spider_core::types::io::TaskOutput;
use spider_core::types::io::TaskOutputsSerializer;
use spider_storage::cache::error::CacheError;
use spider_storage::cache::error::InternalError;
use spider_storage::cache::error::StaleStateError;
use spider_storage::db::InternalJobOrchestration;
use spider_storage::job_submission::ChildJobSubmission;
use spider_storage::limits::MAX_CHILDREN_PER_JOB;
use spider_storage::limits::MAX_DEPTH;
use spider_storage::limits::MAX_JOB_CHILD_BYTES;
use spider_storage::limits::MAX_REPORT_CHILD_BYTES;
use spider_storage::limits::MAX_REPORT_OUTPUT_BYTES;
use spider_storage::state::StorageServerError;
use spider_storage::task_instance_pool::TaskInstancePoolConfig;
use spider_storage::test_hooks::ArmedHook;
use spider_storage::test_hooks::EventKindTag;
use spider_storage::test_hooks::HookAction;
use spider_storage::test_hooks::HookCount;
use spider_storage::test_hooks::HookFilter;
use spider_storage::test_hooks::HookPoint;
use spider_storage::test_hooks::TestHooks;
use sqlx::MySqlPool;
use tokio::task::JoinHandle;

use super::job_tree_infra::TestEnv;
use super::job_tree_infra::TestService;
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

    /// Whether every task but the first consumes the previous task's output. Otherwise the tasks
    /// are independent.
    is_chain: bool,
}

const ONE_TASK: JobShape = JobShape {
    num_tasks: 1,
    max_num_retry: 0,
    with_commit: false,
    with_cleanup: false,
    is_chain: false,
};

/// The number of children that a registering job reports.
const NUM_CHILDREN: usize = 3;

/// A lane of the inbound queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Task,
    Commit,
    Cleanup,
}

impl Lane {
    /// # Returns
    ///
    /// The lane that publishes `task_id`.
    const fn of(task_id: TaskId) -> Self {
        match task_id {
            TaskId::Index(_) => Self::Task,
            TaskId::Commit => Self::Commit,
            TaskId::Cleanup => Self::Cleanup,
        }
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
    /// runs `second` until its event, matched by `second_event`, is submitted.
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

/// How a test restarts the storage service.
#[derive(Clone, Copy, Debug)]
enum RestartMode {
    /// A new runtime starts at once.
    Direct,

    /// A first new runtime crashes in its recovery of the tree at this hook point, then the next
    /// one starts.
    CrashedRecovery(HookPoint),
}

impl RestartMode {
    /// # Returns
    ///
    /// The restart mode of the `index`-th restart of a test: the modes take turns.
    const fn nth(index: usize) -> Self {
        match index % 3 {
            0 => Self::Direct,
            1 => Self::CrashedRecovery(HookPoint::RecoveryBeforeRepair),
            _ => Self::CrashedRecovery(HookPoint::RecoveryAfterRepair),
        }
    }
}

/// What the recovery of a restart did with a tree.
#[derive(Debug)]
struct Recovery {
    /// The IDs of the jobs of the tree that recovery loaded, sorted, or [`None`] if it loaded
    /// none.
    loaded: Option<Vec<JobId>>,

    /// The IDs of the jobs of the tree that recovery repaired, sorted, or [`None`] if it loaded
    /// none.
    repaired: Option<Vec<JobId>>,
}

impl Recovery {
    /// Asserts that recovery loaded exactly the non-terminal jobs of `expected` and repaired
    /// nothing.
    fn assert_loaded_unrepaired(&self, expected: &HashMap<JobId, JobState>, what: &str) {
        let mut non_terminal: Vec<JobId> = expected
            .iter()
            .filter(|(_, state)| !state.is_terminal())
            .map(|(job_id, _)| *job_id)
            .collect();
        non_terminal.sort_by_key(JobId::get);
        let (loaded, repaired) = if non_terminal.is_empty() {
            (None, None)
        } else {
            (Some(non_terminal), Some(Vec::new()))
        };
        assert_eq!(self.loaded, loaded, "{what}: the jobs that recovery loaded");
        assert_eq!(
            self.repaired, repaired,
            "{what}: the jobs that recovery repaired"
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

/// # Returns
///
/// The number of iterations of a test whose interleavings are all forced, so that a repetition
/// only repeats them: a tenth of [`race_iterations`], at least 5.
fn sequential_iterations() -> usize {
    (race_iterations() / 10).max(5)
}

/// Starts a [`TestEnv`] whose task instance pool never considers the test's execution manager,
/// which sends no heartbeats, dead while a test runs.
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

/// Stops `env`'s runtime and starts a new one on its database, as a storage process that crashed
/// and was restarted, after the ready tasks are resent.
///
/// # Returns
///
/// The restarted environment, and what recovery did with the tree rooted at `root_id`.
///
/// # Panics
///
/// Panics if the runtime fails to stop, or a start does not behave as `mode` says.
async fn restart(env: TestEnv, root_id: JobId, mode: RestartMode) -> (TestEnv, Recovery) {
    let stopped = env.stop().await;
    if let RestartMode::CrashedRecovery(point) = mode {
        let hooks = TestHooks::default();
        let crash = hooks.arm(
            point,
            HookFilter::root(root_id),
            HookAction::Crash,
            HookCount::Once,
        );
        stopped.start_failing(hooks).await;
        assert_eq!(crash.hits(), 1, "the recovery crash at {point:?}");
    }
    let hooks = TestHooks::default();
    let loaded = hooks.arm(
        HookPoint::RecoveryBeforeRepair,
        HookFilter::root(root_id),
        HookAction::Observe,
        HookCount::Always,
    );
    let repaired = hooks.arm(
        HookPoint::RecoveryAfterRepair,
        HookFilter::root(root_id),
        HookAction::Observe,
        HookCount::Always,
    );
    let env = stopped.start(hooks).await;
    let recovery = Recovery {
        loaded: observed_job_ids(&loaded),
        repaired: observed_job_ids(&repaired),
    };
    (env, recovery)
}

/// # Returns
///
/// The job IDs in the context of `hook`'s only hit, sorted, or [`None`] if it never fired.
///
/// # Panics
///
/// Panics if the hook fired more than once.
fn observed_job_ids(hook: &ArmedHook) -> Option<Vec<JobId>> {
    assert!(hook.hits() <= 1, "a tree is recovered once per start");
    let mut job_ids = hook.last_context()?.job_ids;
    job_ids.sort_by_key(JobId::get);
    Some(job_ids)
}

/// Builds a job of `shape` whose tasks each take and return one byte-typed value.
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
        let input_sources = if shape.is_chain {
            task_index.checked_sub(1).map(|previous| {
                vec![TaskInputOutputIndex {
                    task_idx: previous,
                    position: 0,
                }]
            })
        } else {
            None
        };
        if input_sources.is_none() {
            inputs
                .append_task_input(&[0u8; 4])
                .expect("task input appending should succeed");
        }
        graph
            .insert_task(TaskDescriptor {
                tdl_context: TdlContext {
                    package: "test".to_owned(),
                    task_func: format!("t{task_index}"),
                },
                execution_policy: execution_policy.clone(),
                inputs: vec![bytes_type.clone()],
                outputs: vec![bytes_type.clone()],
                input_sources,
            })
            .expect("task insertion should succeed");
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
/// A valid one-task child job whose only input is `input_size` incompressible bytes.
///
/// # Panics
///
/// Panics if the job fails to build or to compress.
fn large_child(input_size: usize) -> ChildJobSubmission {
    let (graph, _) = build_job(ONE_TASK);
    let mut xorshift_state = 0x9e37_79b9_7f4a_7c15_u64;
    let input: Vec<u8> = (0..input_size)
        .map(|_| {
            xorshift_state ^= xorshift_state << 13;
            xorshift_state ^= xorshift_state >> 7;
            xorshift_state ^= xorshift_state << 17;
            xorshift_state.to_le_bytes()[0]
        })
        .collect();
    let mut inputs = TaskGraphInputBuilder::new();
    inputs
        .append_task_input(&input)
        .expect("task input appending should succeed");
    ChildJobSubmission {
        compressed_serialized_task_graph: compress_task_graph(&graph)
            .expect("task graph compression should succeed"),
        compressed_serialized_job_inputs: compress_job_inputs(&inputs.build())
            .expect("input compression should succeed"),
    }
}

/// # Returns
///
/// A child job of `size` bytes that never decodes.
fn opaque_child(size: usize) -> ChildJobSubmission {
    ChildJobSubmission {
        compressed_serialized_task_graph: vec![0u8; size],
        compressed_serialized_job_inputs: Vec::new(),
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
/// The outputs that the storage service stores for a job whose output-producing tasks reported
/// `values`, in task order.
///
/// # Panics
///
/// Panics if the deserialization fails.
fn job_outputs(values: &[Vec<u8>]) -> Vec<TaskOutput> {
    values
        .iter()
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
/// The filter of the hits of the tree rooted at `root_id` that concern an event of `kind`, of
/// `job_id` if set.
const fn event_filter(root_id: JobId, job_id: Option<JobId>, kind: EventKindTag) -> HookFilter {
    HookFilter {
        root_id: Some(root_id),
        job_id,
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

/// # Returns
///
/// A crash at the first hit of `point` that `filter` matches.
fn crash(env: &TestEnv, point: HookPoint, filter: HookFilter) -> ArmedHook {
    env.hooks
        .arm(point, filter, HookAction::Crash, HookCount::Once)
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

/// Runs `rpc` as the task of an RPC handler until its work reaches the pause `paused`, drops the
/// RPC by aborting its task, and returns with the work still paused.
///
/// # Panics
///
/// Panics if the RPC completes, which it cannot while its work is paused.
async fn drop_rpc_at<ResultType: Debug + Send + 'static>(
    paused: &ArmedHook,
    rpc: impl Future<Output = ResultType> + Send + 'static,
) {
    let task = tokio::spawn(rpc);
    reached(paused, 1).await;
    task.abort();
    assert!(
        task.await
            .expect_err("the dropped RPC should not complete")
            .is_cancelled()
    );
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

/// Polls `lane` until it returns `sentinel`, and asserts that the entries before it are exactly
/// those of `expected`, in any order, and that none follows it.
///
/// # Panics
///
/// Panics if the entries differ, or if the sentinel does not arrive within 10 seconds.
async fn drain_until_sentinel(
    env: &TestEnv,
    lane: Lane,
    sentinel: (JobId, TaskId),
    expected: &[(JobId, TaskId)],
) {
    let mut polled = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let max_entries = (expected.len() + 1).saturating_sub(polled.len()).max(1);
            let entries = poll_lane(env, lane, max_entries, Duration::from_secs(1)).await;
            if let Some(position) = entries.iter().position(|entry| *entry == sentinel) {
                assert_eq!(
                    &entries[position + 1..],
                    &[],
                    "entries after the sentinel in the {lane:?} lane"
                );
                polled.extend_from_slice(&entries[..position]);
                return;
            }
            polled.extend(entries);
        }
    })
    .await
    .expect("the sentinel should be published");
    let mut expected = expected.to_vec();
    let sort_key = |entry: &(JobId, TaskId)| (entry.0.get(), format!("{:?}", entry.1));
    expected.sort_by_key(sort_key);
    polled.sort_by_key(sort_key);
    assert_eq!(polled, expected, "the entries of the {lane:?} lane");
}

/// Asserts that the three lanes hold exactly the entries of `expected`, in any order. A sentinel
/// job publishes one entry into each lane after them, and each lane is polled up to its sentinel,
/// so that the check does not depend on how long a poll waits.
///
/// # Panics
///
/// Panics if a lane holds other entries, or if the sentinel job does not behave as a plain job.
async fn assert_lanes_hold(env: &TestEnv, expected: &[(JobId, TaskId)]) {
    let in_lane = |lane: Lane| -> Vec<(JobId, TaskId)> {
        expected
            .iter()
            .copied()
            .filter(|(_, task_id)| Lane::of(*task_id) == lane)
            .collect()
    };
    let (graph, inputs) = build_job(JobShape {
        with_commit: true,
        with_cleanup: true,
        ..ONE_TASK
    });
    let sentinel = env.register_job(&graph, &inputs).await;
    env.service
        .start_job(sentinel)
        .await
        .expect("the sentinel should start");
    drain_until_sentinel(
        env,
        Lane::Task,
        (sentinel, TaskId::Index(0)),
        &in_lane(Lane::Task),
    )
    .await;
    assert_eq!(
        run_task(env, sentinel, 0, &value(0xee)).await,
        JobState::CommitReady
    );
    drain_until_sentinel(
        env,
        Lane::Commit,
        (sentinel, TaskId::Commit),
        &in_lane(Lane::Commit),
    )
    .await;
    assert_eq!(
        cancel(env, sentinel)
            .await
            .expect("the sentinel should cancel"),
        JobState::CleanupReady
    );
    drain_until_sentinel(
        env,
        Lane::Cleanup,
        (sentinel, TaskId::Cleanup),
        &in_lane(Lane::Cleanup),
    )
    .await;
    assert_eq!(run_cleanup(env, sentinel).await, JobState::Cancelled);
}

/// # Returns
///
/// The result of creating an instance of `task_id` of `job_id` for `execution_manager_id`: the
/// instance's ID on success.
fn create_instance(
    env: &TestEnv,
    job_id: JobId,
    task_id: TaskId,
    execution_manager_id: ExecutionManagerId,
) -> impl Future<Output = Result<TaskInstanceId, StorageServerError>> + Send + use<> {
    let service = env.service.clone();
    async move {
        service
            .create_task_instance(service.session_id(), job_id, task_id, execution_manager_id)
            .await
            .map(|context| context.task_instance_id)
    }
}

/// # Returns
///
/// The ID of a new instance of `task_id` of `job_id` for the test's execution manager.
///
/// # Panics
///
/// Panics if the instance creation fails.
async fn instance(env: &TestEnv, job_id: JobId, task_id: TaskId) -> TaskInstanceId {
    create_instance(env, job_id, task_id, env.execution_manager_id)
        .await
        .expect("the instance creation should succeed")
}

/// # Returns
///
/// A future that reports, in session `session_id`, the success of `task_instance_id` of task
/// `task_index` of `job_id`, with the output `value` and `children`.
fn succeed_in_session(
    env: &TestEnv,
    session_id: SessionId,
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
                session_id,
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
    succeed_in_session(
        env,
        env.service.session_id(),
        job_id,
        task_index,
        task_instance_id,
        value,
        children,
    )
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
/// A future that reports the success of `task_instance_id` of `job_id`'s cleanup task.
fn succeed_cleanup(
    env: &TestEnv,
    job_id: JobId,
    task_instance_id: TaskInstanceId,
) -> impl Future<Output = JobResult> + Send + use<> {
    let service = env.service.clone();
    async move {
        service
            .succeed_cleanup_task_instance(service.session_id(), job_id, task_instance_id)
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

/// # Returns
///
/// A future that starts `job_id`.
fn start(
    env: &TestEnv,
    job_id: JobId,
) -> impl Future<Output = Result<(), StorageServerError>> + Send + use<> {
    let service = env.service.clone();
    async move { service.start_job(job_id).await }
}

/// Registers a root job of `shape`.
///
/// # Returns
///
/// The job ID.
///
/// # Panics
///
/// Panics if the registration fails.
async fn register_root(env: &TestEnv, shape: JobShape) -> JobId {
    let (graph, inputs) = build_job(shape);
    env.register_job(&graph, &inputs).await
}

/// # Returns
///
/// The task entries that a job of `shape` publishes when it starts.
const fn first_tasks(shape: JobShape) -> usize {
    if shape.is_chain { 1 } else { shape.num_tasks }
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
    let job_id = register_root(env, shape).await;
    env.service
        .start_job(job_id)
        .await
        .expect("the job should start");
    let ready: Vec<_> = (0..first_tasks(shape))
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
        ready.extend(
            (0..first_tasks(*shape)).map(|task_index| (*child_id, TaskId::Index(task_index))),
        );
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

/// Runs a new instance of `job_id`'s commit task to success.
///
/// # Returns
///
/// The job state after the report.
///
/// # Panics
///
/// Panics if the report fails.
async fn run_commit(env: &TestEnv, job_id: JobId) -> JobState {
    let task_instance_id = instance(env, job_id, TaskId::Commit).await;
    succeed_commit(env, job_id, task_instance_id)
        .await
        .expect("the commit report should succeed")
}

/// Runs a new instance of `job_id`'s cleanup task to success.
///
/// # Returns
///
/// The job state after the report.
///
/// # Panics
///
/// Panics if the report fails.
async fn run_cleanup(env: &TestEnv, job_id: JobId) -> JobState {
    let task_instance_id = instance(env, job_id, TaskId::Cleanup).await;
    succeed_cleanup(env, job_id, task_instance_id)
        .await
        .expect("the cleanup report should succeed")
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
/// The parent and root of every non-root row of the tree rooted at `root_id`, by ascending ID.
///
/// # Panics
///
/// Panics if the query fails.
async fn tree_links(env: &TestEnv, root_id: JobId) -> Vec<(JobId, JobId, JobId)> {
    sqlx::query_as::<_, (JobId, JobId, JobId)>(
        "SELECT `id`, `parent_id`, `root_id` FROM `jobs` WHERE `root_id` = ? ORDER BY `id`;",
    )
    .bind(root_id)
    .fetch_all(&env.pool)
    .await
    .expect("the link query should succeed")
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

/// Asserts that `rows` are exactly the jobs of `expected`, in their states.
fn assert_rows(rows: &[(JobId, JobState)], expected: &HashMap<JobId, JobState>, what: &str) {
    let mut expected: Vec<(JobId, JobState)> = expected
        .iter()
        .map(|(job_id, state)| (*job_id, *state))
        .collect();
    expected.sort_by_key(|(job_id, _)| job_id.get());
    assert_eq!(rows, expected, "{what}: the rows of the tree");
}

/// Asserts that the rows of the tree rooted at `root_id` are exactly the jobs of `expected`, in
/// their states, with their end times set exactly for the terminal ones; that every non-terminal
/// job is cached in its state; that every terminal job is cached in its state or, if `restarted`,
/// is not cached; and that the service has not shut down fatally.
async fn assert_tree(
    env: &TestEnv,
    root_id: JobId,
    expected: &HashMap<JobId, JobState>,
    restarted: bool,
    what: &str,
) {
    assert_rows(&tree_rows(&env.pool, root_id).await, expected, what);
    for (job_id, state) in expected {
        let cached = env.cached_job_state(*job_id).await;
        if !state.is_terminal() || (cached.is_some() && !restarted) {
            assert_eq!(cached, Some(*state), "{what}: cached state of job {job_id}");
        } else if restarted {
            assert_eq!(cached, None, "{what}: a recovered terminal job {job_id}");
        }
        let has_ended = sqlx::query_scalar::<_, bool>(
            "SELECT `ended_at` IS NOT NULL FROM `jobs` WHERE `id` = ?;",
        )
        .bind(job_id)
        .fetch_one(&env.pool)
        .await
        .expect("the end time query should succeed");
        assert_eq!(
            has_ended,
            state.is_terminal(),
            "{what}: end time of job {job_id} in state {state:?}"
        );
    }
    assert_not_fatal(env, what);
}

/// Waits until `service` shuts down, and asserts that it shut down fatally. A caller of a crashed
/// tree can see its error before the committer marks the shutdown as fatal; the gRPC handler marks
/// it on the same error.
///
/// # Panics
///
/// Panics if the service does not shut down within 30 seconds, or not fatally.
async fn assert_shut_down_fatally(service: &TestService, what: &str) {
    tokio::time::timeout(
        Duration::from_secs(30),
        service.shutdown().token().cancelled(),
    )
    .await
    .expect("the service should shut down");
    assert!(
        service.shutdown().is_fatal(),
        "{what}: the service should have shut down fatally"
    );
}

/// Asserts that the service has not shut down fatally.
fn assert_not_fatal(env: &TestEnv, what: &str) {
    assert!(
        !env.is_fatal(),
        "{what}: the service should not have shut down fatally"
    );
}

/// Asserts that the tree rooted at `root_id` has no non-terminal row, and that its root answers
/// `GetJobState` with `root_state`.
async fn assert_quiescent(env: &TestEnv, root_id: JobId, root_state: JobState, what: &str) {
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
    assert_eq!(non_terminal, 0, "{what}: the tree should be quiescent");
    assert_eq!(
        env.service
            .get_job_state(root_id)
            .await
            .expect("the root's state should be readable"),
        root_state,
        "{what}: `GetJobState` of the root"
    );
}

/// Asserts that `result` is the stale error `expected`.
fn assert_stale<ValueType: Debug>(
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

/// Asserts that `result` is what a caller sees when the tree that handles its request is poisoned
/// by a crash: the service is stopping, or the committer dropped the request.
fn assert_crashed<ValueType: Debug>(result: &Result<ValueType, StorageServerError>, what: &str) {
    match result {
        Err(
            StorageServerError::Stopping(_)
            | StorageServerError::Cache(CacheError::Internal(InternalError::TreePoisoned)),
        ) => {}
        other => panic!("{what}: expected the crash of the tree, got {other:?}"),
    }
}

/// Asserts that `result` is a stale-session error naming the current session of `env`.
fn assert_stale_session<ValueType: Debug>(
    env: &TestEnv,
    result: &Result<ValueType, StorageServerError>,
    what: &str,
) {
    match result {
        Err(StorageServerError::StaleSession(session_id))
            if *session_id == env.service.session_id() => {}
        other => panic!("{what}: expected a stale session, got {other:?}"),
    }
}

/// Asserts that `result` is the error of a job that the cache does not hold.
fn assert_job_not_found<ValueType: Debug>(
    result: &Result<ValueType, StorageServerError>,
    job_id: JobId,
    what: &str,
) {
    match result {
        Err(StorageServerError::JobNotFound(missing)) if *missing == job_id => {}
        other => panic!("{what}: expected job {job_id} not to be found, got {other:?}"),
    }
}

/// The job whose task registers children in a crashed registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Registrar {
    /// The root, before it has any child.
    Root,

    /// A leaf `L` of the tree `R → {L, S}`.
    Leaf,

    /// A leaf `L` of the tree `R → {L, S}`, whose registration shares its round with the success
    /// of its sibling `S`.
    LeafWithSibling,
}

/// Where the committer crashes in a registration's round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegistrationCrash {
    AfterReceive,
    BeforePlan,
    BeforeDbTransaction,
    AfterCommit,
    AfterRegistrarApply,
    AfterSiblingApply,
}

impl RegistrationCrash {
    /// # Returns
    ///
    /// The crash points of a registration by `registrar`.
    const fn all(registrar: Registrar) -> &'static [Self] {
        match registrar {
            Registrar::Root | Registrar::Leaf => &[
                Self::AfterReceive,
                Self::BeforePlan,
                Self::BeforeDbTransaction,
                Self::AfterCommit,
                Self::AfterRegistrarApply,
            ],
            Registrar::LeafWithSibling => &[
                Self::BeforePlan,
                Self::BeforeDbTransaction,
                Self::AfterCommit,
                Self::AfterRegistrarApply,
                Self::AfterSiblingApply,
            ],
        }
    }

    /// # Returns
    ///
    /// Whether the round's database transaction committed before the crash.
    const fn is_committed(self) -> bool {
        matches!(
            self,
            Self::AfterCommit | Self::AfterRegistrarApply | Self::AfterSiblingApply
        )
    }
}

/// A tree whose job `registrar_id` registers children.
struct RegistrationTree {
    registrar: Registrar,
    root_id: JobId,
    registrar_id: JobId,
    sibling_id: Option<JobId>,
}

impl RegistrationTree {
    /// Builds the tree of `registrar`, with every job running, and the registering job's task and
    /// its sibling's not yet instantiated.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, registrar: Registrar, root_has_commit: bool) -> Self {
        let root_id = start_root(
            env,
            JobShape {
                with_commit: root_has_commit,
                ..ONE_TASK
            },
        )
        .await;
        let (registrar_id, sibling_id) = if registrar == Registrar::Root {
            (root_id, None)
        } else {
            let child_ids = register_children(env, root_id, &[ONE_TASK, ONE_TASK]).await;
            (child_ids[0], Some(child_ids[1]))
        };
        Self {
            registrar,
            root_id,
            registrar_id,
            sibling_id,
        }
    }

    /// # Returns
    ///
    /// The crash hook of `crash` in this tree's registration round.
    fn arm_crash(&self, env: &TestEnv, crash: RegistrationCrash) -> ArmedHook {
        let point = match crash {
            RegistrationCrash::AfterReceive => HookPoint::CommitterAfterReceive,
            RegistrationCrash::BeforePlan => HookPoint::CommitterBeforePlan,
            RegistrationCrash::BeforeDbTransaction => HookPoint::CommitterBeforeDbTransaction,
            RegistrationCrash::AfterCommit => HookPoint::CommitterAfterCommit,
            RegistrationCrash::AfterRegistrarApply | RegistrationCrash::AfterSiblingApply => {
                HookPoint::CommitterAfterApplyJob
            }
        };
        let job_id = match crash {
            RegistrationCrash::AfterRegistrarApply => Some(self.registrar_id),
            RegistrationCrash::AfterSiblingApply => self.sibling_id,
            _ => None,
        };
        self::crash(
            env,
            point,
            event_filter(self.root_id, job_id, EventKindTag::RegularTasksDone),
        )
    }

    /// # Returns
    ///
    /// The states of the tree's jobs, but for the registered children, after the registration
    /// and, if this tree has it, the sibling's success, if `committed`, or before them.
    fn states(&self, committed: bool) -> HashMap<JobId, JobState> {
        let registrar_state = if committed {
            JobState::BlockingOnChildren
        } else {
            JobState::Running
        };
        let mut states = HashMap::from([
            (self.root_id, JobState::BlockingOnChildren),
            (self.registrar_id, registrar_state),
        ]);
        if let Some(sibling_id) = self.sibling_id {
            let sibling_state = if committed && self.registrar == Registrar::LeafWithSibling {
                JobState::Succeeded
            } else {
                JobState::Running
            };
            states.insert(sibling_id, sibling_state);
        }
        states
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_registration_crashed_at_any_point_registers_its_children_exactly_once()
-> anyhow::Result<()> {
    let mut env = start_env().await;
    let mut num_restarts = 0;
    for iteration in 0..race_iterations() {
        for registrar in [Registrar::Root, Registrar::Leaf, Registrar::LeafWithSibling] {
            for crash in RegistrationCrash::all(registrar) {
                let what = format!("{registrar:?}, crash {crash:?}, iteration {iteration}");
                let run =
                    RegistrationRun::crash(&env, registrar, *crash, iteration % 2 == 1, &what)
                        .await;
                let restart_mode = RestartMode::nth(num_restarts);
                num_restarts += 1;
                let (restarted, recovery) = restart(env, run.tree.root_id, restart_mode).await;
                env = restarted;
                let what = format!("{what}, restart {restart_mode:?}");
                recovery.assert_loaded_unrepaired(&run.expected, &what);
                run.assert_recovered(&env, &what).await;
                run.assert_retries_are_stale(&env, &what).await;
                run.finish(&env, &what).await;
            }
        }
    }
    env.finish().await;
    Ok(())
}

/// A registration whose committer round crashed, from the crash to the end of its tree.
struct RegistrationRun {
    tree: RegistrationTree,
    crash: RegistrationCrash,
    root_has_commit: bool,
    registrar_instance: TaskInstanceId,
    sibling_instance: Option<TaskInstanceId>,
    bystander_id: JobId,
    bystander_instance: TaskInstanceId,
    old_session: SessionId,
    children: Vec<ChildJobSubmission>,

    /// The rows of the tree at the crash.
    expected: HashMap<JobId, JobState>,

    /// The children registered at the crash.
    child_ids: Vec<JobId>,
}

impl RegistrationRun {
    /// Builds a registration tree of `registrar` and a one-job bystander tree, crashes the
    /// committer at `crash` in the round of the registration, and asserts the callers' errors, the
    /// fatal shutdown and the rows at the crash.
    ///
    /// # Returns
    ///
    /// The crashed registration.
    async fn crash(
        env: &TestEnv,
        registrar: Registrar,
        crash: RegistrationCrash,
        root_has_commit: bool,
        what: &str,
    ) -> Self {
        let tree = RegistrationTree::start(env, registrar, root_has_commit).await;
        let root_id = tree.root_id;
        let registrar_instance = instance(env, tree.registrar_id, TaskId::Index(0)).await;
        let sibling_instance = match tree.sibling_id {
            Some(sibling_id) => Some(instance(env, sibling_id, TaskId::Index(0)).await),
            None => None,
        };
        let bystander_id = start_root(env, ONE_TASK).await;
        let bystander_instance = instance(env, bystander_id, TaskId::Index(0)).await;
        let children = vec![child(ONE_TASK); NUM_CHILDREN];

        let crash_hook = tree.arm_crash(env, crash);
        let registration = succeed(
            env,
            tree.registrar_id,
            0,
            registrar_instance,
            &value(1),
            children.clone(),
        );
        if let (Registrar::LeafWithSibling, Some(sibling_id), Some(sibling_instance)) =
            (registrar, tree.sibling_id, sibling_instance)
        {
            let sibling = succeed(env, sibling_id, 0, sibling_instance, &value(2), Vec::new());
            let race = Race::start(
                env,
                root_id,
                HookPoint::CommitterAfterReceive,
                registration,
                sibling,
                event_filter(root_id, Some(sibling_id), EventKindTag::RegularTasksDone),
            )
            .await;
            let (registration, sibling) = race.finish().await;
            assert_crashed(&registration, &format!("{what}: the registration"));
            assert_crashed(&sibling, &format!("{what}: the sibling's report"));
        } else {
            assert_crashed(&registration.await, &format!("{what}: the registration"));
        }
        assert_eq!(crash_hook.hits(), 1, "{what}: the crash");
        assert_shut_down_fatally(&env.service, what).await;
        drop(crash_hook);

        let mut expected = tree.states(crash.is_committed());
        let child_ids = children_of(env, tree.registrar_id).await;
        assert_eq!(
            child_ids.len(),
            if crash.is_committed() {
                NUM_CHILDREN
            } else {
                0
            },
            "{what}: the children at the crash"
        );
        for child_id in &child_ids {
            expected.insert(*child_id, JobState::Running);
        }
        assert_rows(
            &tree_rows(&env.pool, root_id).await,
            &expected,
            &format!("{what}: at the crash"),
        );
        Self {
            tree,
            crash,
            root_has_commit,
            registrar_instance,
            sibling_instance,
            bystander_id,
            bystander_instance,
            old_session: env.service.session_id(),
            children,
            expected,
            child_ids,
        }
    }

    /// Asserts what the restarted `env` recovered: the tree's states in the cache and the rows,
    /// the blocked parent's outputs and the tree's links, exactly the resent lane entries, and the
    /// bystander tree, which then runs to its end.
    async fn assert_recovered(&self, env: &TestEnv, what: &str) {
        let tree = &self.tree;
        let root_id = tree.root_id;
        let registrar_id = tree.registrar_id;
        assert_tree(
            env,
            root_id,
            &self.expected,
            true,
            &format!("{what}: recovered"),
        )
        .await;
        if self.crash.is_committed() {
            assert_eq!(
                row_outputs(env, registrar_id).await,
                Some(job_outputs(&[value(1)])),
                "{what}: the blocked parent's outputs"
            );
            let mut links: Vec<_> = tree.sibling_id.map_or_else(Vec::new, |sibling_id| {
                vec![
                    (registrar_id, root_id, root_id),
                    (sibling_id, root_id, root_id),
                ]
            });
            links.extend(
                self.child_ids
                    .iter()
                    .map(|child_id| (*child_id, registrar_id, root_id)),
            );
            links.sort_by_key(|(job_id, ..)| job_id.get());
            assert_eq!(tree_links(env, root_id).await, links, "{what}: links");
        }

        let mut resent: Vec<(JobId, TaskId)> = if self.crash.is_committed() {
            self.child_ids
                .iter()
                .map(|child_id| (*child_id, TaskId::Index(0)))
                .collect()
        } else {
            vec![(registrar_id, TaskId::Index(0))]
        };
        if let Some(sibling_id) = tree.sibling_id
            && self.expected[&sibling_id] == JobState::Running
        {
            resent.push((sibling_id, TaskId::Index(0)));
        }
        resent.push((self.bystander_id, TaskId::Index(0)));
        assert_lanes_hold(env, &resent).await;

        assert_tree(
            env,
            self.bystander_id,
            &HashMap::from([(self.bystander_id, JobState::Running)]),
            true,
            &format!("{what}: the bystander"),
        )
        .await;
        assert_stale(
            &succeed(
                env,
                self.bystander_id,
                0,
                self.bystander_instance,
                &value(6),
                Vec::new(),
            )
            .await,
            &StaleStateError::InvalidTaskInstanceId,
            &format!("{what}: the bystander's report from before the crash"),
        );
        assert_eq!(
            run_task(env, self.bystander_id, 0, &value(6)).await,
            JobState::Succeeded,
            "{what}: the bystander"
        );
    }

    /// Asserts that the reports from before the crash, which an execution manager retries after
    /// the restart, change nothing: in the old session they are rejected for it, and in the new
    /// one they are stale.
    async fn assert_retries_are_stale(&self, env: &TestEnv, what: &str) {
        let tree = &self.tree;
        let retried = |session_id| {
            succeed_in_session(
                env,
                session_id,
                tree.registrar_id,
                0,
                self.registrar_instance,
                &value(1),
                self.children.clone(),
            )
        };
        assert_stale_session(
            env,
            &retried(self.old_session).await,
            &format!("{what}: the retried registration in the old session"),
        );
        let retried_stale = if self.crash.is_committed() {
            StaleStateError::JobNoLongerRunning
        } else {
            StaleStateError::InvalidTaskInstanceId
        };
        assert_stale(
            &retried(env.service.session_id()).await,
            &retried_stale,
            &format!("{what}: the retried registration"),
        );
        if let (Some(sibling_id), Some(sibling_instance)) = (tree.sibling_id, self.sibling_instance)
            && self.expected[&sibling_id] == JobState::Succeeded
        {
            assert_job_not_found(
                &succeed(env, sibling_id, 0, sibling_instance, &value(2), Vec::new()).await,
                sibling_id,
                &format!("{what}: the retried sibling's report"),
            );
        }
        assert_eq!(
            children_of(env, tree.registrar_id).await,
            self.child_ids,
            "{what}: the children after the retries"
        );
    }

    /// Reruns what the crash lost, runs every job to its end, and asserts that the parent has
    /// exactly one set of children and that the tree is quiescent.
    async fn finish(self, env: &TestEnv, what: &str) {
        let tree = &self.tree;
        let root_id = tree.root_id;
        let registrar_id = tree.registrar_id;
        let child_ids = if self.crash.is_committed() {
            self.child_ids
        } else {
            let registrar_instance = instance(env, registrar_id, TaskId::Index(0)).await;
            assert_job_state(
                &succeed(
                    env,
                    registrar_id,
                    0,
                    registrar_instance,
                    &value(1),
                    self.children,
                )
                .await,
                JobState::BlockingOnChildren,
                &format!("{what}: the rerun registration"),
            );
            let child_ids = children_of(env, registrar_id).await;
            assert_eq!(child_ids.len(), NUM_CHILDREN, "{what}: the rerun children");
            let ready: Vec<_> = child_ids
                .iter()
                .map(|child_id| (*child_id, TaskId::Index(0)))
                .collect();
            drain_lane(env, Lane::Task, &ready).await;
            child_ids
        };
        if let Some(sibling_id) = tree.sibling_id
            && env.cached_job_state(sibling_id).await == Some(JobState::Running)
        {
            assert_eq!(
                run_task(env, sibling_id, 0, &value(2)).await,
                JobState::Succeeded
            );
        }
        for (index, child_id) in child_ids.iter().enumerate() {
            assert_eq!(
                run_task(
                    env,
                    *child_id,
                    0,
                    &value(0x20 + u8::try_from(index).expect("small index"))
                )
                .await,
                JobState::Succeeded,
                "{what}: child {index}"
            );
            if index + 1 < child_ids.len() {
                assert_eq!(
                    env.cached_job_state(registrar_id).await,
                    Some(JobState::BlockingOnChildren),
                    "{what}: the parent before its last child"
                );
            }
        }
        if self.root_has_commit {
            drain_lane(env, Lane::Commit, &[(root_id, TaskId::Commit)]).await;
            assert_eq!(
                env.cached_job_state(root_id).await,
                Some(JobState::CommitReady)
            );
            assert_eq!(run_commit(env, root_id).await, JobState::Succeeded);
        }

        let done: HashMap<JobId, JobState> = self
            .expected
            .keys()
            .chain(&child_ids)
            .map(|job_id| (*job_id, JobState::Succeeded))
            .collect();
        assert_tree(env, root_id, &done, false, &format!("{what}: done")).await;
        assert_eq!(children_of(env, registrar_id).await, child_ids);
        assert_eq!(tree_errors(env, root_id).await, Vec::new());
        assert_eq!(
            row_outputs(env, registrar_id).await,
            Some(job_outputs(&[value(1)])),
            "{what}: the parent's outputs"
        );
        assert_lanes_hold(env, &[]).await;
        assert_quiescent(env, root_id, JobState::Succeeded, what).await;
    }
}

/// The event whose cascade a crash interrupts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cascade {
    /// The leaf's last task succeeds, and the success climbs.
    Success,

    /// The leaf's last task fails, and the failure fails the chain and aborts the rest of the
    /// tree.
    Failure,
}

/// A job of the cascade tree `R → {P → {L}, Q}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CascadeJob {
    Root,
    Parent,
    Leaf,
    Uncle,
}

/// Where the committer crashes in the cascade's round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CascadeCrash {
    BeforeDbTransaction,
    AfterCommit,
    AfterApplyOf(CascadeJob),
}

/// The tree `R → {P → {L}, Q}`, where `Q` has a cleanup task.
struct CascadeTree {
    root_id: JobId,
    parent_id: JobId,
    leaf_id: JobId,
    uncle_id: JobId,
    root_has_commit: bool,
    uncle_done: bool,
}

impl CascadeTree {
    /// Builds the tree with the leaf running and, if `uncle_done`, `Q` succeeded.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, root_has_commit: bool, uncle_done: bool) -> Self {
        let root_id = start_root(
            env,
            JobShape {
                with_commit: root_has_commit,
                ..ONE_TASK
            },
        )
        .await;
        let child_ids = register_children(
            env,
            root_id,
            &[
                ONE_TASK,
                JobShape {
                    with_cleanup: true,
                    ..ONE_TASK
                },
            ],
        )
        .await;
        let leaf_id = register_children(env, child_ids[0], &[ONE_TASK]).await[0];
        if uncle_done {
            assert_eq!(
                run_task(env, child_ids[1], 0, &value(3)).await,
                JobState::Succeeded
            );
        }
        Self {
            root_id,
            parent_id: child_ids[0],
            leaf_id,
            uncle_id: child_ids[1],
            root_has_commit,
            uncle_done,
        }
    }

    /// # Returns
    ///
    /// The ID of `job`.
    const fn id(&self, job: CascadeJob) -> JobId {
        match job {
            CascadeJob::Root => self.root_id,
            CascadeJob::Parent => self.parent_id,
            CascadeJob::Leaf => self.leaf_id,
            CascadeJob::Uncle => self.uncle_id,
        }
    }

    /// # Returns
    ///
    /// The crash points of `cascade` in a tree whose uncle is done if `uncle_done`: before the
    /// commit, after it, and after each job's apply.
    fn crashes(cascade: Cascade, uncle_done: bool) -> Vec<CascadeCrash> {
        let mut applied = vec![CascadeJob::Leaf, CascadeJob::Parent];
        if cascade == Cascade::Failure || uncle_done {
            applied.push(CascadeJob::Root);
        }
        if cascade == Cascade::Failure && !uncle_done {
            applied.push(CascadeJob::Uncle);
        }
        let mut crashes = vec![CascadeCrash::BeforeDbTransaction, CascadeCrash::AfterCommit];
        crashes.extend(applied.into_iter().map(CascadeCrash::AfterApplyOf));
        crashes
    }

    /// # Returns
    ///
    /// The states of the tree's jobs after `cascade` if `committed`, or before it.
    fn states(&self, cascade: Cascade, committed: bool) -> HashMap<JobId, JobState> {
        let uncle = if self.uncle_done {
            JobState::Succeeded
        } else {
            JobState::Running
        };
        if !committed {
            return HashMap::from([
                (self.root_id, JobState::BlockingOnChildren),
                (self.parent_id, JobState::BlockingOnChildren),
                (self.leaf_id, JobState::Running),
                (self.uncle_id, uncle),
            ]);
        }
        match cascade {
            Cascade::Success => HashMap::from([
                (self.root_id, self.root_after_success()),
                (self.parent_id, JobState::Succeeded),
                (self.leaf_id, JobState::Succeeded),
                (self.uncle_id, uncle),
            ]),
            Cascade::Failure => HashMap::from([
                (self.root_id, JobState::Failed),
                (self.parent_id, JobState::Failed),
                (self.leaf_id, JobState::Failed),
                (
                    self.uncle_id,
                    if self.uncle_done {
                        JobState::Succeeded
                    } else {
                        JobState::CleanupReady
                    },
                ),
            ]),
        }
    }

    /// # Returns
    ///
    /// The root's state after the leaf's success.
    const fn root_after_success(&self) -> JobState {
        if !self.uncle_done {
            JobState::BlockingOnChildren
        } else if self.root_has_commit {
            JobState::CommitReady
        } else {
            JobState::Succeeded
        }
    }

    /// # Returns
    ///
    /// The lane entries that the tree's jobs in `states` publish when they are recovered.
    fn resent(states: &HashMap<JobId, JobState>) -> Vec<(JobId, TaskId)> {
        let mut resent = Vec::new();
        for (job_id, state) in states {
            match state {
                JobState::Running => resent.push((*job_id, TaskId::Index(0))),
                JobState::CommitReady => resent.push((*job_id, TaskId::Commit)),
                JobState::CleanupReady => resent.push((*job_id, TaskId::Cleanup)),
                _ => {}
            }
        }
        resent
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_cascade_crashed_midway_resumes_from_what_it_committed() -> anyhow::Result<()> {
    let mut env = start_env().await;
    let mut num_restarts = 0;
    for iteration in 0..race_iterations() {
        for cascade in [Cascade::Success, Cascade::Failure] {
            for uncle_done in [false, true] {
                for crash in CascadeTree::crashes(cascade, uncle_done) {
                    let tree = CascadeTree::start(&env, iteration % 2 == 1, uncle_done).await;
                    env = assert_cascade_survives_crash(
                        env,
                        &tree,
                        cascade,
                        crash,
                        RestartMode::nth(num_restarts),
                        iteration,
                    )
                    .await;
                    num_restarts += 1;
                }
            }
        }
    }
    env.finish().await;
    Ok(())
}

/// Crashes the committer at `crash` in the round of `cascade` in `tree`, restarts the storage
/// service as `restart_mode` says (directly if recovery loads nothing of the tree), and drives
/// the tree to the end.
///
/// # Returns
///
/// The restarted environment, with the tree quiescent.
async fn assert_cascade_survives_crash(
    env: TestEnv,
    tree: &CascadeTree,
    cascade: Cascade,
    crash: CascadeCrash,
    restart_mode: RestartMode,
    iteration: usize,
) -> TestEnv {
    let what = format!(
        "{cascade:?}, uncle done {}, root commit {}, crash {crash:?}, iteration {iteration}",
        tree.uncle_done, tree.root_has_commit
    );
    let root_id = tree.root_id;
    let leaf_instance = instance(&env, tree.leaf_id, TaskId::Index(0)).await;
    let kind = match cascade {
        Cascade::Success => EventKindTag::RegularTasksDone,
        Cascade::Failure => EventKindTag::TerminalFailure,
    };
    let (point, job_id) = match crash {
        CascadeCrash::BeforeDbTransaction => (HookPoint::CommitterBeforeDbTransaction, None),
        CascadeCrash::AfterCommit => (HookPoint::CommitterAfterCommit, None),
        CascadeCrash::AfterApplyOf(job) => (HookPoint::CommitterAfterApplyJob, Some(tree.id(job))),
    };
    let crash_hook = self::crash(&env, point, event_filter(root_id, job_id, kind));
    assert_crashed(
        &tree.leaf_event(&env, cascade, leaf_instance).await,
        &format!("{what}: the leaf's report"),
    );
    assert_eq!(crash_hook.hits(), 1, "{what}: the crash");
    assert_shut_down_fatally(&env.service, &what).await;
    drop(crash_hook);

    let committed = !matches!(crash, CascadeCrash::BeforeDbTransaction);
    let expected = tree.states(cascade, committed);
    assert_rows(
        &tree_rows(&env.pool, root_id).await,
        &expected,
        &format!("{what}: at the crash"),
    );
    let loads_tree = expected.values().any(|state| !state.is_terminal());
    let restart_mode = if loads_tree {
        restart_mode
    } else {
        RestartMode::Direct
    };
    let (env, recovery) = restart(env, root_id, restart_mode).await;
    let what = format!("{what}, restart {restart_mode:?}");
    recovery.assert_loaded_unrepaired(&expected, &what);
    assert_tree(
        &env,
        root_id,
        &expected,
        true,
        &format!("{what}: recovered"),
    )
    .await;
    assert_lanes_hold(&env, &CascadeTree::resent(&expected)).await;

    let retried = tree.leaf_event(&env, cascade, leaf_instance).await;
    if committed {
        assert_job_not_found(
            &retried,
            tree.leaf_id,
            &format!("{what}: the retried report"),
        );
    } else {
        assert_stale(
            &retried,
            &StaleStateError::InvalidTaskInstanceId,
            &format!("{what}: the retried report"),
        );
        tree.rerun(&env, cascade, &what).await;
    }
    tree.finish(&env, cascade, &what).await;
    env
}

impl CascadeTree {
    /// # Returns
    ///
    /// A future that reports the leaf's event of `cascade` from `task_instance_id`.
    fn leaf_event(
        &self,
        env: &TestEnv,
        cascade: Cascade,
        task_instance_id: TaskInstanceId,
    ) -> Pin<Box<dyn Future<Output = JobResult> + Send>> {
        match cascade {
            Cascade::Success => Box::pin(succeed(
                env,
                self.leaf_id,
                0,
                task_instance_id,
                &value(4),
                Vec::new(),
            )),
            Cascade::Failure => Box::pin(fail(
                env,
                self.leaf_id,
                TaskId::Index(0),
                task_instance_id,
                "leaf failure",
            )),
        }
    }

    /// Reruns the leaf's event of `cascade` on a new instance, as the leaf's task does after a
    /// crash lost its event, and asserts the tree after it.
    async fn rerun(&self, env: &TestEnv, cascade: Cascade, what: &str) {
        let leaf_instance = instance(env, self.leaf_id, TaskId::Index(0)).await;
        let expected_state = match cascade {
            Cascade::Success => JobState::Succeeded,
            Cascade::Failure => JobState::Failed,
        };
        assert_job_state(
            &self.leaf_event(env, cascade, leaf_instance).await,
            expected_state,
            &format!("{what}: the rerun report"),
        );
        match cascade {
            Cascade::Success if self.uncle_done && self.root_has_commit => {
                drain_lane(env, Lane::Commit, &[(self.root_id, TaskId::Commit)]).await;
            }
            Cascade::Failure if !self.uncle_done => {
                drain_lane(env, Lane::Cleanup, &[(self.uncle_id, TaskId::Cleanup)]).await;
            }
            _ => {}
        }
        assert_tree(
            env,
            self.root_id,
            &self.states(cascade, true),
            false,
            &format!("{what}: rerun"),
        )
        .await;
    }

    /// Runs the rest of the tree after `cascade` to its end, and asserts its final states and
    /// errors, and that it is quiescent.
    async fn finish(&self, env: &TestEnv, cascade: Cascade, what: &str) {
        let root_id = self.root_id;
        let final_states = match cascade {
            Cascade::Success => {
                if !self.uncle_done {
                    assert_eq!(
                        run_task(env, self.uncle_id, 0, &value(3)).await,
                        JobState::Succeeded,
                        "{what}: the uncle's report"
                    );
                    if self.root_has_commit {
                        drain_lane(env, Lane::Commit, &[(root_id, TaskId::Commit)]).await;
                    }
                }
                if self.root_has_commit {
                    assert_eq!(run_commit(env, root_id).await, JobState::Succeeded);
                }
                assert_eq!(tree_errors(env, root_id).await, Vec::new());
                HashMap::from([
                    (root_id, JobState::Succeeded),
                    (self.parent_id, JobState::Succeeded),
                    (self.leaf_id, JobState::Succeeded),
                    (self.uncle_id, JobState::Succeeded),
                ])
            }
            Cascade::Failure => {
                let uncle = if self.uncle_done {
                    JobState::Succeeded
                } else {
                    assert_eq!(run_cleanup(env, self.uncle_id).await, JobState::Cancelled);
                    JobState::Cancelled
                };
                let propagated = format!("descendant job {} failed: leaf failure", self.leaf_id);
                let mut errors = vec![
                    (root_id, propagated.clone()),
                    (self.parent_id, propagated),
                    (self.leaf_id, "leaf failure".to_owned()),
                ];
                errors.sort_by_key(|(job_id, _)| job_id.get());
                assert_eq!(tree_errors(env, root_id).await, errors, "{what}: errors");
                HashMap::from([
                    (root_id, JobState::Failed),
                    (self.parent_id, JobState::Failed),
                    (self.leaf_id, JobState::Failed),
                    (self.uncle_id, uncle),
                ])
            }
        };
        assert_tree(env, root_id, &final_states, false, &format!("{what}: done")).await;
        assert_lanes_hold(env, &[]).await;
        assert_quiescent(env, root_id, final_states[&root_id], what).await;
    }
}

/// Where the committer crashes in a cancel's round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CancelCrash {
    BeforeDbTransaction,
    AfterCommit,
    AfterApplyOf(CancelJob),
}

/// A job of the cancel tree `R → {P → {L}, K, M}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CancelJob {
    Root,
    Parent,
    Leaf,
    Plain,
}

/// The tree `R → {P → {L}, K, M}`: `R`, `P` and `L` have cleanup tasks, `K` has a commit task and
/// is `CommitReady`, `M` has neither.
struct CancelTree {
    root: JobId,
    parent: JobId,
    leaf: JobId,
    commit_ready: JobId,
    plain: JobId,
}

impl CancelTree {
    /// Builds the tree, and drains `K`'s commit entry.
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv) -> Self {
        let with_cleanup = JobShape {
            with_cleanup: true,
            ..ONE_TASK
        };
        let root_id = start_root(env, with_cleanup).await;
        let child_ids = register_children(
            env,
            root_id,
            &[
                with_cleanup,
                JobShape {
                    with_commit: true,
                    ..ONE_TASK
                },
                ONE_TASK,
            ],
        )
        .await;
        let leaf_id = register_children(env, child_ids[0], &[with_cleanup]).await[0];
        assert_eq!(
            run_task(env, child_ids[1], 0, &value(5)).await,
            JobState::CommitReady
        );
        drain_lane(env, Lane::Commit, &[(child_ids[1], TaskId::Commit)]).await;
        Self {
            root: root_id,
            parent: child_ids[0],
            leaf: leaf_id,
            commit_ready: child_ids[1],
            plain: child_ids[2],
        }
    }

    /// # Returns
    ///
    /// The ID of `job`.
    const fn id(&self, job: CancelJob) -> JobId {
        match job {
            CancelJob::Root => self.root,
            CancelJob::Parent => self.parent,
            CancelJob::Leaf => self.leaf,
            CancelJob::Plain => self.plain,
        }
    }

    /// # Returns
    ///
    /// The states of the tree's jobs after the cancel if `cancelled`, or before it.
    fn states(&self, cancelled: bool) -> HashMap<JobId, JobState> {
        if cancelled {
            HashMap::from([
                (self.root, JobState::CleanupReady),
                (self.parent, JobState::CleanupReady),
                (self.leaf, JobState::CleanupReady),
                (self.commit_ready, JobState::CommitReady),
                (self.plain, JobState::Cancelled),
            ])
        } else {
            HashMap::from([
                (self.root, JobState::BlockingOnChildren),
                (self.parent, JobState::BlockingOnChildren),
                (self.leaf, JobState::Running),
                (self.commit_ready, JobState::CommitReady),
                (self.plain, JobState::Running),
            ])
        }
    }

    /// # Returns
    ///
    /// The cleanup entries that the cancel publishes.
    const fn cleanups(&self) -> [(JobId, TaskId); 3] {
        [
            (self.root, TaskId::Cleanup),
            (self.parent, TaskId::Cleanup),
            (self.leaf, TaskId::Cleanup),
        ]
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_cancel_crashed_at_any_point_cancels_the_tree_once() -> anyhow::Result<()> {
    let mut env = start_env().await;
    let mut num_restarts = 0;
    for iteration in 0..race_iterations() {
        for crash in [
            CancelCrash::BeforeDbTransaction,
            CancelCrash::AfterCommit,
            CancelCrash::AfterApplyOf(CancelJob::Leaf),
            CancelCrash::AfterApplyOf(CancelJob::Parent),
            CancelCrash::AfterApplyOf(CancelJob::Plain),
            CancelCrash::AfterApplyOf(CancelJob::Root),
        ] {
            let what = format!("crash {crash:?}, iteration {iteration}");
            env = assert_cancel_survives_crash(env, crash, RestartMode::nth(num_restarts), &what)
                .await;
            num_restarts += 1;
        }
    }
    env.finish().await;
    Ok(())
}

/// Crashes the committer at `crash` in the round of a root cancel, restarts the storage service as
/// `restart_mode` says, and drives the tree to the end.
///
/// # Returns
///
/// The restarted environment, with the tree quiescent.
async fn assert_cancel_survives_crash(
    env: TestEnv,
    crash: CancelCrash,
    restart_mode: RestartMode,
    what: &str,
) -> TestEnv {
    let tree = CancelTree::start(&env).await;
    let root_id = tree.root;
    let (point, job_id) = match crash {
        CancelCrash::BeforeDbTransaction => (HookPoint::CommitterBeforeDbTransaction, None),
        CancelCrash::AfterCommit => (HookPoint::CommitterAfterCommit, None),
        CancelCrash::AfterApplyOf(job) => (HookPoint::CommitterAfterApplyJob, Some(tree.id(job))),
    };
    let crash_hook = self::crash(
        &env,
        point,
        event_filter(root_id, job_id, EventKindTag::CancelRoot),
    );
    assert_crashed(&cancel(&env, root_id).await, &format!("{what}: the cancel"));
    assert_eq!(crash_hook.hits(), 1, "{what}: the crash");
    assert_shut_down_fatally(&env.service, what).await;
    drop(crash_hook);

    let committed = crash != CancelCrash::BeforeDbTransaction;
    let expected = tree.states(committed);
    assert_rows(
        &tree_rows(&env.pool, root_id).await,
        &expected,
        &format!("{what}: at the crash"),
    );
    let (env, recovery) = restart(env, root_id, restart_mode).await;
    let what = format!("{what}, restart {restart_mode:?}");
    recovery.assert_loaded_unrepaired(&expected, &what);
    assert_tree(
        &env,
        root_id,
        &expected,
        true,
        &format!("{what}: recovered"),
    )
    .await;
    let mut resent = vec![(tree.commit_ready, TaskId::Commit)];
    if committed {
        resent.extend(tree.cleanups());
    } else {
        resent.push((tree.leaf, TaskId::Index(0)));
        resent.push((tree.plain, TaskId::Index(0)));
    }
    assert_lanes_hold(&env, &resent).await;

    let retried = cancel(&env, root_id).await;
    if committed {
        assert_stale(
            &retried,
            &StaleStateError::JobCancellationAlreadyRequested,
            &format!("{what}: the retried cancel"),
        );
    } else {
        assert_job_state(
            &retried,
            JobState::CleanupReady,
            &format!("{what}: the retried cancel"),
        );
        drain_lane(&env, Lane::Cleanup, &tree.cleanups()).await;
        assert_tree(
            &env,
            root_id,
            &tree.states(true),
            false,
            &format!("{what}: cancelled again"),
        )
        .await;
    }

    tree.finish(&env, &what).await;
    env
}

impl CancelTree {
    /// Runs the skipped job's commit and every cleanup after the cancel, and asserts that the
    /// tree ends cancelled but for the skipped job, which succeeds without propagating.
    async fn finish(&self, env: &TestEnv, what: &str) {
        assert_eq!(
            run_commit(env, self.commit_ready).await,
            JobState::Succeeded,
            "{what}: the skipped job's commit"
        );
        assert_eq!(
            env.cached_job_state(self.root).await,
            Some(JobState::CleanupReady),
            "{what}: the root after the skipped job's commit"
        );
        for job_id in [self.leaf, self.parent, self.root] {
            assert_eq!(
                run_cleanup(env, job_id).await,
                JobState::Cancelled,
                "{what}: the cleanup of job {job_id}"
            );
        }
        let mut done = self.states(true);
        for state in done.values_mut() {
            *state = if *state == JobState::CommitReady {
                JobState::Succeeded
            } else {
                JobState::Cancelled
            };
        }
        assert_tree(env, self.root, &done, false, &format!("{what}: done")).await;
        assert_eq!(tree_errors(env, self.root).await, Vec::new());
        assert_stale(
            &cancel(env, self.root).await,
            &StaleStateError::JobAlreadyCancelled,
            &format!("{what}: one more cancel"),
        );
        assert_lanes_hold(env, &[]).await;
        assert_quiescent(env, self.root, JobState::Cancelled, what).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_superseded_runtime_never_registers_children_twice() -> anyhow::Result<()> {
    let mut env = start_env().await;
    for iteration in 0..race_iterations() {
        for committed_before_successor in [false, true] {
            let what = format!(
                "committed before the successor {committed_before_successor}, iteration \
                 {iteration}"
            );
            env = assert_superseded_registration(env, committed_before_successor, &what).await;
        }
    }
    env.finish().await;
    Ok(())
}

/// Pauses a registration of a leaf of `R → {L, S}` before its database transaction, or after its
/// commit if `committed_before_successor`, starts a successor runtime on the same database, lets
/// the superseded runtime go on, and drives the tree to the end on the successor.
///
/// # Returns
///
/// The environment of the successor, with the tree quiescent.
async fn assert_superseded_registration(
    mut env: TestEnv,
    committed_before_successor: bool,
    what: &str,
) -> TestEnv {
    let tree = RegistrationTree::start(&env, Registrar::Leaf, false).await;
    let root_id = tree.root_id;
    let leaf_id = tree.registrar_id;
    let sibling_id = tree.sibling_id.expect("the tree should have a sibling");
    let leaf_instance = instance(&env, leaf_id, TaskId::Index(0)).await;
    let sibling_instance = instance(&env, sibling_id, TaskId::Index(0)).await;
    let point = if committed_before_successor {
        HookPoint::CommitterAfterCommit
    } else {
        HookPoint::CommitterBeforeDbTransaction
    };
    let paused = pause(
        &env,
        point,
        event_filter(root_id, Some(leaf_id), EventKindTag::RegularTasksDone),
    );
    let registration = tokio::spawn(succeed(
        &env,
        leaf_id,
        0,
        leaf_instance,
        &value(1),
        vec![child(ONE_TASK); NUM_CHILDREN],
    ));
    reached(&paused, 1).await;
    let old_hooks = env.hooks.clone();

    let (old_runtime, old_service) = env.start_successor().await;
    paused.release();
    let registration = registration
        .await
        .expect("the registration should not panic");
    if committed_before_successor {
        assert_job_state(
            &registration,
            JobState::BlockingOnChildren,
            &format!("{what}: the registration committed before the successor"),
        );
        assert!(
            !old_service.shutdown().is_fatal(),
            "{what}: the superseded runtime has not written since"
        );
        let sibling_report = old_service
            .succeed_task_instance(
                old_service.session_id(),
                sibling_id,
                sibling_instance,
                0,
                serialized_output(&value(2)),
                Vec::new(),
            )
            .await;
        assert_crashed(
            &sibling_report,
            &format!("{what}: a write of the superseded runtime"),
        );
    } else {
        assert_crashed(
            &registration,
            &format!("{what}: the fenced-off registration"),
        );
    }
    assert_shut_down_fatally(&old_service, &format!("{what}: the superseded runtime")).await;
    old_hooks.release_all();
    drop(paused);
    drop(old_service);
    let _ = old_runtime.stop().await;
    assert_successor_finishes(&env, &tree, committed_before_successor, what).await;
    env
}

/// Asserts that the successor runtime of a superseded registration of `tree` recovered exactly the
/// rows that the superseded runtime committed before it, the registration if
/// `committed_before_successor`, and drives the tree to the end on it.
async fn assert_successor_finishes(
    env: &TestEnv,
    tree: &RegistrationTree,
    committed_before_successor: bool,
    what: &str,
) {
    let root_id = tree.root_id;
    let leaf_id = tree.registrar_id;
    let sibling_id = tree.sibling_id.expect("the tree should have a sibling");

    let mut expected = HashMap::from([
        (root_id, JobState::BlockingOnChildren),
        (
            leaf_id,
            if committed_before_successor {
                JobState::BlockingOnChildren
            } else {
                JobState::Running
            },
        ),
        (sibling_id, JobState::Running),
    ]);
    let child_ids = children_of(env, leaf_id).await;
    assert_eq!(
        child_ids.len(),
        if committed_before_successor {
            NUM_CHILDREN
        } else {
            0
        },
        "{what}: the children the superseded runtime wrote"
    );
    for child_id in &child_ids {
        expected.insert(*child_id, JobState::Running);
    }
    assert_tree(env, root_id, &expected, true, &format!("{what}: successor")).await;
    let mut resent = vec![(sibling_id, TaskId::Index(0))];
    if committed_before_successor {
        resent.extend(
            child_ids
                .iter()
                .map(|child_id| (*child_id, TaskId::Index(0))),
        );
    } else {
        resent.push((leaf_id, TaskId::Index(0)));
    }
    assert_lanes_hold(env, &resent).await;

    let child_ids = if committed_before_successor {
        child_ids
    } else {
        assert_eq!(
            register_children(env, leaf_id, &[ONE_TASK; NUM_CHILDREN])
                .await
                .len(),
            NUM_CHILDREN
        );
        children_of(env, leaf_id).await
    };
    assert_eq!(
        run_task(env, sibling_id, 0, &value(2)).await,
        JobState::Succeeded
    );
    for child_id in &child_ids {
        assert_eq!(
            run_task(env, *child_id, 0, &value(3)).await,
            JobState::Succeeded
        );
    }
    let done: HashMap<JobId, JobState> = expected
        .keys()
        .chain(&child_ids)
        .map(|job_id| (*job_id, JobState::Succeeded))
        .collect();
    assert_tree(env, root_id, &done, false, &format!("{what}: done")).await;
    assert_eq!(children_of(env, leaf_id).await, child_ids);
    assert_lanes_hold(env, &[]).await;
    assert_quiescent(env, root_id, JobState::Succeeded, what).await;
}

/// An RPC whose future a test drops while its work is in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DropCase {
    /// The last task's success report, between its task's success and the job's count of
    /// incomplete tasks.
    AfterTcbFlip,

    /// As [`Self::AfterTcbFlip`], for a report that registers children.
    AfterTcbFlipRegistering,

    /// As [`Self::AfterTcbFlip`], for the report of task 0 of a two-task job while the report of
    /// task 1 completes in full.
    AfterTcbFlipWhileSiblingTaskReports,

    /// As [`Self::AfterTcbFlip`], for the reports of both tasks of a two-task job.
    AfterTcbFlipBothTasks,

    /// As [`Self::AfterTcbFlip`], while a cancel of the root commits and waits to apply.
    AfterTcbFlipRacingCancel,

    /// The last task's success report, after the job's guard is dropped, before the submit.
    BeforeSubmit,

    /// The success report of the first task of a two-task chain, before its successor is locked.
    BeforeSuccessorLock,

    /// The last task's success report, with its event paused in the committer at this point.
    InCommitter(HookPoint),

    /// A terminal task failure, with its event paused in the committer.
    FailureInCommitter,

    /// A commit task's success, with its event paused in the committer.
    CommitInCommitter,

    /// A cleanup task's success, with its event paused in the committer.
    CleanupInCommitter,

    /// A root cancel, with its event paused in the committer.
    CancelInCommitter,

    /// A root start, with its event paused in the committer.
    StartInCommitter,

    /// An instance creation, before the instance is sent to the task instance pool.
    CreateInstance,

    /// As [`Self::CreateInstance`], for an execution manager that the pool finds dead.
    CreateInstanceForDeadExecutionManager,
}

impl DropCase {
    const ALL: [Self; 18] = [
        Self::AfterTcbFlip,
        Self::AfterTcbFlipRegistering,
        Self::AfterTcbFlipWhileSiblingTaskReports,
        Self::AfterTcbFlipBothTasks,
        Self::AfterTcbFlipRacingCancel,
        Self::BeforeSubmit,
        Self::BeforeSuccessorLock,
        Self::InCommitter(HookPoint::CommitterAfterReceive),
        Self::InCommitter(HookPoint::CommitterBeforePlan),
        Self::InCommitter(HookPoint::CommitterBeforeDbTransaction),
        Self::InCommitter(HookPoint::CommitterAfterCommit),
        Self::FailureInCommitter,
        Self::CommitInCommitter,
        Self::CleanupInCommitter,
        Self::CancelInCommitter,
        Self::StartInCommitter,
        Self::CreateInstance,
        Self::CreateInstanceForDeadExecutionManager,
    ];

    /// # Returns
    ///
    /// The shape of the job whose RPC is dropped.
    const fn shape(self) -> JobShape {
        match self {
            Self::AfterTcbFlipWhileSiblingTaskReports | Self::AfterTcbFlipBothTasks => JobShape {
                num_tasks: 2,
                ..ONE_TASK
            },
            Self::BeforeSuccessorLock => JobShape {
                num_tasks: 2,
                is_chain: true,
                ..ONE_TASK
            },
            Self::CommitInCommitter => JobShape {
                with_commit: true,
                ..ONE_TASK
            },
            Self::CleanupInCommitter | Self::AfterTcbFlipRacingCancel => JobShape {
                with_cleanup: true,
                ..ONE_TASK
            },
            _ => ONE_TASK,
        }
    }
}

/// The tree of a dropped RPC: the job `J` alone, or `J` as a leaf of `R → {J, S}`, where `S` has
/// a cleanup task.
struct DropTree {
    root: JobId,
    job: JobId,
    sibling: Option<JobId>,
}

impl DropTree {
    /// Builds the tree with `J` of `shape`, started unless `started` is false (a root only).
    ///
    /// # Returns
    ///
    /// The tree.
    async fn start(env: &TestEnv, shape: JobShape, in_tree: bool, started: bool) -> Self {
        if !in_tree {
            let job_id = if started {
                start_root(env, shape).await
            } else {
                register_root(env, shape).await
            };
            return Self {
                root: job_id,
                job: job_id,
                sibling: None,
            };
        }
        let root_id = start_root(env, ONE_TASK).await;
        let child_ids = register_children(
            env,
            root_id,
            &[
                shape,
                JobShape {
                    with_cleanup: true,
                    ..ONE_TASK
                },
            ],
        )
        .await;
        Self {
            root: root_id,
            job: child_ids[0],
            sibling: Some(child_ids[1]),
        }
    }

    /// Runs the sibling, if any, to success, and asserts that the root then succeeds.
    async fn finish_sibling(&self, env: &TestEnv, what: &str) {
        if let Some(sibling_id) = self.sibling {
            assert_eq!(
                env.cached_job_state(self.root).await,
                Some(JobState::BlockingOnChildren),
                "{what}: the root before the sibling's success"
            );
            assert_eq!(
                run_task(env, sibling_id, 0, &value(9)).await,
                JobState::Succeeded
            );
        }
    }

    /// # Returns
    ///
    /// The states of every job of the tree once `J` is `job_state` and the rest of the tree has
    /// finished as `rest`.
    fn states(&self, job_state: JobState, rest: JobState) -> HashMap<JobId, JobState> {
        let mut states = HashMap::from([(self.root, rest), (self.job, job_state)]);
        if let Some(sibling_id) = self.sibling {
            states.insert(sibling_id, rest);
        }
        states
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_dropped_rpc_never_strands_its_job() -> anyhow::Result<()> {
    let env = start_env().await;
    let dead_execution_manager_id = register_dead_execution_manager(&env).await;
    for iteration in 0..race_iterations() {
        for case in DropCase::ALL {
            for in_tree in [false, true] {
                if in_tree && case == DropCase::StartInCommitter {
                    continue;
                }
                let run = DropRun {
                    env: &env,
                    tree: DropTree::start(
                        &env,
                        case.shape(),
                        in_tree,
                        case != DropCase::StartInCommitter,
                    )
                    .await,
                    what: format!("{case:?}, in a tree {in_tree}, iteration {iteration}"),
                };
                let probes = DropProbes::arm(&run);
                let outcome = run.drop_rpc(case, &probes, dead_execution_manager_id).await;
                run.assert_outcome(case, &probes, outcome).await;
            }
        }
    }
    env.finish().await;
    Ok(())
}

/// Registers an execution manager and marks it dead in the database.
///
/// # Returns
///
/// The execution manager's ID.
///
/// # Panics
///
/// Panics if the registration or the update fails.
async fn register_dead_execution_manager(env: &TestEnv) -> ExecutionManagerId {
    let execution_manager_id = env
        .service
        .register_execution_manager(IpAddr::from([127, 0, 0, 2]), None)
        .await
        .expect("the execution manager registration should succeed")
        .0;
    let result = sqlx::query("UPDATE `execution_managers` SET `state` = ? WHERE `id` = ?;")
        .bind("Dead")
        .bind(execution_manager_id)
        .execute(&env.pool)
        .await
        .expect("the execution manager update should succeed");
    assert_eq!(result.rows_affected(), 1);
    execution_manager_id
}

/// Observers of the job `J` of a dropped RPC: its cache applies, and its submitted completions of
/// its regular tasks.
struct DropProbes {
    applied: ArmedHook,
    submitted_done: ArmedHook,
}

impl DropProbes {
    /// # Returns
    ///
    /// The probes of `run`'s job `J`.
    fn arm(run: &DropRun<'_>) -> Self {
        let root_id = run.tree.root;
        let job_id = run.tree.job;
        Self {
            applied: observe(
                run.env,
                HookPoint::CommitterAfterApplyJob,
                job_filter(root_id, job_id),
            ),
            submitted_done: observe(
                run.env,
                HookPoint::EventSubmitted,
                event_filter(root_id, Some(job_id), EventKindTag::RegularTasksDone),
            ),
        }
    }
}

/// How the tree of a dropped RPC ends.
struct DropOutcome {
    /// The final state of `J`.
    job_state: JobState,

    /// The final state of the root and the sibling, unless `J` is the root.
    rest_state: JobState,

    /// The output values of `J`'s output-producing tasks, if `J` stores outputs.
    outputs: Option<Vec<Vec<u8>>>,
}

impl DropOutcome {
    /// # Returns
    ///
    /// The outcome of a tree that succeeds, with `J`'s outputs `outputs`.
    const fn succeeded(outputs: Vec<Vec<u8>>) -> Self {
        Self {
            job_state: JobState::Succeeded,
            rest_state: JobState::Succeeded,
            outputs: Some(outputs),
        }
    }

    /// # Returns
    ///
    /// The outcome of a tree that ends cancelled.
    const fn cancelled() -> Self {
        Self {
            job_state: JobState::Cancelled,
            rest_state: JobState::Cancelled,
            outputs: None,
        }
    }
}

/// One dropped RPC in its tree.
struct DropRun<'env> {
    env: &'env TestEnv,
    tree: DropTree,
    what: String,
}

impl DropRun<'_> {
    /// Drops the RPC of `case` while its work is paused, releases the work, and drives the tree to
    /// its end, without repeating the dropped RPC.
    ///
    /// # Returns
    ///
    /// How the tree ended.
    async fn drop_rpc(
        &self,
        case: DropCase,
        probes: &DropProbes,
        dead_execution_manager_id: ExecutionManagerId,
    ) -> DropOutcome {
        match case {
            DropCase::AfterTcbFlip => {
                self.drop_last_report(HookPoint::PhaseAAfterTcbFlip, probes)
                    .await
            }
            DropCase::BeforeSubmit => {
                self.drop_last_report(HookPoint::PhaseABeforeSubmit, probes)
                    .await
            }
            DropCase::AfterTcbFlipRegistering => self.drop_registering_report(probes).await,
            DropCase::AfterTcbFlipWhileSiblingTaskReports => {
                self.drop_report_while_sibling_task_reports(probes).await
            }
            DropCase::AfterTcbFlipBothTasks => self.drop_both_reports(probes).await,
            DropCase::AfterTcbFlipRacingCancel => self.drop_report_racing_cancel(probes).await,
            DropCase::BeforeSuccessorLock => self.drop_report_before_successor_lock().await,
            DropCase::InCommitter(point) => self.drop_report_in_committer(point, probes).await,
            DropCase::FailureInCommitter => self.drop_failure_in_committer(probes).await,
            DropCase::CommitInCommitter => self.drop_commit_in_committer(probes).await,
            DropCase::CleanupInCommitter => self.drop_cleanup_in_committer(probes).await,
            DropCase::CancelInCommitter => self.drop_cancel_in_committer(probes).await,
            DropCase::StartInCommitter => self.drop_start_in_committer(probes).await,
            DropCase::CreateInstance => {
                self.drop_instance_creation(self.env.execution_manager_id, false)
                    .await
            }
            DropCase::CreateInstanceForDeadExecutionManager => {
                self.drop_instance_creation(dead_execution_manager_id, true)
                    .await
            }
        }
    }

    /// Asserts that the tree ended as `outcome` says, with `J`'s registered children succeeded,
    /// exactly the completions of `J`'s regular tasks that the case calls for, no lane entry left,
    /// and no fatal shutdown.
    async fn assert_outcome(&self, case: DropCase, probes: &DropProbes, outcome: DropOutcome) {
        let env = self.env;
        let what = self.what.as_str();
        let root_id = self.tree.root;
        let job_id = self.tree.job;
        let mut states = self.tree.states(outcome.job_state, outcome.rest_state);
        if case == DropCase::FailureInCommitter
            && let Some(sibling_id) = self.tree.sibling
        {
            states.insert(sibling_id, JobState::Cancelled);
        }
        for child_id in children_of(env, job_id).await {
            states.insert(child_id, JobState::Succeeded);
        }
        assert_tree(env, root_id, &states, false, what).await;
        assert_eq!(
            row_outputs(env, job_id).await,
            outcome.outputs.map(|values| job_outputs(&values)),
            "{what}: the job's stored outputs"
        );
        let mut errors = Vec::new();
        if case == DropCase::FailureInCommitter {
            errors.push((job_id, "dropped failure".to_owned()));
            if self.tree.sibling.is_some() {
                errors.push((
                    root_id,
                    format!("descendant job {job_id} failed: dropped failure"),
                ));
            }
            errors.sort_by_key(|(job_id, _)| job_id.get());
        }
        assert_eq!(tree_errors(env, root_id).await, errors, "{what}: errors");
        let num_done_submissions = usize::from(!matches!(
            case,
            DropCase::CleanupInCommitter
                | DropCase::CancelInCommitter
                | DropCase::FailureInCommitter
        ));
        assert_eq!(
            probes.submitted_done.hits(),
            num_done_submissions,
            "{what}: the job's regular completions"
        );
        assert_lanes_hold(env, &[]).await;
        assert_quiescent(env, root_id, outcome.rest_state, what).await;
    }

    /// Drops the last task's success report at `point` of phase A.
    async fn drop_last_report(&self, point: HookPoint, probes: &DropProbes) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        let paused = pause(env, point, job_filter(self.tree.root, job_id));
        let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
        drop_rpc_at(
            &paused,
            succeed(env, job_id, 0, task_instance_id, &value(1), Vec::new()),
        )
        .await;
        assert_eq!(
            probes.submitted_done.hits(),
            0,
            "{what}: submitted while paused"
        );
        assert_eq!(env.cached_job_state(job_id).await, Some(JobState::Running));
        paused.release();
        reached(&probes.applied, 1).await;
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(1)])
    }

    /// Drops a registering last report after its task's success.
    async fn drop_registering_report(&self, probes: &DropProbes) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        let paused = pause(
            env,
            HookPoint::PhaseAAfterTcbFlip,
            job_filter(self.tree.root, job_id),
        );
        let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
        drop_rpc_at(
            &paused,
            succeed(
                env,
                job_id,
                0,
                task_instance_id,
                &value(1),
                vec![child(ONE_TASK); NUM_CHILDREN],
            ),
        )
        .await;
        assert_eq!(children_of(env, job_id).await, Vec::new());
        paused.release();
        reached(&probes.applied, 1).await;
        assert_eq!(
            env.cached_job_state(job_id).await,
            Some(JobState::BlockingOnChildren)
        );
        let child_ids = children_of(env, job_id).await;
        assert_eq!(child_ids.len(), NUM_CHILDREN, "{what}: the children");
        let ready: Vec<_> = child_ids
            .iter()
            .map(|child_id| (*child_id, TaskId::Index(0)))
            .collect();
        drain_lane(env, Lane::Task, &ready).await;
        for child_id in &child_ids {
            assert_eq!(
                run_task(env, *child_id, 0, &value(2)).await,
                JobState::Succeeded
            );
        }
        assert_eq!(probes.applied.hits(), 2, "{what}: the parent's applies");
        assert_eq!(children_of(env, job_id).await, child_ids);
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(1)])
    }

    /// Drops the report of task 0 of a two-task job after its task's success, and reports task 1
    /// in full meanwhile.
    async fn drop_report_while_sibling_task_reports(&self, probes: &DropProbes) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        let paused = pause(
            env,
            HookPoint::PhaseAAfterTcbFlip,
            job_filter(self.tree.root, job_id),
        );
        let first = instance(env, job_id, TaskId::Index(0)).await;
        let second = instance(env, job_id, TaskId::Index(1)).await;
        drop_rpc_at(
            &paused,
            succeed(env, job_id, 0, first, &value(1), Vec::new()),
        )
        .await;
        assert_job_state(
            &succeed(env, job_id, 1, second, &value(2), Vec::new()).await,
            JobState::Running,
            &format!("{what}: the second task's report"),
        );
        assert_eq!(
            probes.submitted_done.hits(),
            0,
            "{what}: submitted while paused"
        );
        paused.release();
        reached(&probes.applied, 1).await;
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(1), value(2)])
    }

    /// Drops the reports of both tasks of a two-task job after their tasks' successes.
    async fn drop_both_reports(&self, probes: &DropProbes) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        let paused = env.hooks.arm(
            HookPoint::PhaseAAfterTcbFlip,
            job_filter(self.tree.root, job_id),
            HookAction::Pause,
            HookCount::Times(2),
        );
        let first = instance(env, job_id, TaskId::Index(0)).await;
        let second = instance(env, job_id, TaskId::Index(1)).await;
        let reports = [
            tokio::spawn(succeed(env, job_id, 0, first, &value(1), Vec::new())),
            tokio::spawn(succeed(env, job_id, 1, second, &value(2), Vec::new())),
        ];
        reached(&paused, 2).await;
        for report in reports {
            report.abort();
            assert!(
                report
                    .await
                    .expect_err("the dropped RPC should not complete")
                    .is_cancelled()
            );
        }
        paused.release();
        paused.release();
        reached(&probes.applied, 1).await;
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(1), value(2)])
    }

    /// Drops the last report after its task's success, holding the job's guard, while a cancel of
    /// the root commits and waits to apply.
    async fn drop_report_racing_cancel(&self, probes: &DropProbes) -> DropOutcome {
        let (env, what, root_id, job_id) =
            (self.env, self.what.as_str(), self.tree.root, self.tree.job);
        let paused = pause(
            env,
            HookPoint::PhaseAAfterTcbFlip,
            job_filter(root_id, job_id),
        );
        let cancel_committed = observe(
            env,
            HookPoint::CommitterAfterCommit,
            event_filter(root_id, None, EventKindTag::CancelRoot),
        );
        let late_round = observe(
            env,
            HookPoint::CommitterBeforePlan,
            event_filter(root_id, Some(job_id), EventKindTag::RegularTasksDone),
        );
        let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
        drop_rpc_at(
            &paused,
            succeed(env, job_id, 0, task_instance_id, &value(1), Vec::new()),
        )
        .await;
        let cancel = tokio::spawn(cancel(env, root_id));
        reached(&cancel_committed, 1).await;
        assert!(!cancel.is_finished(), "{what}: the cancel waits to apply");
        paused.release();
        let root_target = if self.tree.sibling.is_some() {
            JobState::Cancelled
        } else {
            JobState::CleanupReady
        };
        assert_job_state(
            &cancel.await.expect("the cancel should not panic"),
            root_target,
            &format!("{what}: the cancel"),
        );
        reached(&probes.submitted_done, 1).await;
        reached(&late_round, 1).await;
        assert_eq!(cancel_committed.hits(), 1);
        assert_eq!(probes.applied.hits(), 1, "{what}: the job's applies");
        self.run_cleanups().await;
        DropOutcome::cancelled()
    }

    /// Drops the report of the first task of a two-task chain before its successor is locked.
    async fn drop_report_before_successor_lock(&self) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        let paused = pause(
            env,
            HookPoint::PhaseABeforeSuccessorLock,
            job_filter(self.tree.root, job_id),
        );
        let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
        drop_rpc_at(
            &paused,
            succeed(env, job_id, 0, task_instance_id, &value(1), Vec::new()),
        )
        .await;
        paused.release();
        drain_lane(env, Lane::Task, &[(job_id, TaskId::Index(1))]).await;
        assert_eq!(
            run_task(env, job_id, 1, &value(2)).await,
            JobState::Succeeded
        );
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(2)])
    }

    /// Drops the last report while its event is paused in the committer at `point`.
    async fn drop_report_in_committer(&self, point: HookPoint, probes: &DropProbes) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        let paused = self.pause_in_committer(EventKindTag::RegularTasksDone, point);
        let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
        drop_rpc_at(
            &paused,
            succeed(env, job_id, 0, task_instance_id, &value(1), Vec::new()),
        )
        .await;
        paused.release();
        reached(&probes.applied, 1).await;
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(1)])
    }

    /// Drops a terminal failure while its event is paused in the committer.
    async fn drop_failure_in_committer(&self, probes: &DropProbes) -> DropOutcome {
        let (env, job_id) = (self.env, self.tree.job);
        let paused = self.pause_in_committer(
            EventKindTag::TerminalFailure,
            HookPoint::CommitterBeforeDbTransaction,
        );
        let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
        drop_rpc_at(
            &paused,
            fail(
                env,
                job_id,
                TaskId::Index(0),
                task_instance_id,
                "dropped failure",
            ),
        )
        .await;
        paused.release();
        reached(&probes.applied, 1).await;
        if let Some(sibling_id) = self.tree.sibling {
            drain_lane(env, Lane::Cleanup, &[(sibling_id, TaskId::Cleanup)]).await;
            assert_eq!(run_cleanup(env, sibling_id).await, JobState::Cancelled);
        }
        DropOutcome {
            job_state: JobState::Failed,
            rest_state: JobState::Failed,
            outputs: None,
        }
    }

    /// Drops a commit success while its event is paused in the committer.
    async fn drop_commit_in_committer(&self, probes: &DropProbes) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        assert_eq!(
            run_task(env, job_id, 0, &value(1)).await,
            JobState::CommitReady
        );
        drain_lane(env, Lane::Commit, &[(job_id, TaskId::Commit)]).await;
        let paused = self.pause_in_committer(
            EventKindTag::CommitDone,
            HookPoint::CommitterBeforeDbTransaction,
        );
        let task_instance_id = instance(env, job_id, TaskId::Commit).await;
        drop_rpc_at(&paused, succeed_commit(env, job_id, task_instance_id)).await;
        paused.release();
        reached(&probes.applied, 2).await;
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(1)])
    }

    /// Drops a cleanup success while its event is paused in the committer.
    async fn drop_cleanup_in_committer(&self, probes: &DropProbes) -> DropOutcome {
        let (env, what, root_id, job_id) =
            (self.env, self.what.as_str(), self.tree.root, self.tree.job);
        let root_target = if self.tree.sibling.is_some() {
            JobState::Cancelled
        } else {
            JobState::CleanupReady
        };
        assert_job_state(&cancel(env, root_id).await, root_target, what);
        let mut cleanups = vec![(job_id, TaskId::Cleanup)];
        cleanups.extend(
            self.tree
                .sibling
                .map(|sibling_id| (sibling_id, TaskId::Cleanup)),
        );
        drain_lane(env, Lane::Cleanup, &cleanups).await;
        let paused = self.pause_in_committer(
            EventKindTag::CleanupDone,
            HookPoint::CommitterBeforeDbTransaction,
        );
        let task_instance_id = instance(env, job_id, TaskId::Cleanup).await;
        drop_rpc_at(&paused, succeed_cleanup(env, job_id, task_instance_id)).await;
        paused.release();
        reached(&probes.applied, 2).await;
        if let Some(sibling_id) = self.tree.sibling {
            assert_eq!(run_cleanup(env, sibling_id).await, JobState::Cancelled);
        }
        DropOutcome::cancelled()
    }

    /// Drops a root cancel while its event is paused in the committer.
    async fn drop_cancel_in_committer(&self, probes: &DropProbes) -> DropOutcome {
        let env = self.env;
        let paused = self.pause_in_committer(
            EventKindTag::CancelRoot,
            HookPoint::CommitterBeforeDbTransaction,
        );
        drop_rpc_at(&paused, cancel(env, self.tree.root)).await;
        paused.release();
        reached(&probes.applied, 1).await;
        if let Some(sibling_id) = self.tree.sibling {
            drain_lane(env, Lane::Cleanup, &[(sibling_id, TaskId::Cleanup)]).await;
            assert_eq!(run_cleanup(env, sibling_id).await, JobState::Cancelled);
        }
        DropOutcome::cancelled()
    }

    /// Drops a root start while its event is paused in the committer.
    async fn drop_start_in_committer(&self, probes: &DropProbes) -> DropOutcome {
        let (env, job_id) = (self.env, self.tree.job);
        let paused =
            self.pause_in_committer(EventKindTag::Start, HookPoint::CommitterBeforeDbTransaction);
        drop_rpc_at(&paused, start(env, job_id)).await;
        paused.release();
        reached(&probes.applied, 1).await;
        drain_lane(env, Lane::Task, &[(job_id, TaskId::Index(0))]).await;
        assert_eq!(
            run_task(env, job_id, 0, &value(1)).await,
            JobState::Succeeded
        );
        DropOutcome::succeeded(vec![value(1)])
    }

    /// Drops an instance creation for `execution_manager_id` before the instance is sent to the
    /// task instance pool. If `is_dead`, the pool finds the execution manager dead, which proves
    /// that the dropped creation still sent the instance: the pool removes it and republishes its
    /// task.
    async fn drop_instance_creation(
        &self,
        execution_manager_id: ExecutionManagerId,
        is_dead: bool,
    ) -> DropOutcome {
        let (env, what, job_id) = (self.env, self.what.as_str(), self.tree.job);
        let paused = pause(
            env,
            HookPoint::CreateInstanceBeforePoolSend,
            job_filter(self.tree.root, job_id),
        );
        drop_rpc_at(
            &paused,
            create_instance(env, job_id, TaskId::Index(0), execution_manager_id),
        )
        .await;
        let dropped_instance_id = paused
            .last_context()
            .and_then(|context| context.task_instance_id)
            .expect("the hook context should carry the instance ID");
        paused.release();
        let task_instance_id = if is_dead {
            drain_lane(env, Lane::Task, &[(job_id, TaskId::Index(0))]).await;
            assert_stale(
                &succeed(env, job_id, 0, dropped_instance_id, &value(1), Vec::new()).await,
                &StaleStateError::InvalidTaskInstanceId,
                &format!("{what}: the report of the removed instance"),
            );
            instance(env, job_id, TaskId::Index(0)).await
        } else {
            dropped_instance_id
        };
        assert_job_state(
            &succeed(env, job_id, 0, task_instance_id, &value(1), Vec::new()).await,
            JobState::Succeeded,
            &format!("{what}: the report"),
        );
        self.tree.finish_sibling(env, what).await;
        DropOutcome::succeeded(vec![value(1)])
    }

    /// # Returns
    ///
    /// A pause of the first committer round of the tree that holds an event of `kind`, at
    /// `point`.
    fn pause_in_committer(&self, kind: EventKindTag, point: HookPoint) -> ArmedHook {
        pause(self.env, point, event_filter(self.tree.root, None, kind))
    }

    /// Drains and runs the cleanups of `J` and of the sibling, if any, after a cancel.
    async fn run_cleanups(&self) {
        let mut cleanups = vec![(self.tree.job, TaskId::Cleanup)];
        cleanups.extend(
            self.tree
                .sibling
                .map(|sibling_id| (sibling_id, TaskId::Cleanup)),
        );
        drain_lane(self.env, Lane::Cleanup, &cleanups).await;
        for (job_id, _) in cleanups {
            assert_eq!(run_cleanup(self.env, job_id).await, JobState::Cancelled);
        }
    }
}

/// A success report that breaks a limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LimitCase {
    /// One child more than a job may have.
    TooManyChildren,

    /// A child over the database's statement payload.
    ChildTooLarge,

    /// Children over the per-report byte limit.
    ChildrenTooLarge,

    /// A child that does not decode.
    InvalidChild,

    /// Outputs over the per-report limit.
    OutputsTooLarge,

    /// Children whose insert the database rejects for their payload.
    PayloadRejected,
}

/// The report of a [`LimitCase`]: its output, its children, and the failure message it causes,
/// or [`None`] if the message is checked by its prefix only.
struct LimitReport {
    output: Vec<u8>,
    children: Vec<ChildJobSubmission>,
    message: Option<String>,
    message_prefix: String,
}

impl LimitReport {
    /// # Returns
    ///
    /// The report of `case`, against the database's statement payload limit `max_child_size`.
    fn new(case: LimitCase, max_child_size: u64) -> Self {
        let max_child_size = usize::try_from(max_child_size).expect("the limit should fit");
        let (output, children, message) = match case {
            LimitCase::TooManyChildren => (
                value(1),
                vec![child(ONE_TASK); MAX_CHILDREN_PER_JOB + 1],
                format!(
                    "{} child jobs exceed the limit of {MAX_CHILDREN_PER_JOB} per report",
                    MAX_CHILDREN_PER_JOB + 1
                ),
            ),
            LimitCase::ChildTooLarge => (
                value(1),
                vec![child(ONE_TASK), opaque_child(max_child_size + 1)],
                format!(
                    "child job 1 of {} bytes exceeds the limit of {max_child_size} bytes",
                    max_child_size + 1
                ),
            ),
            LimitCase::ChildrenTooLarge => {
                let child_size = max_child_size - 1024;
                let num_children = usize::try_from(MAX_REPORT_CHILD_BYTES)
                    .expect("the limit should fit")
                    / child_size
                    + 1;
                (
                    value(1),
                    vec![opaque_child(child_size); num_children],
                    format!(
                        "child jobs of {} bytes exceed the limit of {MAX_REPORT_CHILD_BYTES} \
                         bytes per report",
                        child_size * num_children
                    ),
                )
            }
            LimitCase::InvalidChild => (
                value(1),
                vec![child(ONE_TASK), opaque_child(16)],
                String::new(),
            ),
            LimitCase::OutputsTooLarge => {
                let output = vec![
                    0x5a;
                    usize::try_from(MAX_REPORT_OUTPUT_BYTES)
                        .expect("the limit should fit")
                ];
                let size: usize = job_outputs(std::slice::from_ref(&output))
                    .iter()
                    .map(Vec::len)
                    .sum();
                (
                    output,
                    Vec::new(),
                    format!(
                        "task outputs of {size} bytes exceed the limit of \
                         {MAX_REPORT_OUTPUT_BYTES} bytes"
                    ),
                )
            }
            LimitCase::PayloadRejected => (
                value(1),
                vec![child(ONE_TASK); NUM_CHILDREN],
                "child payload rejected by the database".to_owned(),
            ),
        };
        if case == LimitCase::InvalidChild {
            return Self {
                output,
                children,
                message: None,
                message_prefix: "child job 1 is invalid: ".to_owned(),
            };
        }
        Self {
            output,
            children,
            message_prefix: message.clone(),
            message: Some(message),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_report_over_a_limit_fails_its_job_and_tree_but_not_the_service() -> anyhow::Result<()> {
    let env = start_env().await;
    let max_child_size = env.service.tree_env().db.max_statement_payload();
    let cases = [
        LimitCase::TooManyChildren,
        LimitCase::ChildTooLarge,
        LimitCase::ChildrenTooLarge,
        LimitCase::InvalidChild,
        LimitCase::OutputsTooLarge,
        LimitCase::PayloadRejected,
    ];
    let reports: Vec<(LimitCase, LimitReport)> = cases
        .into_iter()
        .map(|case| (case, LimitReport::new(case, max_child_size)))
        .collect();
    for iteration in 0..sequential_iterations() {
        for (case, report) in &reports {
            for max_num_retry in [0, 1] {
                for in_tree in [false, true] {
                    let what = format!(
                        "{case:?}, retries {max_num_retry}, in a tree {in_tree}, iteration \
                         {iteration}"
                    );
                    assert_limit_fails_the_job(&env, *case, report, max_num_retry, in_tree, &what)
                        .await;
                }
            }
        }
    }
    env.finish().await;
    Ok(())
}

/// Reports `report` for the only task of a job `J` with `max_num_retry` retries, alone or as a leaf
/// of `R → {J, S}`, once per allowed attempt, and asserts that `J` and its tree fail, that no
/// child is stored, and that the service still serves another tree.
async fn assert_limit_fails_the_job(
    env: &TestEnv,
    case: LimitCase,
    report: &LimitReport,
    max_num_retry: u32,
    in_tree: bool,
    what: &str,
) {
    let tree = DropTree::start(
        env,
        JobShape {
            max_num_retry,
            ..ONE_TASK
        },
        in_tree,
        true,
    )
    .await;
    let root_id = tree.root;
    let job_id = tree.job;
    let bystander_id = start_root(env, ONE_TASK).await;
    let fail_db = (case == LimitCase::PayloadRejected).then(|| {
        env.hooks.arm(
            HookPoint::CommitterBeforeDbTransaction,
            event_filter(root_id, Some(job_id), EventKindTag::RegularTasksDone),
            HookAction::FailDb {
                deterministic_payload: true,
            },
            HookCount::Once,
        )
    });
    let num_attempts = if case == LimitCase::PayloadRejected {
        1
    } else {
        max_num_retry + 1
    };
    for attempt in 0..num_attempts {
        if attempt > 0 {
            drain_lane(env, Lane::Task, &[(job_id, TaskId::Index(0))]).await;
        }
        let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
        let expected = if attempt + 1 == num_attempts {
            JobState::Failed
        } else {
            JobState::Running
        };
        assert_job_state(
            &succeed(
                env,
                job_id,
                0,
                task_instance_id,
                &report.output,
                report.children.clone(),
            )
            .await,
            expected,
            &format!("{what}: attempt {attempt}"),
        );
        assert_eq!(
            children_of(env, job_id).await,
            Vec::new(),
            "{what}: children after attempt {attempt}"
        );
    }
    if let Some(fail_db) = fail_db {
        assert_eq!(fail_db.hits(), 1, "{what}: the rejected insert");
    }

    assert_failed_for_limit(env, &tree, report, bystander_id, what).await;
}

/// Asserts that a report over a limit failed the job `J` of `tree` with the report's message,
/// failed the root with the propagated message, and aborted the sibling, whose cleanup then runs;
/// that nothing of the report is stored; and that the bystander tree `bystander_id` still runs.
async fn assert_failed_for_limit(
    env: &TestEnv,
    tree: &DropTree,
    report: &LimitReport,
    bystander_id: JobId,
    what: &str,
) {
    let root_id = tree.root;
    let job_id = tree.job;
    let errors = tree_errors(env, root_id).await;
    let job_error = errors
        .iter()
        .find(|(error_job_id, _)| *error_job_id == job_id)
        .map(|(_, message)| message.clone())
        .expect("the failed job should have an error message");
    match &report.message {
        Some(message) => assert_eq!(&job_error, message, "{what}: the job's error"),
        None => assert!(
            job_error.starts_with(&report.message_prefix)
                && job_error.len() > report.message_prefix.len(),
            "{what}: unexpected error `{job_error}`"
        ),
    }
    let mut expected_errors = vec![(job_id, job_error.clone())];
    if tree.sibling.is_some() {
        expected_errors.push((
            root_id,
            format!("descendant job {job_id} failed: {job_error}"),
        ));
    }
    expected_errors.sort_by_key(|(job_id, _)| job_id.get());
    assert_eq!(errors, expected_errors, "{what}: the tree's errors");

    let mut states = tree.states(JobState::Failed, JobState::Failed);
    if let Some(sibling_id) = tree.sibling {
        states.insert(sibling_id, JobState::CleanupReady);
        assert_tree(env, root_id, &states, false, what).await;
        drain_lane(env, Lane::Cleanup, &[(sibling_id, TaskId::Cleanup)]).await;
        assert_eq!(run_cleanup(env, sibling_id).await, JobState::Cancelled);
        states.insert(sibling_id, JobState::Cancelled);
    }
    assert_tree(env, root_id, &states, false, what).await;
    assert_eq!(
        row_outputs(env, job_id).await,
        None,
        "{what}: outputs stored"
    );
    assert_eq!(
        tree_rows(&env.pool, root_id).await.len(),
        states.len(),
        "{what}: rows of the tree"
    );
    assert_eq!(
        run_task(env, bystander_id, 0, &value(7)).await,
        JobState::Succeeded,
        "{what}: the bystander"
    );
    assert_lanes_hold(env, &[]).await;
    assert_quiescent(env, root_id, JobState::Failed, what).await;
    assert_not_fatal(env, what);
}

/// The order of two reports whose children together exceed the per-job child bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReservationRace {
    /// Both reports run at once.
    Free,

    /// The first report is held after its task's success, holding its job's guard, while the
    /// second report is rejected and its failure commits.
    FirstHeldAfterTcbFlip,

    /// The first report completes before the second starts.
    Sequential,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn reports_racing_over_the_job_child_limit_fail_one_task_only() -> anyhow::Result<()> {
    let env = start_env().await;
    let max_child_size = env.service.tree_env().db.max_statement_payload();
    let child_size_target = usize::try_from(MAX_JOB_CHILD_BYTES).expect("the limit should fit") / 5;
    let large = large_child(child_size_target);
    let child_size = large.size();
    assert!(
        child_size <= max_child_size
            && 3 * child_size <= MAX_REPORT_CHILD_BYTES
            && 6 * child_size > MAX_JOB_CHILD_BYTES,
        "a child of {child_size} bytes does not fit the test"
    );
    for iteration in 0..race_iterations() {
        for race in [
            ReservationRace::Free,
            ReservationRace::FirstHeldAfterTcbFlip,
            ReservationRace::Sequential,
        ] {
            for max_num_retry in [0, 1] {
                let what = format!("{race:?}, retries {max_num_retry}, iteration {iteration}");
                assert_reservation_race(&env, &large, race, max_num_retry, &what).await;
            }
        }
    }
    env.finish().await;
    Ok(())
}

/// Races two reports of the two tasks of one job, each with three children of `large`, whose
/// union exceeds [`MAX_JOB_CHILD_BYTES`], and asserts that exactly one task fails for it.
async fn assert_reservation_race(
    env: &TestEnv,
    large: &ChildJobSubmission,
    race: ReservationRace,
    max_num_retry: u32,
    what: &str,
) {
    let job_id = start_root(
        env,
        JobShape {
            num_tasks: 2,
            max_num_retry,
            ..ONE_TASK
        },
    )
    .await;
    let (winner, results) = race_reports(env, job_id, large, race, max_num_retry, what).await;
    let loser = 1 - winner;
    let loser_state = if max_num_retry == 0 {
        JobState::Failed
    } else {
        JobState::Running
    };
    assert_job_state(&results[loser], loser_state, &format!("{what}: the loser"));
    match (&results[winner], race) {
        (Ok(JobState::Running), _) => {}
        (result, ReservationRace::Free) if max_num_retry == 0 => assert_stale(
            result,
            &StaleStateError::JobNoLongerRunning,
            &format!("{what}: the winner"),
        ),
        (other, _) => panic!("{what}: unexpected winner result {other:?}"),
    }
    assert_eq!(children_of(env, job_id).await, Vec::new());

    if max_num_retry == 0 {
        let message = format!(
            "child jobs would bring the job to {} child jobs of {} bytes, over the limits of \
             {MAX_CHILDREN_PER_JOB} child jobs and {MAX_JOB_CHILD_BYTES} bytes per job",
            2 * NUM_CHILDREN,
            2 * NUM_CHILDREN as u64 * large.size()
        );
        assert_eq!(
            tree_errors(env, job_id).await,
            vec![(job_id, message)],
            "{what}: the job's error"
        );
        assert_tree(
            env,
            job_id,
            &HashMap::from([(job_id, JobState::Failed)]),
            false,
            what,
        )
        .await;
        assert_eq!(row_outputs(env, job_id).await, None);
        assert_lanes_hold(env, &[]).await;
        assert_quiescent(env, job_id, JobState::Failed, what).await;
        return;
    }
    if race != ReservationRace::Free {
        drain_lane(env, Lane::Task, &[(job_id, TaskId::Index(loser))]).await;
    }
    finish_after_rejected_report(env, job_id, loser, what).await;
}

/// Races the reports of the two tasks of `job_id` as `race` says, each with three children of
/// `large`.
///
/// # Returns
///
/// A pair, containing:
///
/// * The index of the task whose children were reserved.
/// * The results of the reports of task 0 and task 1.
async fn race_reports(
    env: &TestEnv,
    job_id: JobId,
    large: &ChildJobSubmission,
    race: ReservationRace,
    max_num_retry: u32,
    what: &str,
) -> (usize, [JobResult; 2]) {
    let instances = [
        instance(env, job_id, TaskId::Index(0)).await,
        instance(env, job_id, TaskId::Index(1)).await,
    ];
    let report = |task_index: TaskIndex| {
        succeed(
            env,
            job_id,
            task_index,
            instances[task_index],
            &value(u8::try_from(task_index).expect("small index")),
            vec![large.clone(); NUM_CHILDREN],
        )
    };
    match race {
        ReservationRace::Free => {
            let first = tokio::spawn(report(0));
            let second = tokio::spawn(report(1));
            let results = [
                first.await.expect("the first report should not panic"),
                second.await.expect("the second report should not panic"),
            ];
            let winner = if max_num_retry == 0 {
                let losers: Vec<usize> = (0..2)
                    .filter(|index| matches!(results[*index], Ok(JobState::Failed)))
                    .collect();
                assert_eq!(losers.len(), 1, "{what}: one loser, got {results:?}");
                1 - losers[0]
            } else {
                let republished = poll_lane(env, Lane::Task, 1, Duration::from_secs(10)).await;
                assert_eq!(republished.len(), 1, "{what}: the retried task");
                let (republished_job_id, TaskId::Index(loser)) = republished[0] else {
                    panic!("{what}: unexpected entry {republished:?}");
                };
                assert_eq!(republished_job_id, job_id);
                1 - loser
            };
            (winner, results)
        }
        ReservationRace::FirstHeldAfterTcbFlip => {
            let paused = pause(env, HookPoint::PhaseAAfterTcbFlip, HookFilter::job(job_id));
            let failure_committed = observe(
                env,
                HookPoint::CommitterAfterCommit,
                event_filter(job_id, Some(job_id), EventKindTag::TerminalFailure),
            );
            let first = tokio::spawn(report(0));
            reached(&paused, 1).await;
            let second = tokio::spawn(report(1));
            let second = if max_num_retry == 0 {
                reached(&failure_committed, 1).await;
                assert!(!second.is_finished(), "{what}: the failure waits to apply");
                assert_eq!(env.row_state(job_id).await, JobState::Failed);
                paused.release();
                second.await.expect("the second report should not panic")
            } else {
                let second = second.await.expect("the second report should not panic");
                paused.release();
                second
            };
            let first = first.await.expect("the first report should not panic");
            assert_eq!(failure_committed.hits(), usize::from(max_num_retry == 0));
            (0, [first, second])
        }
        ReservationRace::Sequential => {
            let first = report(0).await;
            let second = report(1).await;
            (0, [first, second])
        }
    }
}

/// Reruns the rejected task `loser` of `job_id` without children, and asserts that the job then
/// registers exactly the other task's children and succeeds with both tasks' outputs.
async fn finish_after_rejected_report(env: &TestEnv, job_id: JobId, loser: usize, what: &str) {
    let loser_instance = instance(env, job_id, TaskId::Index(loser)).await;
    assert_job_state(
        &succeed(
            env,
            job_id,
            loser,
            loser_instance,
            &value(u8::try_from(loser).expect("small index")),
            Vec::new(),
        )
        .await,
        JobState::BlockingOnChildren,
        &format!("{what}: the retried task"),
    );
    let child_ids = children_of(env, job_id).await;
    assert_eq!(
        child_ids.len(),
        NUM_CHILDREN,
        "{what}: the winner's children"
    );
    let ready: Vec<_> = child_ids
        .iter()
        .map(|child_id| (*child_id, TaskId::Index(0)))
        .collect();
    drain_lane(env, Lane::Task, &ready).await;
    for child_id in &child_ids {
        assert_eq!(
            run_task(env, *child_id, 0, &value(3)).await,
            JobState::Succeeded
        );
    }
    let mut done = HashMap::from([(job_id, JobState::Succeeded)]);
    for child_id in &child_ids {
        done.insert(*child_id, JobState::Succeeded);
    }
    assert_tree(env, job_id, &done, false, what).await;
    assert_eq!(tree_errors(env, job_id).await, Vec::new());
    assert_eq!(
        row_outputs(env, job_id).await,
        Some(job_outputs(&[value(0), value(1)])),
        "{what}: the job's outputs"
    );
    assert_lanes_hold(env, &[]).await;
    assert_quiescent(env, job_id, JobState::Succeeded, what).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MariaDB"]
async fn a_report_beyond_the_maximum_depth_fails_the_whole_chain() -> anyhow::Result<()> {
    let env = start_env().await;
    let with_cleanup = JobShape {
        with_cleanup: true,
        ..ONE_TASK
    };
    for iteration in 0..sequential_iterations() {
        let what = format!("iteration {iteration}");
        let root_id = start_root(&env, ONE_TASK).await;
        let first_children = register_children(&env, root_id, &[ONE_TASK, with_cleanup]).await;
        let sibling_id = first_children[1];
        let mut chain = vec![root_id, first_children[0]];
        for _ in 2..=MAX_DEPTH {
            let parent_id = *chain.last().expect("the chain should not be empty");
            chain.push(register_children(&env, parent_id, &[ONE_TASK]).await[0]);
        }
        let deepest_id = *chain.last().expect("the chain should not be empty");
        assert_eq!(
            run_task_with_children(&env, deepest_id, vec![child(ONE_TASK)]).await,
            JobState::Failed,
            "{what}: the report beyond the maximum depth"
        );
        assert_eq!(children_of(&env, deepest_id).await, Vec::new());
        let message = format!(
            "child jobs at depth {} exceed the maximum depth {MAX_DEPTH}",
            MAX_DEPTH + 1
        );
        let propagated = format!("descendant job {deepest_id} failed: {message}");
        let mut errors: Vec<(JobId, String)> = chain
            .iter()
            .map(|job_id| {
                let error = if *job_id == deepest_id {
                    message.clone()
                } else {
                    propagated.clone()
                };
                (*job_id, error)
            })
            .collect();
        errors.sort_by_key(|(job_id, _)| job_id.get());
        assert_eq!(tree_errors(&env, root_id).await, errors, "{what}: errors");

        drain_lane(&env, Lane::Cleanup, &[(sibling_id, TaskId::Cleanup)]).await;
        assert_eq!(run_cleanup(&env, sibling_id).await, JobState::Cancelled);
        let mut states: HashMap<JobId, JobState> = chain
            .iter()
            .map(|job_id| (*job_id, JobState::Failed))
            .collect();
        states.insert(sibling_id, JobState::Cancelled);
        assert_tree(&env, root_id, &states, false, &what).await;
        assert_lanes_hold(&env, &[]).await;
        assert_quiescent(&env, root_id, JobState::Failed, &what).await;
    }
    env.finish().await;
    Ok(())
}

/// Runs a new instance of the only task of `job_id` to success with `children`.
///
/// # Returns
///
/// The job state after the report.
///
/// # Panics
///
/// Panics if the report fails.
async fn run_task_with_children(
    env: &TestEnv,
    job_id: JobId,
    children: Vec<ChildJobSubmission>,
) -> JobState {
    let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
    succeed(env, job_id, 0, task_instance_id, &value(8), children)
        .await
        .expect("the report should succeed")
}
