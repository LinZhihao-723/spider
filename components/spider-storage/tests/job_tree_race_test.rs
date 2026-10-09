//! Propagation, abort and root-only cancel on job trees, raced deterministically through the test
//! hooks (T1-T5, T7, T10), and the root-only rule of the external job RPCs.

use std::future::Future;

use spider_core::job::JobState;
use spider_core::task::DataTypeDescriptor;
use spider_core::task::ExecutionPolicy;
use spider_core::task::TaskDescriptor;
use spider_core::task::TaskState;
use spider_core::task::TdlContext;
use spider_core::task::TerminationTaskDescriptor;
use spider_core::task::ValueTypeDescriptor;
use spider_core::types::id::JobId;
use spider_core::types::id::TaskId;
use spider_core::types::io::TaskGraphInput;
use spider_core::types::io::TaskGraphInputBuilder;
use spider_core::types::io::TaskOutputsSerializer;
use spider_storage::cache::error::CacheError;
use spider_storage::cache::error::StaleStateError;
use spider_storage::db::DbError;
use spider_storage::job_submission::ChildJobSubmission;
use spider_storage::limits::MAX_CHILDREN_PER_JOB;
use spider_storage::limits::MAX_REPORT_OUTPUT_BYTES;
use spider_storage::state::StorageServerError;
use spider_storage::test_hooks::ArmedHook;
use spider_storage::test_hooks::EventKindTag;
use spider_storage::test_hooks::HookAction;
use spider_storage::test_hooks::HookCount;
use spider_storage::test_hooks::HookFilter;
use spider_storage::test_hooks::HookPoint;

use super::job_tree_infra::TestEnv;
use super::job_tree_infra::TestService;
use super::task_graph_builder::SubmittedTaskGraph;
use super::task_graph_builder::compress_job_inputs;
use super::task_graph_builder::compress_task_graph;

/// The result of a job RPC.
type JobResult = Result<JobState, StorageServerError>;

/// The shape of a test job.
#[derive(Clone, Copy)]
struct JobShape {
    num_tasks: usize,
    max_num_instances: u32,
    with_commit: bool,
    with_cleanup: bool,
}

const ONE_TASK: JobShape = JobShape {
    num_tasks: 1,
    max_num_instances: 1,
    with_commit: false,
    with_cleanup: false,
};

const WITH_COMMIT: JobShape = JobShape {
    with_commit: true,
    ..ONE_TASK
};

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

/// Builds a job of `shape` whose independent tasks each take and return one byte-typed value, and
/// that never retries a task.
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
        max_num_retry: 0,
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
/// The serialized outputs of a task whose only output is `value`.
///
/// # Panics
///
/// Panics if the serialization fails.
fn serialized_output(value: Vec<u8>) -> Vec<u8> {
    TaskOutputsSerializer::from_tuple(&(value,)).expect("output serialization should succeed")
}

/// Registers and starts a root job of `shape`.
///
/// # Returns
///
/// The job ID.
///
/// # Panics
///
/// Panics if the registration or the start fails.
async fn start_root(env: &TestEnv, shape: JobShape) -> JobId {
    let (graph, inputs) = build_job(shape);
    let job_id = env.register_job(&graph, &inputs).await;
    env.service
        .start_job(job_id)
        .await
        .expect("the job should start");
    job_id
}

/// # Returns
///
/// The ID of a new instance of `task_id` of `job_id`.
///
/// # Panics
///
/// Panics if the instance creation fails.
async fn instance(env: &TestEnv, job_id: JobId, task_id: TaskId) -> u64 {
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
/// the output `[task_index; 4]` and `children`.
fn succeed(
    env: &TestEnv,
    job_id: JobId,
    task_index: usize,
    task_instance_id: u64,
    children: Vec<ChildJobSubmission>,
) -> impl Future<Output = JobResult> + Send + 'static {
    let service = env.service.clone();
    async move {
        service
            .succeed_task_instance(
                service.session_id(),
                job_id,
                task_instance_id,
                task_index,
                serialized_output(vec![u8::try_from(task_index).expect("small index"); 4]),
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
    task_index: usize,
    task_instance_id: u64,
    message: &str,
) -> impl Future<Output = JobResult> + Send + 'static {
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
/// A future that cancels `job_id`.
fn cancel(env: &TestEnv, job_id: JobId) -> impl Future<Output = JobResult> + Send + 'static {
    let service: TestService = env.service.clone();
    async move { service.cancel_job(job_id).await }
}

/// # Returns
///
/// A future that reports the success of `task_instance_id` of `job_id`'s commit task.
fn succeed_commit(
    env: &TestEnv,
    job_id: JobId,
    task_instance_id: u64,
) -> impl Future<Output = JobResult> + Send + 'static {
    let service = env.service.clone();
    async move {
        service
            .succeed_commit_task_instance(service.session_id(), job_id, task_instance_id)
            .await
    }
}

/// Runs a new instance of the only task of `parent_id` to success with one child per shape.
///
/// # Returns
///
/// The IDs of the registered children, by ascending ID.
///
/// # Panics
///
/// Panics if the report fails or does not block the parent on its children.
async fn register_children(env: &TestEnv, parent_id: JobId, shapes: &[JobShape]) -> Vec<JobId> {
    let task_instance_id = instance(env, parent_id, TaskId::Index(0)).await;
    let children = shapes.iter().copied().map(child).collect();
    assert_eq!(
        succeed(env, parent_id, 0, task_instance_id, children)
            .await
            .expect("the registering report should succeed"),
        JobState::BlockingOnChildren
    );
    let child_ids = children_of(env, parent_id).await;
    assert_eq!(child_ids.len(), shapes.len());
    child_ids
}

/// Runs a new instance of the only task of `job_id` to success without children.
///
/// # Returns
///
/// The job state after the report.
///
/// # Panics
///
/// Panics if the report fails.
async fn run_only_task(env: &TestEnv, job_id: JobId) -> JobState {
    let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
    succeed(env, job_id, 0, task_instance_id, Vec::new())
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
    env.service
        .succeed_cleanup_task_instance(env.service.session_id(), job_id, task_instance_id)
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
/// The error message stored in `job_id`'s row, if any.
///
/// # Panics
///
/// Panics if the query fails.
async fn row_error(env: &TestEnv, job_id: JobId) -> Option<String> {
    sqlx::query_scalar::<_, Option<String>>("SELECT `error_message` FROM `jobs` WHERE `id` = ?;")
        .bind(job_id)
        .fetch_one(&env.pool)
        .await
        .expect("the error query should succeed")
}

/// # Returns
///
/// The serialized outputs stored in `job_id`'s row, if any.
///
/// # Panics
///
/// Panics if the query fails.
async fn row_outputs(env: &TestEnv, job_id: JobId) -> Option<Vec<u8>> {
    sqlx::query_scalar::<_, Option<Vec<u8>>>(
        "SELECT `serialized_job_outputs` FROM `jobs` WHERE `id` = ?;",
    )
    .bind(job_id)
    .fetch_one(&env.pool)
    .await
    .expect("the outputs query should succeed")
}

/// Asserts that every job of `expected` has its state both in the cache (when cached) and in its
/// row, and that the service has not shut down fatally.
async fn assert_states(env: &TestEnv, expected: &[(JobId, JobState)]) {
    for (job_id, state) in expected {
        if let Some(cached) = env.cached_job_state(*job_id).await {
            assert_eq!(cached, *state, "cached state of job {job_id:?}");
        }
        assert_eq!(
            env.row_state(*job_id).await,
            *state,
            "row state of job {job_id:?}"
        );
    }
    assert!(!env.is_fatal());
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
    assert_eq!(
        non_terminal, 0,
        "the tree of {root_id:?} should be quiescent"
    );
}

/// Asserts that `result` is the stale error `expected`.
fn assert_stale(result: &JobResult, expected: &StaleStateError, what: &str) {
    match result {
        Err(StorageServerError::Cache(CacheError::StaleState(error)))
            if error.to_string() == expected.to_string() => {}
        other => panic!("{what}: expected the stale error `{expected}`, got {other:?}"),
    }
}

/// # Returns
///
/// An observer of the cache applies of each job of `job_ids`, in the tree rooted at `root_id`.
fn observe_applies(env: &TestEnv, root_id: JobId, job_ids: &[JobId]) -> Vec<ArmedHook> {
    job_ids
        .iter()
        .map(|job_id| {
            env.hooks.arm(
                HookPoint::CommitterAfterApplyJob,
                HookFilter {
                    root_id: Some(root_id),
                    job_id: Some(*job_id),
                    event_kind: None,
                },
                HookAction::Observe,
                HookCount::Always,
            )
        })
        .collect()
}

/// # Returns
///
/// The filter of the event of `kind` that `job_id` submits in the tree rooted at `root_id`.
const fn event_filter(root_id: JobId, job_id: JobId, kind: EventKindTag) -> HookFilter {
    HookFilter {
        root_id: Some(root_id),
        job_id: Some(job_id),
        event_kind: Some(kind),
    }
}

/// Runs `first` and then `second` so that `second`'s event, matched by `second_event`, is submitted
/// while `first`'s round of the tree rooted at `root_id` is paused at `point`:
///
/// * At [`HookPoint::CommitterAfterReceive`], both events are planned in one round.
/// * At [`HookPoint::CommitterBeforeDbTransaction`], `second`'s event is planned in the round after
///   `first`'s.
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
    let pause = env.hooks.arm(
        point,
        HookFilter::root(root_id),
        HookAction::Pause,
        HookCount::Once,
    );
    let submitted = env.hooks.arm(
        HookPoint::EventSubmitted,
        second_event,
        HookAction::Observe,
        HookCount::Always,
    );
    let first = tokio::spawn(first);
    pause.reached(1).await;
    let second = tokio::spawn(second);
    submitted.reached(1).await;
    pause.release();
    let first = first.await.expect("the first RPC should not panic");
    let second = second.await.expect("the second RPC should not panic");
    assert_eq!(pause.hits(), 1);
    (first, second)
}

/// Starts a root whose two children are a leaf of `leaf_shape` and a one-task sibling.
///
/// # Returns
///
/// The IDs of the root, the leaf and the sibling.
async fn start_two_leaf_tree(env: &TestEnv, leaf_shape: JobShape) -> [JobId; 3] {
    let root_id = start_root(env, ONE_TASK).await;
    let children = register_children(env, root_id, &[leaf_shape, ONE_TASK]).await;
    [root_id, children[0], children[1]]
}

/// Starts a root `R` whose only child `A` blocks on two children: `J`, of two tasks, and `S`, of
/// one task.
///
/// # Returns
///
/// `[R, A, J, S]`.
async fn start_failing_tree(env: &TestEnv) -> [JobId; 4] {
    let root_id = start_root(env, ONE_TASK).await;
    let middle_id = register_children(env, root_id, &[ONE_TASK]).await[0];
    let leaves = register_children(
        env,
        middle_id,
        &[
            JobShape {
                num_tasks: 2,
                ..ONE_TASK
            },
            ONE_TASK,
        ],
    )
    .await;
    [root_id, middle_id, leaves[0], leaves[1]]
}

/// T1: two tasks of a leaf fail in one round and in consecutive rounds.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn two_tasks_failing_a_leaf_fail_its_tree_once() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    for _ in 0..race_iterations() {
        for point in [
            HookPoint::CommitterAfterReceive,
            HookPoint::CommitterBeforeDbTransaction,
        ] {
            let [root_id, middle_id, leaf_id, sibling_id] = start_failing_tree(&env).await;
            let first = instance(&env, leaf_id, TaskId::Index(0)).await;
            let second = instance(&env, leaf_id, TaskId::Index(1)).await;
            let applies =
                observe_applies(&env, root_id, &[root_id, middle_id, leaf_id, sibling_id]);

            let (first, second) = race(
                &env,
                root_id,
                point,
                fail(&env, leaf_id, 0, first, "first failure"),
                fail(&env, leaf_id, 1, second, "second failure"),
                event_filter(root_id, leaf_id, EventKindTag::TerminalFailure),
            )
            .await;

            assert_eq!(first?, JobState::Failed, "at {point:?}");
            assert_stale(
                &second,
                &StaleStateError::JobAlreadyTerminated(JobState::Failed),
                &format!("the second failure at {point:?}"),
            );
            assert_states(
                &env,
                &[
                    (root_id, JobState::Failed),
                    (middle_id, JobState::Failed),
                    (leaf_id, JobState::Failed),
                    (sibling_id, JobState::Cancelled),
                ],
            )
            .await;
            assert_eq!(
                row_error(&env, leaf_id).await.as_deref(),
                Some("first failure")
            );
            for ancestor_id in [root_id, middle_id] {
                let error = row_error(&env, ancestor_id)
                    .await
                    .expect("a failed ancestor should have an error message");
                assert!(
                    error.starts_with(&format!("descendant job {leaf_id} failed")),
                    "unexpected ancestor error `{error}`"
                );
            }
            for applied in &applies {
                assert_eq!(applied.hits(), 1, "every job should be updated once");
            }
            assert_quiescent(&env, root_id).await;
        }
    }
    env.finish().await;
    Ok(())
}

/// T2: a root cancel races a leaf's registration of children: in one round, with the registration
/// in flight, and after the cancel committed.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_root_cancel_racing_a_registration_leaves_no_running_child() -> anyhow::Result<()> {
    const NUM_CHILDREN: usize = 3;
    let env = TestEnv::start().await;
    let grandchildren = || vec![child(ONE_TASK); NUM_CHILDREN];
    for _ in 0..race_iterations() {
        let root_id = start_root(&env, ONE_TASK).await;
        let leaf_id = register_children(&env, root_id, &[ONE_TASK]).await[0];
        let task_instance_id = instance(&env, leaf_id, TaskId::Index(0)).await;
        let (report, cancelled) = race(
            &env,
            root_id,
            HookPoint::CommitterAfterReceive,
            succeed(&env, leaf_id, 0, task_instance_id, grandchildren()),
            cancel(&env, root_id),
            event_filter(root_id, root_id, EventKindTag::CancelRoot),
        )
        .await;
        assert_eq!(cancelled?, JobState::Cancelled);
        assert_stale(
            &report,
            &StaleStateError::JobAlreadyTerminated(JobState::Cancelled),
            "a registration in the cancel's round",
        );
        assert_eq!(children_of(&env, leaf_id).await, Vec::new());
        assert_states(
            &env,
            &[
                (root_id, JobState::Cancelled),
                (leaf_id, JobState::Cancelled),
            ],
        )
        .await;
        assert_quiescent(&env, root_id).await;

        let root_id = start_root(&env, ONE_TASK).await;
        let leaf_id = register_children(&env, root_id, &[ONE_TASK]).await[0];
        let task_instance_id = instance(&env, leaf_id, TaskId::Index(0)).await;
        let (report, cancelled) = race(
            &env,
            root_id,
            HookPoint::CommitterBeforeDbTransaction,
            succeed(&env, leaf_id, 0, task_instance_id, grandchildren()),
            cancel(&env, root_id),
            event_filter(root_id, root_id, EventKindTag::CancelRoot),
        )
        .await;
        assert_eq!(report?, JobState::BlockingOnChildren);
        assert_eq!(cancelled?, JobState::Cancelled);
        let grandchild_ids = children_of(&env, leaf_id).await;
        assert_eq!(grandchild_ids.len(), NUM_CHILDREN);
        let mut expected = vec![
            (root_id, JobState::Cancelled),
            (leaf_id, JobState::Cancelled),
        ];
        expected.extend(
            grandchild_ids
                .iter()
                .map(|grandchild_id| (*grandchild_id, JobState::Cancelled)),
        );
        assert_states(&env, &expected).await;
        assert_quiescent(&env, root_id).await;

        let root_id = start_root(&env, ONE_TASK).await;
        let leaf_id = register_children(&env, root_id, &[ONE_TASK]).await[0];
        let task_instance_id = instance(&env, leaf_id, TaskId::Index(0)).await;
        assert_eq!(cancel(&env, root_id).await?, JobState::Cancelled);
        let report = succeed(&env, leaf_id, 0, task_instance_id, grandchildren()).await;
        assert_stale(
            &report,
            &StaleStateError::JobNoLongerRunning,
            "a registration after the cancel",
        );
        assert_eq!(children_of(&env, leaf_id).await, Vec::new());
        assert_states(
            &env,
            &[
                (root_id, JobState::Cancelled),
                (leaf_id, JobState::Cancelled),
            ],
        )
        .await;
        assert_quiescent(&env, root_id).await;
    }
    env.finish().await;
    Ok(())
}

/// T3: a root cancel races a leaf failure in one round, and in consecutive rounds in both orders.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_root_cancel_racing_a_leaf_failure_is_planned_first() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    for _ in 0..race_iterations() {
        let [root_id, leaf_id, sibling_id] = start_two_leaf_tree(&env, ONE_TASK).await;
        let task_instance_id = instance(&env, leaf_id, TaskId::Index(0)).await;
        let (failure, cancelled) = race(
            &env,
            root_id,
            HookPoint::CommitterAfterReceive,
            fail(&env, leaf_id, 0, task_instance_id, "leaf failure"),
            cancel(&env, root_id),
            event_filter(root_id, root_id, EventKindTag::CancelRoot),
        )
        .await;
        assert_eq!(cancelled?, JobState::Cancelled);
        assert_stale(
            &failure,
            &StaleStateError::JobAlreadyTerminated(JobState::Cancelled),
            "a failure in the cancel's round",
        );
        assert_eq!(row_error(&env, leaf_id).await, None);
        assert_states(
            &env,
            &[
                (root_id, JobState::Cancelled),
                (leaf_id, JobState::Cancelled),
                (sibling_id, JobState::Cancelled),
            ],
        )
        .await;
        assert_quiescent(&env, root_id).await;

        let [root_id, leaf_id, sibling_id] = start_two_leaf_tree(&env, ONE_TASK).await;
        let task_instance_id = instance(&env, leaf_id, TaskId::Index(0)).await;
        let (failure, cancelled) = race(
            &env,
            root_id,
            HookPoint::CommitterBeforeDbTransaction,
            fail(&env, leaf_id, 0, task_instance_id, "leaf failure"),
            cancel(&env, root_id),
            event_filter(root_id, root_id, EventKindTag::CancelRoot),
        )
        .await;
        assert_eq!(failure?, JobState::Failed);
        assert_stale(
            &cancelled,
            &StaleStateError::JobAlreadyTerminated(JobState::Failed),
            "a cancel after the failure",
        );
        assert_states(
            &env,
            &[
                (root_id, JobState::Failed),
                (leaf_id, JobState::Failed),
                (sibling_id, JobState::Cancelled),
            ],
        )
        .await;
        assert_quiescent(&env, root_id).await;

        let [root_id, leaf_id, sibling_id] = start_two_leaf_tree(&env, ONE_TASK).await;
        let task_instance_id = instance(&env, leaf_id, TaskId::Index(0)).await;
        let (cancelled, failure) = race(
            &env,
            root_id,
            HookPoint::CommitterBeforeDbTransaction,
            cancel(&env, root_id),
            fail(&env, leaf_id, 0, task_instance_id, "leaf failure"),
            event_filter(root_id, leaf_id, EventKindTag::TerminalFailure),
        )
        .await;
        assert_eq!(cancelled?, JobState::Cancelled);
        assert_stale(
            &failure,
            &StaleStateError::JobAlreadyTerminated(JobState::Cancelled),
            "a failure after the cancel",
        );
        assert_eq!(row_error(&env, leaf_id).await, None);
        assert_states(
            &env,
            &[
                (root_id, JobState::Cancelled),
                (leaf_id, JobState::Cancelled),
                (sibling_id, JobState::Cancelled),
            ],
        )
        .await;
        assert_quiescent(&env, root_id).await;
    }
    env.finish().await;
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

/// T4: `k` siblings complete at the same time; their parent advances exactly once.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn siblings_completing_together_advance_their_parent_once() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    for _ in 0..race_iterations().div_ceil(5) {
        for num_children in [2, 8, 64] {
            for rounds in [SiblingRounds::One, SiblingRounds::Each, SiblingRounds::Free] {
                for parent_shape in [ONE_TASK, WITH_COMMIT] {
                    assert_siblings_advance_their_parent_once(
                        &env,
                        num_children,
                        rounds,
                        parent_shape,
                    )
                    .await?;
                }
            }
        }
    }
    env.finish().await;
    Ok(())
}

/// Registers `num_children` one-task children under a root of `parent_shape`, then reports their
/// successes as `rounds` says.
///
/// # Errors
///
/// Returns an error if any report fails.
async fn assert_siblings_advance_their_parent_once(
    env: &TestEnv,
    num_children: usize,
    rounds: SiblingRounds,
    parent_shape: JobShape,
) -> anyhow::Result<()> {
    let what = format!("{num_children} children in {rounds:?} rounds");
    let parent_id = start_root(env, parent_shape).await;
    let child_ids = register_children(env, parent_id, &vec![ONE_TASK; num_children]).await;
    let parent_outputs = row_outputs(env, parent_id).await;
    let mut instances = Vec::with_capacity(num_children);
    for child_id in &child_ids {
        instances.push(instance(env, *child_id, TaskId::Index(0)).await);
    }
    let parent_applies = observe_applies(env, parent_id, &[parent_id]);
    let reports: Vec<_> = child_ids
        .iter()
        .zip(&instances)
        .map(|(child_id, task_instance_id)| {
            succeed(env, *child_id, 0, *task_instance_id, Vec::new())
        })
        .collect();

    let mut results = Vec::with_capacity(num_children);
    match rounds {
        SiblingRounds::One => {
            let pause = env.hooks.arm(
                HookPoint::CommitterAfterReceive,
                HookFilter::root(parent_id),
                HookAction::Pause,
                HookCount::Once,
            );
            let submitted = env.hooks.arm(
                HookPoint::EventSubmitted,
                HookFilter {
                    root_id: Some(parent_id),
                    job_id: None,
                    event_kind: Some(EventKindTag::RegularTasksDone),
                },
                HookAction::Observe,
                HookCount::Always,
            );
            let mut reports = reports.into_iter();
            let mut handles = vec![tokio::spawn(
                reports.next().expect("there is at least one child"),
            )];
            pause.reached(1).await;
            handles.extend(reports.map(tokio::spawn));
            submitted.reached(num_children).await;
            pause.release();
            for handle in handles {
                results.push(handle.await.expect("a report should not panic"));
            }
        }
        SiblingRounds::Each => {
            for report in reports {
                results.push(report.await);
            }
        }
        SiblingRounds::Free => {
            let handles: Vec<_> = reports.into_iter().map(tokio::spawn).collect();
            for handle in handles {
                results.push(handle.await.expect("a report should not panic"));
            }
        }
    }

    for result in results {
        assert_eq!(result?, JobState::Succeeded, "{what}");
    }
    assert_eq!(
        parent_applies[0].hits(),
        1,
        "{what}: the parent advances once"
    );
    let parent_state = if parent_shape.with_commit {
        JobState::CommitReady
    } else {
        JobState::Succeeded
    };
    let mut expected = vec![(parent_id, parent_state)];
    expected.extend(
        child_ids
            .iter()
            .map(|child_id| (*child_id, JobState::Succeeded)),
    );
    assert_states(env, &expected).await;
    assert!(
        parent_outputs.is_some(),
        "{what}: the parent stored its outputs"
    );
    for child_id in &child_ids {
        assert_eq!(
            row_outputs(env, *child_id).await,
            parent_outputs,
            "{what}: outputs of child {child_id:?}"
        );
    }
    assert_eq!(row_outputs(env, parent_id).await, parent_outputs, "{what}");
    if parent_shape.with_commit {
        assert_eq!(run_commit(env, parent_id).await, JobState::Succeeded);
    }
    assert_quiescent(env, parent_id).await;
    Ok(())
}

/// T5: a straggler instance of a leaf's task fails after another instance succeeded, both before
/// and after the success is applied.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_straggler_failure_after_a_leaf_success_is_stale() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let straggling = JobShape {
        max_num_instances: 2,
        ..ONE_TASK
    };
    for _ in 0..race_iterations() {
        for in_window in [true, false] {
            let [root_id, leaf_id, sibling_id] = start_two_leaf_tree(&env, straggling).await;
            let winner = instance(&env, leaf_id, TaskId::Index(0)).await;
            let straggler = instance(&env, leaf_id, TaskId::Index(0)).await;

            let (success, failure) = if in_window {
                let pause = env.hooks.arm(
                    HookPoint::CommitterBeforeDbTransaction,
                    HookFilter::root(root_id),
                    HookAction::Pause,
                    HookCount::Once,
                );
                let success = tokio::spawn(succeed(&env, leaf_id, 0, winner, Vec::new()));
                pause.reached(1).await;
                let failure = fail(&env, leaf_id, 0, straggler, "straggler").await;
                pause.release();
                (success.await.expect("the report should not panic"), failure)
            } else {
                let success = succeed(&env, leaf_id, 0, winner, Vec::new()).await;
                (
                    success,
                    fail(&env, leaf_id, 0, straggler, "straggler").await,
                )
            };

            assert_eq!(success?, JobState::Succeeded);
            let stale_error = if in_window {
                StaleStateError::TaskAlreadyTerminated(TaskState::Succeeded)
            } else {
                StaleStateError::JobNoLongerRunning
            };
            assert_stale(
                &failure,
                &stale_error,
                &format!("a straggler failure (in the window: {in_window})"),
            );
            assert_eq!(row_error(&env, leaf_id).await, None);
            assert_states(
                &env,
                &[
                    (root_id, JobState::BlockingOnChildren),
                    (leaf_id, JobState::Succeeded),
                    (sibling_id, JobState::Running),
                ],
            )
            .await;
            assert_eq!(run_only_task(&env, sibling_id).await, JobState::Succeeded);
            assert_states(&env, &[(root_id, JobState::Succeeded)]).await;
            assert_quiescent(&env, root_id).await;
        }
    }
    env.finish().await;
    Ok(())
}

/// Starts a root `R` whose child `P`, with a commit task, blocks on one child `C`, then runs `C`
/// so that `P` is commit-ready, and registers an instance of `P`'s commit task.
///
/// # Returns
///
/// `[R, P, C]` and the instance of `P`'s commit task.
async fn start_commit_ready_child(env: &TestEnv) -> ([JobId; 3], u64) {
    let root_id = start_root(env, ONE_TASK).await;
    let parent_id = register_children(env, root_id, &[WITH_COMMIT]).await[0];
    let child_id = register_children(env, parent_id, &[ONE_TASK]).await[0];
    assert_eq!(run_only_task(env, child_id).await, JobState::Succeeded);
    assert_states(
        env,
        &[
            (root_id, JobState::BlockingOnChildren),
            (parent_id, JobState::CommitReady),
        ],
    )
    .await;
    let commit_instance = instance(env, parent_id, TaskId::Commit).await;
    ([root_id, parent_id, child_id], commit_instance)
}

/// T7: a cancel skips a non-root commit-ready job, whose commit then succeeds without
/// propagating; a commit-ready root is deactivated, and its late commit is stale.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_cancel_skips_a_non_root_commit_ready_job_but_not_a_commit_ready_root()
-> anyhow::Result<()> {
    let env = TestEnv::start().await;
    for _ in 0..race_iterations() {
        let ([root_id, parent_id, child_id], commit_instance) =
            start_commit_ready_child(&env).await;
        assert_eq!(cancel(&env, root_id).await?, JobState::Cancelled);
        assert_states(
            &env,
            &[
                (root_id, JobState::Cancelled),
                (parent_id, JobState::CommitReady),
                (child_id, JobState::Succeeded),
            ],
        )
        .await;
        assert_eq!(
            succeed_commit(&env, parent_id, commit_instance).await?,
            JobState::Succeeded
        );
        assert_states(
            &env,
            &[
                (root_id, JobState::Cancelled),
                (parent_id, JobState::Succeeded),
            ],
        )
        .await;
        assert_quiescent(&env, root_id).await;

        let ([root_id, parent_id, _], commit_instance) = start_commit_ready_child(&env).await;
        let (cancelled, committed) = race(
            &env,
            root_id,
            HookPoint::CommitterAfterReceive,
            cancel(&env, root_id),
            succeed_commit(&env, parent_id, commit_instance),
            event_filter(root_id, parent_id, EventKindTag::CommitDone),
        )
        .await;
        assert_eq!(cancelled?, JobState::Cancelled);
        assert_eq!(committed?, JobState::Succeeded);
        assert_states(
            &env,
            &[
                (root_id, JobState::Cancelled),
                (parent_id, JobState::Succeeded),
            ],
        )
        .await;
        assert_quiescent(&env, root_id).await;

        for with_cleanup in [false, true] {
            let root_id = start_root(
                &env,
                JobShape {
                    with_cleanup,
                    ..WITH_COMMIT
                },
            )
            .await;
            assert_eq!(run_only_task(&env, root_id).await, JobState::CommitReady);
            let commit_instance = instance(&env, root_id, TaskId::Commit).await;
            let deactivated = if with_cleanup {
                JobState::CleanupReady
            } else {
                JobState::Cancelled
            };
            assert_eq!(cancel(&env, root_id).await?, deactivated);
            let late_commit = succeed_commit(&env, root_id, commit_instance).await;
            assert_stale(
                &late_commit,
                &StaleStateError::JobNoLongerCommitReady,
                &format!("a late commit of a cancelled root (cleanup: {with_cleanup})"),
            );
            assert_states(&env, &[(root_id, deactivated)]).await;
            if with_cleanup {
                assert_eq!(run_cleanup(&env, root_id).await, JobState::Cancelled);
            }
            assert_quiescent(&env, root_id).await;
        }
    }
    env.finish().await;
    Ok(())
}

/// T10: a leaf's report that breaks a limit fails the leaf and its tree, registers nothing, and is
/// not fatal.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_limit_violation_in_a_tree_fails_the_tree() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let output_limit = usize::try_from(MAX_REPORT_OUTPUT_BYTES)?;
    let cases: Vec<(&str, Vec<u8>, Vec<ChildJobSubmission>)> = vec![
        (
            "per report",
            vec![0u8; 4],
            vec![
                ChildJobSubmission {
                    compressed_serialized_task_graph: vec![0u8; 1],
                    compressed_serialized_job_inputs: Vec::new(),
                };
                MAX_CHILDREN_PER_JOB + 1
            ],
        ),
        (
            "is invalid",
            vec![0u8; 4],
            vec![ChildJobSubmission {
                compressed_serialized_task_graph: vec![0u8; 16],
                compressed_serialized_job_inputs: Vec::new(),
            }],
        ),
        ("task outputs of", vec![0u8; output_limit + 1], Vec::new()),
    ];
    for (message, output, children) in cases {
        let [root_id, leaf_id, sibling_id] = start_two_leaf_tree(&env, ONE_TASK).await;
        let task_instance_id = instance(&env, leaf_id, TaskId::Index(0)).await;
        let state = env
            .service
            .succeed_task_instance(
                env.service.session_id(),
                leaf_id,
                task_instance_id,
                0,
                serialized_output(output),
                children,
            )
            .await?;
        assert_eq!(state, JobState::Failed, "case `{message}`");
        assert_states(
            &env,
            &[
                (root_id, JobState::Failed),
                (leaf_id, JobState::Failed),
                (sibling_id, JobState::Cancelled),
            ],
        )
        .await;
        let error = row_error(&env, leaf_id)
            .await
            .expect("the failed leaf should have an error message");
        assert!(
            error.contains(message),
            "case `{message}`: unexpected error `{error}`"
        );
        assert_eq!(children_of(&env, leaf_id).await, Vec::new());
        assert_quiescent(&env, root_id).await;
    }
    env.finish().await;
    Ok(())
}

/// Asserts that `result` is a [`StorageServerError::BadRequest`].
fn assert_bad_request<ValueType: std::fmt::Debug>(
    result: &Result<ValueType, StorageServerError>,
    what: &str,
) {
    assert!(
        matches!(result, Err(StorageServerError::BadRequest(_))),
        "{what}: expected a bad request, got {result:?}"
    );
}

/// The external job RPCs reject a child job, also while its registration is between its commit
/// and its cache insert, and keep their answers for roots and unknown jobs.
#[tokio::test]
#[ignore = "requires MariaDB"]
async fn external_job_rpcs_accept_only_roots() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, ONE_TASK).await;
    let task_instance_id = instance(&env, root_id, TaskId::Index(0)).await;
    let pause = env.hooks.arm(
        HookPoint::CommitterAfterCommit,
        HookFilter::root(root_id),
        HookAction::Pause,
        HookCount::Once,
    );
    let report = tokio::spawn(succeed(
        &env,
        root_id,
        0,
        task_instance_id,
        vec![child(ONE_TASK)],
    ));
    pause.reached(1).await;
    let child_id = children_of(&env, root_id).await[0];
    assert_eq!(env.cached_job_state(child_id).await, None);
    assert_bad_request(
        &env.service.cancel_job(child_id).await,
        "a cancel of an uncached child",
    );
    assert_bad_request(
        &env.service.get_job_state(child_id).await,
        "the state of an uncached child",
    );
    assert!(
        matches!(
            env.service.start_job(child_id).await,
            Err(StorageServerError::JobNotFound(_))
        ),
        "a start of an uncached child should not find the job"
    );
    assert!(!env.is_fatal());
    pause.release();
    assert_eq!(
        report.await.expect("the report should not panic")?,
        JobState::BlockingOnChildren
    );

    assert_bad_request(&env.service.start_job(child_id).await, "a child start");
    assert_bad_request(&env.service.cancel_job(child_id).await, "a child cancel");
    assert_bad_request(&env.service.get_job_state(child_id).await, "a child state");
    assert_bad_request(
        &env.service.get_job_outputs(child_id).await,
        "a child's outputs",
    );
    assert_bad_request(
        &env.service.get_job_error(child_id).await,
        "a child's error",
    );
    assert_states(
        &env,
        &[
            (root_id, JobState::BlockingOnChildren),
            (child_id, JobState::Running),
        ],
    )
    .await;

    assert_eq!(run_only_task(&env, child_id).await, JobState::Succeeded);
    assert_eq!(
        env.service.get_job_state(root_id).await?,
        JobState::Succeeded
    );
    assert_eq!(env.service.get_job_outputs(root_id).await?.len(), 1);
    assert_bad_request(
        &env.service.get_job_outputs(child_id).await,
        "a succeeded child's outputs",
    );

    let unknown_id = JobId::from(u64::MAX);
    assert!(matches!(
        env.service.start_job(unknown_id).await,
        Err(StorageServerError::JobNotFound(_))
    ));
    assert!(matches!(
        env.service.cancel_job(unknown_id).await,
        Err(StorageServerError::JobNotFound(_))
    ));
    assert!(matches!(
        env.service.get_job_state(unknown_id).await,
        Err(StorageServerError::Db(DbError::JobNotFound(_)))
    ));
    assert!(matches!(
        env.service.get_job_outputs(unknown_id).await,
        Err(StorageServerError::Db(DbError::JobNotFound(_)))
    ));
    assert!(matches!(
        env.service.get_job_error(unknown_id).await,
        Err(StorageServerError::Db(DbError::JobNotFound(_)))
    ));
    assert!(!env.is_fatal());
    assert_quiescent(&env, root_id).await;
    env.finish().await;
    Ok(())
}
