//! Child-job registration through the storage service: the parent blocks on its children, the
//! children run and unblock it, and a report that breaks a limit fails its task instead.

use std::collections::HashSet;
use std::time::Duration;

use spider_core::job::JobState;
use spider_core::task::DataTypeDescriptor;
use spider_core::task::ExecutionPolicy;
use spider_core::task::TaskDescriptor;
use spider_core::task::TdlContext;
use spider_core::task::TerminationTaskDescriptor;
use spider_core::task::ValueTypeDescriptor;
use spider_core::types::id::JobId;
use spider_core::types::id::TaskId;
use spider_core::types::io::SerializedTaskOutputs;
use spider_core::types::io::TaskGraphInput;
use spider_core::types::io::TaskGraphInputBuilder;
use spider_core::types::io::TaskOutputsSerializer;
use spider_storage::cache::error::CacheError;
use spider_storage::cache::error::StaleStateError;
use spider_storage::job_submission::ChildJobSubmission;
use spider_storage::limits::MAX_CHILDREN_PER_JOB;
use spider_storage::limits::MAX_DEPTH;
use spider_storage::limits::MAX_ERROR_MESSAGE_CHARS;
use spider_storage::limits::MAX_JOB_CHILD_BYTES;
use spider_storage::limits::MAX_REPORT_OUTPUT_BYTES;
use spider_storage::state::StorageServerError;
use spider_storage::test_hooks::HookAction;
use spider_storage::test_hooks::HookCount;
use spider_storage::test_hooks::HookFilter;
use spider_storage::test_hooks::HookPoint;

use super::job_tree_infra::TestEnv;
use super::task_graph_builder::SubmittedTaskGraph;
use super::task_graph_builder::compress_job_inputs;
use super::task_graph_builder::compress_task_graph;

/// The shape of a test job.
#[derive(Clone, Copy)]
struct JobShape {
    num_tasks: usize,
    max_num_retry: u32,
    with_commit: bool,
}

const ONE_TASK: JobShape = JobShape {
    num_tasks: 1,
    max_num_retry: 0,
    with_commit: false,
};

/// Builds a job of independent tasks, each taking and returning one byte-typed value. Each task's
/// function name is `label` followed by its index, so that two jobs of different labels differ.
///
/// # Returns
///
/// The task graph and its inputs.
///
/// # Panics
///
/// Panics if the graph fails to build.
fn build_job(shape: JobShape, label: &str) -> (SubmittedTaskGraph, TaskGraphInput) {
    let execution_policy = Some(ExecutionPolicy {
        max_num_retry: shape.max_num_retry,
        ..ExecutionPolicy::default()
    });
    let commit_task = shape.with_commit.then(|| TerminationTaskDescriptor {
        tdl_context: TdlContext {
            package: "test".to_owned(),
            task_func: format!("{label}_commit"),
        },
        execution_policy: execution_policy.clone(),
    });
    let mut graph =
        SubmittedTaskGraph::new(commit_task, None).expect("task graph creation should succeed");
    let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
    let mut inputs = TaskGraphInputBuilder::new();
    for task_index in 0..shape.num_tasks {
        graph
            .insert_task(TaskDescriptor {
                tdl_context: TdlContext {
                    package: "test".to_owned(),
                    task_func: format!("{label}{task_index}"),
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
/// A one-task child job whose task's function name is `label`0.
///
/// # Panics
///
/// Panics if the job fails to compress.
fn child(label: &str) -> ChildJobSubmission {
    let (graph, inputs) = build_job(ONE_TASK, label);
    ChildJobSubmission {
        compressed_serialized_task_graph: compress_task_graph(&graph)
            .expect("task graph compression should succeed"),
        compressed_serialized_job_inputs: compress_job_inputs(&inputs)
            .expect("input compression should succeed"),
    }
}

/// # Returns
///
/// `size` pseudo-random bytes from `seed`.
fn incompressible(size: usize, seed: u64) -> Vec<u8> {
    let mut xorshift_state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed;
    (0..size)
        .map(|_| {
            xorshift_state ^= xorshift_state << 13;
            xorshift_state ^= xorshift_state >> 7;
            xorshift_state ^= xorshift_state << 17;
            xorshift_state.to_le_bytes()[0]
        })
        .collect()
}

/// # Returns
///
/// A valid one-task child job whose only input is `input_size` incompressible bytes.
///
/// # Panics
///
/// Panics if the job fails to build or to compress.
fn large_child(label: &str, input_size: usize) -> ChildJobSubmission {
    let (graph, _) = build_job(ONE_TASK, label);
    let mut inputs = TaskGraphInputBuilder::new();
    inputs
        .append_task_input(&incompressible(input_size, 0))
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
/// A child job of `size` bytes that is never decoded.
fn opaque_child(size: usize) -> ChildJobSubmission {
    ChildJobSubmission {
        compressed_serialized_task_graph: vec![0u8; size],
        compressed_serialized_job_inputs: Vec::new(),
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

/// Registers and starts a job of `shape`.
///
/// # Returns
///
/// The job ID.
///
/// # Panics
///
/// Panics if the registration or the start fails, or if the job's tasks are not published.
async fn start_job(env: &TestEnv, shape: JobShape) -> JobId {
    let (graph, inputs) = build_job(shape, "root");
    let job_id = env.register_job(&graph, &inputs).await;
    env.service
        .start_job(job_id)
        .await
        .expect("the job should start");
    assert_eq!(
        env.poll_ready_tasks(job_id, shape.num_tasks).await.len(),
        shape.num_tasks
    );
    job_id
}

/// # Panics
///
/// Panics if the instance creation fails.
async fn create_instance(env: &TestEnv, job_id: JobId, task_id: TaskId) -> u64 {
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

/// Reports the success of `task_instance_id` of task `task_index`, with the output `[value; 4]`
/// and `children`.
///
/// # Returns
///
/// Forwards [`spider_storage::state::ServiceState::succeed_task_instance`]'s return
/// values.
async fn report(
    env: &TestEnv,
    job_id: JobId,
    task_index: usize,
    task_instance_id: u64,
    children: Vec<ChildJobSubmission>,
) -> Result<JobState, StorageServerError> {
    env.service
        .succeed_task_instance(
            env.service.session_id(),
            job_id,
            task_instance_id,
            task_index,
            serialized_output(vec![u8::try_from(task_index).expect("small index"); 4]),
            children,
        )
        .await
}

/// Runs a new instance of the only task of `job_id` to success with `children`.
///
/// # Returns
///
/// Forwards [`report`]'s return values.
async fn run_only_task(
    env: &TestEnv,
    job_id: JobId,
    children: Vec<ChildJobSubmission>,
) -> Result<JobState, StorageServerError> {
    let task_instance_id = create_instance(env, job_id, TaskId::Index(0)).await;
    report(env, job_id, 0, task_instance_id, children).await
}

/// A child row of the jobs table.
#[derive(Debug, PartialEq, Eq)]
struct ChildRow {
    id: JobId,
    root_id: Option<JobId>,
    state: JobState,
    compressed_task_graph: Vec<u8>,
}

/// # Returns
///
/// The rows whose parent is `parent_id`, by ascending ID.
///
/// # Panics
///
/// Panics if the query fails.
async fn child_rows(env: &TestEnv, parent_id: JobId) -> Vec<ChildRow> {
    sqlx::query_as::<_, (JobId, Option<JobId>, JobState, Vec<u8>)>(
        "SELECT `id`, `root_id`, `state`, `compressed_serialized_task_graph` FROM `jobs` WHERE \
         `parent_id` = ? ORDER BY `id`;",
    )
    .bind(parent_id)
    .fetch_all(&env.pool)
    .await
    .expect("the child row query should succeed")
    .into_iter()
    .map(|(id, root_id, state, compressed_task_graph)| ChildRow {
        id,
        root_id,
        state,
        compressed_task_graph,
    })
    .collect()
}

/// Polls the task lane until every job of `job_ids` has published exactly one ready task.
///
/// # Panics
///
/// Panics if the entries do not arrive within 10 seconds, or if a job publishes twice.
async fn poll_one_task_each(env: &TestEnv, job_ids: &[JobId]) {
    let mut remaining: HashSet<JobId> = job_ids.iter().copied().collect();
    tokio::time::timeout(Duration::from_secs(10), async {
        while !remaining.is_empty() {
            let entries = env
                .service
                .poll_ready_tasks(remaining.len(), Duration::from_secs(1))
                .await
                .expect("polling should succeed");
            for entry in entries {
                assert!(
                    remaining.remove(&entry.job_id),
                    "job {:?} published an unexpected task",
                    entry.job_id
                );
            }
        }
    })
    .await
    .expect("every child's ready task should be published");
}

/// Asserts that `job_id`'s cached and persisted states are both `expected`, and that the service
/// is not shutting down fatally.
async fn assert_job_state(env: &TestEnv, job_id: JobId, expected: JobState) {
    let state = match env.cached_job_state(job_id).await {
        Some(state) => state,
        None => env.row_state(job_id).await,
    };
    assert_eq!(state, expected);
    assert_eq!(env.row_state(job_id).await, expected);
    assert!(!env.is_fatal());
}

/// # Panics
///
/// Panics if the query fails.
async fn has_outputs(env: &TestEnv, job_id: JobId) -> bool {
    sqlx::query_scalar::<_, bool>(
        "SELECT `serialized_job_outputs` IS NOT NULL FROM `jobs` WHERE `id` = ?;",
    )
    .bind(job_id)
    .fetch_one(&env.pool)
    .await
    .expect("the outputs query should succeed")
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_registered_parent_commits_after_its_children_succeed() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let parent_id = start_job(
        &env,
        JobShape {
            with_commit: true,
            ..ONE_TASK
        },
    )
    .await;
    let children = vec![child("a"), child("b"), child("c")];

    assert_eq!(
        run_only_task(&env, parent_id, children.clone()).await?,
        JobState::BlockingOnChildren
    );
    assert_job_state(&env, parent_id, JobState::BlockingOnChildren).await;
    assert!(has_outputs(&env, parent_id).await);
    let rows = child_rows(&env, parent_id).await;
    assert_eq!(
        rows.iter()
            .map(|row| (row.root_id, row.state, row.compressed_task_graph.clone()))
            .collect::<Vec<_>>(),
        children
            .iter()
            .map(|child| {
                (
                    Some(parent_id),
                    JobState::Running,
                    child.compressed_serialized_task_graph.clone(),
                )
            })
            .collect::<Vec<_>>()
    );
    assert!(rows.iter().all(|row| row.id.get() > parent_id.get()));
    let child_ids: Vec<JobId> = rows.iter().map(|row| row.id).collect();
    for child_id in &child_ids {
        assert_job_state(&env, *child_id, JobState::Running).await;
    }
    poll_one_task_each(&env, &child_ids).await;

    for (position, child_id) in child_ids.iter().enumerate() {
        assert_eq!(
            run_only_task(&env, *child_id, Vec::new()).await?,
            JobState::Succeeded
        );
        let parent_state = if position + 1 == child_ids.len() {
            JobState::CommitReady
        } else {
            JobState::BlockingOnChildren
        };
        assert_job_state(&env, parent_id, parent_state).await;
    }
    let commits = env
        .service
        .poll_commit_ready_tasks(1, Duration::from_secs(5))
        .await?;
    assert_eq!(
        commits.iter().map(|entry| entry.job_id).collect::<Vec<_>>(),
        vec![parent_id]
    );
    let commit = env
        .service
        .create_task_instance(
            env.service.session_id(),
            parent_id,
            TaskId::Commit,
            env.execution_manager_id,
        )
        .await?;
    assert_eq!(
        SerializedTaskOutputs::deserialize_from_raw(&commit.serialized_task_io)?,
        TaskOutputsSerializer::deserialize(&serialized_output(vec![0; 4]))?
    );
    assert_eq!(
        env.service
            .succeed_commit_task_instance(
                env.service.session_id(),
                parent_id,
                commit.task_instance_id,
            )
            .await?,
        JobState::Succeeded
    );
    assert_job_state(&env, parent_id, JobState::Succeeded).await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn sibling_tasks_register_the_union_of_their_children_in_task_order() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let parent_id = start_job(
        &env,
        JobShape {
            num_tasks: 2,
            ..ONE_TASK
        },
    )
    .await;
    let first = create_instance(&env, parent_id, TaskId::Index(0)).await;
    let second = create_instance(&env, parent_id, TaskId::Index(1)).await;

    assert_eq!(
        report(&env, parent_id, 1, second, vec![child("b")]).await?,
        JobState::Running
    );
    assert_eq!(child_rows(&env, parent_id).await, Vec::new());
    assert_eq!(
        report(&env, parent_id, 0, first, vec![child("a0"), child("a1")]).await?,
        JobState::BlockingOnChildren
    );
    let rows = child_rows(&env, parent_id).await;
    assert_eq!(
        rows.iter()
            .map(|row| row.compressed_task_graph.clone())
            .collect::<Vec<_>>(),
        [child("a0"), child("a1"), child("b")]
            .into_iter()
            .map(|child| child.compressed_serialized_task_graph)
            .collect::<Vec<_>>()
    );
    let child_ids: Vec<JobId> = rows.iter().map(|row| row.id).collect();
    poll_one_task_each(&env, &child_ids).await;
    for child_id in &child_ids {
        run_only_task(&env, *child_id, Vec::new()).await?;
    }
    assert_job_state(&env, parent_id, JobState::Succeeded).await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_duplicate_report_with_children_stores_nothing() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let parent_id = start_job(
        &env,
        JobShape {
            num_tasks: 2,
            ..ONE_TASK
        },
    )
    .await;
    let first = create_instance(&env, parent_id, TaskId::Index(0)).await;
    assert_eq!(
        report(&env, parent_id, 0, first, vec![child("a")]).await?,
        JobState::Running
    );

    let duplicate = report(&env, parent_id, 0, first, vec![child("x"), child("y")]).await;
    assert!(
        matches!(
            duplicate,
            Err(StorageServerError::Cache(CacheError::StaleState(
                StaleStateError::InvalidTaskInstanceId
            )))
        ),
        "a duplicate report should be stale, got {duplicate:?}"
    );
    let second = create_instance(&env, parent_id, TaskId::Index(1)).await;
    assert_eq!(
        report(&env, parent_id, 1, second, Vec::new()).await?,
        JobState::BlockingOnChildren
    );
    let rows = child_rows(&env, parent_id).await;
    assert_eq!(
        rows.iter()
            .map(|row| row.compressed_task_graph.clone())
            .collect::<Vec<_>>(),
        vec![child("a").compressed_serialized_task_graph]
    );
    poll_one_task_each(&env, &[rows[0].id]).await;
    run_only_task(&env, rows[0].id, Vec::new()).await?;
    assert_job_state(&env, parent_id, JobState::Succeeded).await;
    env.finish().await;
    Ok(())
}

/// Starts a one-task job with `max_num_retry`, then reports its task's success with `children`
/// and `output` once per allowed attempt, each time on a new instance.
///
/// # Returns
///
/// The job ID, and the job state after each report.
///
/// # Panics
///
/// Panics if any step fails.
async fn report_until_failed(
    env: &TestEnv,
    max_num_retry: u32,
    output: &[u8],
    children: &[ChildJobSubmission],
) -> (JobId, Vec<JobState>) {
    let job_id = start_job(
        env,
        JobShape {
            max_num_retry,
            ..ONE_TASK
        },
    )
    .await;
    let mut states = Vec::new();
    for attempt in 0..=max_num_retry {
        if attempt > 0 {
            assert_eq!(env.poll_ready_tasks(job_id, 1).await, vec![0]);
        }
        let task_instance_id = create_instance(env, job_id, TaskId::Index(0)).await;
        states.push(
            env.service
                .succeed_task_instance(
                    env.service.session_id(),
                    job_id,
                    task_instance_id,
                    0,
                    serialized_output(output.to_vec()),
                    children.to_vec(),
                )
                .await
                .expect("a rejected report should fail the task, not the request"),
        );
    }
    (job_id, states)
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_report_over_a_limit_fails_its_task() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let output_limit = usize::try_from(MAX_REPORT_OUTPUT_BYTES)?;
    let cases = [
        (
            "child jobs exceed the limit",
            vec![0u8; 4],
            vec![child("c"); MAX_CHILDREN_PER_JOB + 1],
        ),
        (
            "child job 1 of",
            vec![0u8; 4],
            vec![child("c"), opaque_child(4 * 1024 * 1024 + 1)],
        ),
        (
            "bytes per report",
            vec![0u8; 4],
            vec![opaque_child(3 * 1024 * 1024 + 512 * 1024); 5],
        ),
        ("is invalid", vec![0u8; 4], vec![opaque_child(16)]),
        ("task outputs of", vec![0u8; output_limit + 1], Vec::new()),
    ];
    for (message, output, children) in cases {
        for max_num_retry in [0, 1] {
            let (job_id, states) =
                report_until_failed(&env, max_num_retry, &output, &children).await;
            let mut expected = vec![JobState::Running; usize::try_from(max_num_retry)?];
            expected.push(JobState::Failed);
            assert_eq!(states, expected, "case `{message}`");
            assert_job_state(&env, job_id, JobState::Failed).await;
            let error = env.service.get_job_error(job_id).await?;
            assert!(
                error.contains(message),
                "case `{message}`: unexpected error `{error}`"
            );
            assert_eq!(child_rows(&env, job_id).await, Vec::new());
        }
    }
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_report_beyond_the_maximum_depth_fails_its_task() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_job(&env, ONE_TASK).await;
    let mut chain = vec![root_id];
    for depth in 1..=MAX_DEPTH {
        let parent_id = *chain.last().expect("the chain should not be empty");
        assert_eq!(
            run_only_task(&env, parent_id, vec![child(&format!("d{depth}"))]).await?,
            JobState::BlockingOnChildren
        );
        let rows = child_rows(&env, parent_id).await;
        assert_eq!(rows.len(), 1);
        poll_one_task_each(&env, &[rows[0].id]).await;
        chain.push(rows[0].id);
    }

    let deepest = *chain.last().expect("the chain should not be empty");
    assert_eq!(
        run_only_task(&env, deepest, vec![child("too_deep")]).await?,
        JobState::Failed
    );
    assert_eq!(child_rows(&env, deepest).await, Vec::new());
    assert!(
        env.row_error(deepest)
            .await
            .contains("exceed the maximum depth")
    );
    for job_id in chain {
        assert_job_state(&env, job_id, JobState::Failed).await;
    }
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_rejected_registration_payload_fails_the_parent() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let parent_id = start_job(&env, ONE_TASK).await;
    let fail_db = env.hooks.arm(
        HookPoint::CommitterBeforeDbTransaction,
        HookFilter::root(parent_id),
        HookAction::FailDb {
            deterministic_payload: true,
        },
        HookCount::Once,
    );

    assert_eq!(
        run_only_task(&env, parent_id, vec![child("a"), child("b")]).await?,
        JobState::Failed
    );
    assert_eq!(fail_db.hits(), 1);
    assert_job_state(&env, parent_id, JobState::Failed).await;
    assert_eq!(
        env.service.get_job_error(parent_id).await?,
        "child payload rejected by the database"
    );
    assert_eq!(child_rows(&env, parent_id).await, Vec::new());
    drop(fail_db);
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn children_over_the_job_limits_across_reports_fail_the_task_before_decoding()
-> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let parent_id = start_job(
        &env,
        JobShape {
            num_tasks: 2,
            ..ONE_TASK
        },
    )
    .await;
    let first = create_instance(&env, parent_id, TaskId::Index(0)).await;
    let second = create_instance(&env, parent_id, TaskId::Index(1)).await;
    let stored = vec![large_child("a", 2 * 1024 * 1024); 2];
    assert_eq!(
        report(&env, parent_id, 0, first, stored).await?,
        JobState::Running
    );

    let undecodable = vec![opaque_child(4 * 1024 * 1024); 4];
    assert_eq!(
        report(&env, parent_id, 1, second, undecodable).await?,
        JobState::Failed
    );
    assert_job_state(&env, parent_id, JobState::Failed).await;
    let error = env.service.get_job_error(parent_id).await?;
    assert!(
        error.contains(&format!("{MAX_JOB_CHILD_BYTES} bytes per job")),
        "unexpected error `{error}`"
    );
    assert_eq!(child_rows(&env, parent_id).await, Vec::new());
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn rejected_reports_release_their_children_from_the_job_limits() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let parent_id = start_job(
        &env,
        JobShape {
            num_tasks: 2,
            max_num_retry: 1,
            ..ONE_TASK
        },
    )
    .await;
    let child = large_child("a", 2 * 1024 * 1024);
    let child_size = child.size();
    let max_children = MAX_JOB_CHILD_BYTES / child_size;
    assert!(
        max_children >= 3,
        "a child of {child_size} bytes is too large for the test"
    );
    let num_later_children = usize::try_from(max_children - 2)?;

    let invalid = create_instance(&env, parent_id, TaskId::Index(0)).await;
    assert_eq!(
        report(
            &env,
            parent_id,
            0,
            invalid,
            vec![child.clone(), child.clone(), opaque_child(16)]
        )
        .await?,
        JobState::Running
    );
    assert_eq!(env.poll_ready_tasks(parent_id, 1).await, vec![0]);
    let first = create_instance(&env, parent_id, TaskId::Index(0)).await;
    assert_eq!(
        report(&env, parent_id, 0, first, vec![child.clone(); 2]).await?,
        JobState::Running
    );
    let duplicate = report(&env, parent_id, 0, first, vec![child.clone(); 2]).await;
    assert!(
        matches!(
            duplicate,
            Err(StorageServerError::Cache(CacheError::StaleState(
                StaleStateError::InvalidTaskInstanceId
            )))
        ),
        "a duplicate report should be stale, got {duplicate:?}"
    );

    let second = create_instance(&env, parent_id, TaskId::Index(1)).await;
    assert_eq!(
        report(
            &env,
            parent_id,
            1,
            second,
            vec![child.clone(); num_later_children]
        )
        .await?,
        JobState::BlockingOnChildren
    );
    assert_job_state(&env, parent_id, JobState::BlockingOnChildren).await;
    assert_eq!(
        child_rows(&env, parent_id).await.len(),
        num_later_children + 2
    );
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn outputs_over_the_packet_size_fail_the_job() -> anyhow::Result<()> {
    const NUM_TASKS: usize = 12;

    let env = TestEnv::start().await;
    let job_id = start_job(
        &env,
        JobShape {
            num_tasks: NUM_TASKS,
            ..ONE_TASK
        },
    )
    .await;
    for task_index in 0..NUM_TASKS {
        let task_instance_id = create_instance(&env, job_id, TaskId::Index(task_index)).await;
        env.service
            .succeed_task_instance(
                env.service.session_id(),
                job_id,
                task_instance_id,
                task_index,
                serialized_output(incompressible(2 * 1024 * 1024, u64::try_from(task_index)?)),
                Vec::new(),
            )
            .await?;
    }
    assert_job_state(&env, job_id, JobState::Failed).await;
    let error = env.service.get_job_error(job_id).await?;
    assert!(
        error.starts_with("job outputs of") && error.contains("serialized bytes exceed the limit"),
        "unexpected error `{error}`"
    );
    assert!(!has_outputs(&env, job_id).await);
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_failure_message_over_the_packet_size_is_truncated() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let job_id = start_job(&env, ONE_TASK).await;
    let max_allowed_packet: u64 = sqlx::query_scalar("SELECT @@max_allowed_packet;")
        .fetch_one(&env.pool)
        .await?;
    let message = "x".repeat(usize::try_from(max_allowed_packet)? + 1);

    let task_instance_id = create_instance(&env, job_id, TaskId::Index(0)).await;
    assert_eq!(
        env.service
            .fail_task_instance(
                env.service.session_id(),
                job_id,
                task_instance_id,
                TaskId::Index(0),
                message.clone(),
            )
            .await?,
        JobState::Failed
    );
    assert_job_state(&env, job_id, JobState::Failed).await;
    assert_eq!(
        env.service.get_job_error(job_id).await?,
        message[..MAX_ERROR_MESSAGE_CHARS]
    );
    env.finish().await;
    Ok(())
}
