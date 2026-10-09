//! Storage restarts of job trees: crashes at the committer's crash points and during recovery, a
//! superseded runtime (T8), and the startup repair of trees forced into broken states.

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
use spider_core::types::io::TaskGraphInput;
use spider_core::types::io::TaskGraphInputBuilder;
use spider_core::types::io::TaskOutputsSerializer;
use spider_storage::job_submission::ChildJobSubmission;
use spider_storage::state::StorageServerError;
use spider_storage::test_hooks::ArmedHook;
use spider_storage::test_hooks::EventKindTag;
use spider_storage::test_hooks::HookAction;
use spider_storage::test_hooks::HookCount;
use spider_storage::test_hooks::HookFilter;
use spider_storage::test_hooks::HookPoint;
use spider_storage::test_hooks::TestHooks;

use super::job_tree_infra::StoppedEnv;
use super::job_tree_infra::TestEnv;
use super::task_graph_builder::SubmittedTaskGraph;
use super::task_graph_builder::compress_job_inputs;
use super::task_graph_builder::compress_task_graph;

/// The shape of a test job: one task, and optional commit and cleanup tasks.
#[derive(Clone, Copy)]
struct JobShape {
    with_commit: bool,
    with_cleanup: bool,
}

const PLAIN: JobShape = JobShape {
    with_commit: false,
    with_cleanup: false,
};

const WITH_COMMIT: JobShape = JobShape {
    with_commit: true,
    ..PLAIN
};

const WITH_CLEANUP: JobShape = JobShape {
    with_cleanup: true,
    ..PLAIN
};

/// Builds a one-task job of `shape` that never retries its task.
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
    graph
        .insert_task(TaskDescriptor {
            tdl_context: TdlContext {
                package: "test".to_owned(),
                task_func: "t0".to_owned(),
            },
            execution_policy,
            inputs: vec![bytes_type.clone()],
            outputs: vec![bytes_type],
            input_sources: None,
        })
        .expect("task insertion should succeed");
    let mut inputs = TaskGraphInputBuilder::new();
    inputs
        .append_task_input(&[0u8; 4])
        .expect("task input appending should succeed");
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

/// Runs a new instance of the only task of `job_id` to success, with one child per shape.
///
/// # Returns
///
/// Forwards [`spider_storage::state::ServiceState::succeed_task_instance`]'s return
/// values.
async fn run_task(
    env: &TestEnv,
    job_id: JobId,
    children: &[JobShape],
) -> Result<JobState, StorageServerError> {
    let task_instance_id = instance(env, job_id, TaskId::Index(0)).await;
    env.service
        .succeed_task_instance(
            env.service.session_id(),
            job_id,
            task_instance_id,
            0,
            TaskOutputsSerializer::from_tuple(&(vec![1u8; 4],))
                .expect("output serialization should succeed"),
            children.iter().copied().map(child).collect(),
        )
        .await
}

/// Runs the only task of `parent_id` to success with one child per shape.
///
/// # Type Parameters
///
/// * `NUM_CHILDREN` - The number of children.
///
/// # Returns
///
/// The IDs of the registered children, by ascending ID.
///
/// # Panics
///
/// Panics if the report fails or does not block the parent on its children.
async fn register_children<const NUM_CHILDREN: usize>(
    env: &TestEnv,
    parent_id: JobId,
    shapes: [JobShape; NUM_CHILDREN],
) -> [JobId; NUM_CHILDREN] {
    assert_eq!(
        run_task(env, parent_id, &shapes)
            .await
            .expect("the registering report should succeed"),
        JobState::BlockingOnChildren
    );
    <[JobId; NUM_CHILDREN]>::try_from(children_of(env, parent_id).await)
        .expect("every child should be registered once")
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
    env.service
        .succeed_commit_task_instance(env.service.session_id(), job_id, task_instance_id)
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

/// Asserts that every job of `expected` is in its state in its row and in the cache, where a
/// terminal job may already be evicted, and that the service has not shut down fatally.
async fn assert_states(env: &TestEnv, expected: &[(JobId, JobState)]) {
    for (job_id, state) in expected {
        let cached = env.cached_job_state(*job_id).await;
        if !state.is_terminal() || cached.is_some() {
            assert_eq!(cached, Some(*state), "cached state of job {job_id:?}");
        }
        assert_eq!(
            env.row_state(*job_id).await,
            *state,
            "row state of job {job_id:?}"
        );
    }
    assert!(!env.is_fatal());
}

/// The job IDs of the entries that one poll of each lane returns.
#[derive(Debug, Default, PartialEq, Eq)]
struct LaneEntries {
    ready: HashSet<JobId>,
    commit: HashSet<JobId>,
    cleanup: HashSet<JobId>,
}

/// Polls the three lanes until they hold at least `expected`'s entries, with every entry polled
/// recorded.
///
/// # Returns
///
/// The job IDs of every polled entry, by lane.
///
/// # Panics
///
/// Panics if a poll fails, or if the expected entries do not arrive within 10 seconds.
async fn poll_lanes(env: &TestEnv, expected: &LaneEntries) -> LaneEntries {
    let mut polled = LaneEntries::default();
    tokio::time::timeout(Duration::from_secs(10), async {
        while !(polled.ready.is_superset(&expected.ready)
            && polled.commit.is_superset(&expected.commit)
            && polled.cleanup.is_superset(&expected.cleanup))
        {
            let wait = Duration::from_millis(200);
            let service = &env.service;
            polled.ready.extend(
                service
                    .poll_ready_tasks(64, wait)
                    .await
                    .expect("the ready poll should succeed")
                    .into_iter()
                    .map(|entry| entry.job_id),
            );
            polled.commit.extend(
                service
                    .poll_commit_ready_tasks(64, wait)
                    .await
                    .expect("the commit poll should succeed")
                    .into_iter()
                    .map(|entry| entry.job_id),
            );
            polled.cleanup.extend(
                service
                    .poll_cleanup_ready_tasks(64, wait)
                    .await
                    .expect("the cleanup poll should succeed")
                    .into_iter()
                    .map(|entry| entry.job_id),
            );
        }
    })
    .await
    .expect("the expected entries should be published");
    polled
}

/// # Returns
///
/// The hooks of a runtime about to start, with an `Observe` hook at
/// [`HookPoint::RecoveryAfterRepair`] for the tree rooted at `root_id`.
fn observe_repair(root_id: JobId) -> (TestHooks, ArmedHook) {
    let hooks = TestHooks::default();
    let observed = hooks.arm(
        HookPoint::RecoveryAfterRepair,
        HookFilter::root(root_id),
        HookAction::Observe,
        HookCount::Always,
    );
    (hooks, observed)
}

/// # Returns
///
/// The IDs of the jobs that the observed recovery repaired, sorted, or [`None`] if recovery loaded
/// no job of the tree.
fn repaired_jobs(observed: &ArmedHook) -> Option<Vec<JobId>> {
    assert!(observed.hits() <= 1, "a tree is recovered once per start");
    let mut job_ids = observed.last_context()?.job_ids;
    job_ids.sort_by_key(JobId::get);
    Some(job_ids)
}

/// Restarts `env`, letting `corrupt` write the stopped database first.
///
/// # Returns
///
/// The restarted environment, and [`repaired_jobs`] of the tree rooted at `root_id` in its
/// recovery.
async fn restart_observing_repair(
    env: TestEnv,
    root_id: JobId,
    corrupt: &[(JobId, JobState)],
) -> (TestEnv, Option<Vec<JobId>>) {
    let stopped: StoppedEnv = env.stop().await;
    for (job_id, state) in corrupt {
        stopped.set_row_state(*job_id, *state).await;
    }
    let (hooks, observed) = observe_repair(root_id);
    let env = stopped.start(hooks).await;
    let repaired = repaired_jobs(&observed);
    (env, repaired)
}

/// Arms a crash at `point` for the rounds of `event_kind` of the tree rooted at `root_id`.
fn arm_crash(
    env: &TestEnv,
    point: HookPoint,
    root_id: JobId,
    event_kind: EventKindTag,
) -> ArmedHook {
    env.hooks.arm(
        point,
        HookFilter {
            root_id: Some(root_id),
            event_kind: Some(event_kind),
            ..HookFilter::default()
        },
        HookAction::Crash,
        HookCount::Once,
    )
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_crash_before_a_registration_commits_reruns_the_parent() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, PLAIN).await;
    let _crash = arm_crash(
        &env,
        HookPoint::CommitterBeforeDbTransaction,
        root_id,
        EventKindTag::RegularTasksDone,
    );
    assert!(run_task(&env, root_id, &[PLAIN; 3]).await.is_err());
    assert!(env.is_fatal());

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, Some(Vec::new()));
    assert_states(&env, &[(root_id, JobState::Running)]).await;
    assert_eq!(children_of(&env, root_id).await, []);
    assert_eq!(env.poll_ready_tasks(root_id, 1).await, [0]);

    let child_ids = register_children(&env, root_id, [PLAIN; 3]).await;
    for child_id in &child_ids[..2] {
        assert_eq!(run_task(&env, *child_id, &[]).await?, JobState::Succeeded);
    }
    assert_states(&env, &[(root_id, JobState::BlockingOnChildren)]).await;
    assert_eq!(
        run_task(&env, child_ids[2], &[]).await?,
        JobState::Succeeded
    );
    assert_states(&env, &[(root_id, JobState::Succeeded)]).await;
    assert_eq!(children_of(&env, root_id).await, child_ids);
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_crash_after_a_registration_commits_recovers_the_blocked_parent() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, WITH_COMMIT).await;
    let _crash = arm_crash(
        &env,
        HookPoint::CommitterAfterCommit,
        root_id,
        EventKindTag::RegularTasksDone,
    );
    assert!(run_task(&env, root_id, &[PLAIN; 3]).await.is_err());
    assert!(env.is_fatal());

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, Some(Vec::new()));
    let child_ids = children_of(&env, root_id).await;
    assert_eq!(child_ids.len(), 3);
    let mut expected = vec![(root_id, JobState::BlockingOnChildren)];
    expected.extend(
        child_ids
            .iter()
            .map(|child_id| (*child_id, JobState::Running)),
    );
    assert_states(&env, &expected).await;
    assert!(
        matches!(
            env.service.get_job_state(child_ids[0]).await,
            Err(StorageServerError::BadRequest(_))
        ),
        "a recovered child is not a root"
    );

    let polled = poll_lanes(
        &env,
        &LaneEntries {
            ready: child_ids.iter().copied().collect(),
            ..LaneEntries::default()
        },
    )
    .await;
    assert_eq!(
        polled,
        LaneEntries {
            ready: child_ids.iter().copied().collect(),
            ..LaneEntries::default()
        }
    );

    for child_id in &child_ids {
        assert_eq!(run_task(&env, *child_id, &[]).await?, JobState::Succeeded);
    }
    assert_states(&env, &[(root_id, JobState::CommitReady)]).await;
    assert_eq!(run_commit(&env, root_id).await, JobState::Succeeded);
    assert_states(&env, &[(root_id, JobState::Succeeded)]).await;
    assert_eq!(children_of(&env, root_id).await, child_ids);
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_crash_in_the_middle_of_a_cascade_recovers_what_it_committed() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, WITH_COMMIT).await;
    let [parent_id] = register_children(&env, root_id, [PLAIN]).await;
    let [leaf_id] = register_children(&env, parent_id, [PLAIN]).await;
    let _crash = env.hooks.arm(
        HookPoint::CommitterAfterApplyJob,
        HookFilter::job(leaf_id),
        HookAction::Crash,
        HookCount::Once,
    );
    assert!(run_task(&env, leaf_id, &[]).await.is_err());
    assert!(env.is_fatal());

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, Some(Vec::new()));
    assert_states(
        &env,
        &[
            (root_id, JobState::CommitReady),
            (parent_id, JobState::Succeeded),
            (leaf_id, JobState::Succeeded),
        ],
    )
    .await;
    let polled = poll_lanes(
        &env,
        &LaneEntries {
            commit: HashSet::from([root_id]),
            ..LaneEntries::default()
        },
    )
    .await;
    assert_eq!(polled.ready, HashSet::new());
    assert_eq!(run_commit(&env, root_id).await, JobState::Succeeded);
    assert_states(&env, &[(root_id, JobState::Succeeded)]).await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_crash_after_a_cancel_commits_recovers_only_cleanups_and_skipped_commits()
-> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, WITH_CLEANUP).await;
    let [parent_id, skipped_id] =
        register_children(&env, root_id, [WITH_CLEANUP, WITH_COMMIT]).await;
    let [leaf_id] = register_children(&env, parent_id, [WITH_CLEANUP]).await;
    assert_eq!(
        run_task(&env, skipped_id, &[]).await?,
        JobState::CommitReady
    );
    let _crash = arm_crash(
        &env,
        HookPoint::CommitterAfterCommit,
        root_id,
        EventKindTag::CancelRoot,
    );
    assert!(env.service.cancel_job(root_id).await.is_err());
    assert!(env.is_fatal());

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, Some(Vec::new()));
    assert_states(
        &env,
        &[
            (root_id, JobState::CleanupReady),
            (parent_id, JobState::CleanupReady),
            (leaf_id, JobState::CleanupReady),
            (skipped_id, JobState::CommitReady),
        ],
    )
    .await;
    let expected = LaneEntries {
        ready: HashSet::new(),
        commit: HashSet::from([skipped_id]),
        cleanup: HashSet::from([root_id, parent_id, leaf_id]),
    };
    assert_eq!(poll_lanes(&env, &expected).await, expected);

    for job_id in [leaf_id, parent_id, root_id] {
        assert_eq!(run_cleanup(&env, job_id).await, JobState::Cancelled);
    }
    assert_eq!(run_commit(&env, skipped_id).await, JobState::Succeeded);
    assert_states(
        &env,
        &[
            (root_id, JobState::Cancelled),
            (parent_id, JobState::Cancelled),
            (leaf_id, JobState::Cancelled),
            (skipped_id, JobState::Succeeded),
        ],
    )
    .await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_crash_during_recovery_repairs_the_tree_once() -> anyhow::Result<()> {
    for (point, repaired_by_second_start) in [
        (HookPoint::RecoveryBeforeRepair, true),
        (HookPoint::RecoveryAfterRepair, false),
    ] {
        let env = TestEnv::start().await;
        let root_id = start_root(&env, PLAIN).await;
        let child_ids = register_children(&env, root_id, [PLAIN; 2]).await;
        let stopped = env.stop().await;
        for child_id in &child_ids {
            stopped.set_row_state(*child_id, JobState::Succeeded).await;
        }

        let crashing_hooks = TestHooks::default();
        let _crash = crashing_hooks.arm(
            point,
            HookFilter::root(root_id),
            HookAction::Crash,
            HookCount::Once,
        );
        stopped.start_failing(crashing_hooks).await;
        let (hooks, observed) = observe_repair(root_id);
        let env = stopped.start(hooks).await;
        let expected_repair = if repaired_by_second_start {
            Some(vec![root_id])
        } else {
            None
        };
        assert_eq!(repaired_jobs(&observed), expected_repair, "{point:?}");
        assert_states(&env, &[(root_id, JobState::Succeeded)]).await;
        assert_eq!(
            env.service.get_job_state(root_id).await?,
            JobState::Succeeded
        );

        let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
        assert_eq!(repaired, None);
        env.finish().await;
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_superseded_runtime_is_fenced_off_and_its_successor_matches_the_database()
-> anyhow::Result<()> {
    let mut env = TestEnv::start().await;
    let root_id = start_root(&env, PLAIN).await;
    let child_ids = register_children(&env, root_id, [PLAIN; 2]).await;
    let old_instance_id = instance(&env, child_ids[0], TaskId::Index(0)).await;

    let (old_runtime, old_service) = env.start_successor().await;
    let report = old_service
        .succeed_task_instance(
            old_service.session_id(),
            child_ids[0],
            old_instance_id,
            0,
            TaskOutputsSerializer::from_tuple(&(vec![1u8; 4],))?,
            Vec::new(),
        )
        .await;
    assert!(report.is_err());
    assert!(old_service.shutdown().is_fatal());
    drop(old_service);
    let _ = old_runtime.stop().await;

    assert_states(
        &env,
        &[
            (root_id, JobState::BlockingOnChildren),
            (child_ids[0], JobState::Running),
            (child_ids[1], JobState::Running),
        ],
    )
    .await;
    let expected = LaneEntries {
        ready: child_ids.iter().copied().collect(),
        ..LaneEntries::default()
    };
    assert_eq!(poll_lanes(&env, &expected).await, expected);
    for child_id in &child_ids {
        assert_eq!(run_task(&env, *child_id, &[]).await?, JobState::Succeeded);
    }
    assert_states(&env, &[(root_id, JobState::Succeeded)]).await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn startup_aborts_an_orphan() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, PLAIN).await;
    let [orphan_id] = register_children(&env, root_id, [WITH_CLEANUP]).await;

    let (env, repaired) =
        restart_observing_repair(env, root_id, &[(root_id, JobState::Failed)]).await;
    assert_eq!(repaired, Some(vec![orphan_id]));
    assert_states(&env, &[(orphan_id, JobState::CleanupReady)]).await;

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, Some(Vec::new()));
    let expected = LaneEntries {
        cleanup: HashSet::from([orphan_id]),
        ..LaneEntries::default()
    };
    assert_eq!(poll_lanes(&env, &expected).await, expected);
    assert_eq!(run_cleanup(&env, orphan_id).await, JobState::Cancelled);
    assert_states(&env, &[(orphan_id, JobState::Cancelled)]).await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn startup_unblocks_a_stuck_parent_and_climbs() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, PLAIN).await;
    let [parent_id] = register_children(&env, root_id, [PLAIN]).await;
    let [leaf_id] = register_children(&env, parent_id, [PLAIN]).await;

    let (env, repaired) =
        restart_observing_repair(env, root_id, &[(leaf_id, JobState::Succeeded)]).await;
    assert_eq!(repaired, Some(vec![root_id, parent_id]));
    assert_states(
        &env,
        &[
            (root_id, JobState::Succeeded),
            (parent_id, JobState::Succeeded),
        ],
    )
    .await;

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, None);
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn startup_unblocks_a_stuck_parent_with_a_commit_task_to_commit_ready() -> anyhow::Result<()>
{
    let env = TestEnv::start().await;
    let root_id = start_root(&env, WITH_COMMIT).await;
    let [child_id] = register_children(&env, root_id, [PLAIN]).await;

    let (env, repaired) =
        restart_observing_repair(env, root_id, &[(child_id, JobState::Succeeded)]).await;
    assert_eq!(repaired, Some(vec![root_id]));
    assert_states(&env, &[(root_id, JobState::CommitReady)]).await;

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, Some(Vec::new()));
    let expected = LaneEntries {
        commit: HashSet::from([root_id]),
        ..LaneEntries::default()
    };
    assert_eq!(poll_lanes(&env, &expected).await, expected);
    assert_eq!(run_commit(&env, root_id).await, JobState::Succeeded);
    assert_states(&env, &[(root_id, JobState::Succeeded)]).await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn startup_fails_a_blocked_parent_with_a_failed_child() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, PLAIN).await;
    let [failed_id, running_id] = register_children(&env, root_id, [PLAIN, WITH_CLEANUP]).await;

    let (env, repaired) =
        restart_observing_repair(env, root_id, &[(failed_id, JobState::Failed)]).await;
    assert_eq!(repaired, Some(vec![root_id, running_id]));
    assert_states(
        &env,
        &[
            (root_id, JobState::Failed),
            (running_id, JobState::CleanupReady),
        ],
    )
    .await;
    assert_eq!(
        env.row_error(root_id).await,
        "tree repair: a child job failed or was cancelled"
    );

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, Some(Vec::new()));
    assert_eq!(run_cleanup(&env, running_id).await, JobState::Cancelled);
    assert_states(&env, &[(running_id, JobState::Cancelled)]).await;
    env.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn startup_fails_a_tree_that_repair_cannot_fix() -> anyhow::Result<()> {
    let env = TestEnv::start().await;
    let root_id = start_root(&env, PLAIN).await;
    let [ready_id, running_id] = register_children(&env, root_id, [PLAIN; 2]).await;

    let (env, repaired) =
        restart_observing_repair(env, root_id, &[(ready_id, JobState::Ready)]).await;
    assert_eq!(repaired, Some(vec![root_id, ready_id, running_id]));
    assert_states(
        &env,
        &[
            (root_id, JobState::Failed),
            (ready_id, JobState::Cancelled),
            (running_id, JobState::Failed),
        ],
    )
    .await;
    let expected_error = format!(
        "tree invariant violation: invariant I1 broken at job {ready_id}: a non-root job is Ready"
    );
    assert_eq!(env.row_error(root_id).await, expected_error);
    assert_eq!(env.row_error(running_id).await, expected_error);
    assert_eq!(env.service.get_job_state(root_id).await?, JobState::Failed);

    let (env, repaired) = restart_observing_repair(env, root_id, &[]).await;
    assert_eq!(repaired, None);
    env.finish().await;
    Ok(())
}
