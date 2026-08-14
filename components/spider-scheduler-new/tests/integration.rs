//! Full-stack integration scenarios for the prototype scheduler.
//!
//! Every scenario drives a complete stack through [`Harness`]: the core on its own dedicated
//! thread, the gRPC dispatch service in front of it, and gRPC workers pulling assignments. Nothing
//! here reaches into the core's state, so a scenario observes exactly what a deployed execution
//! manager would.
//!
//! The assertions are on observed behaviour -- which assignment reached which worker, and in what
//! mint order -- rather than on elapsed time, so the suite stays meaningful on a loaded machine.
//! The core *mints* an assignment when it stamps a globally unique ID onto it at publication, so
//! sorting the workers' records by that ID recovers the core's decision order, even though the
//! workers received those assignments concurrently and out of order.

use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::time::Duration;

use spider_scheduler_new::CoreConfig;
use spider_scheduler_new::harness::DispatchRecord;
use spider_scheduler_new::harness::FakeStorageConfig;
use spider_scheduler_new::harness::FakeWorkerConfig;
use spider_scheduler_new::harness::Harness;
use spider_scheduler_new::harness::HarnessConfig;
use spider_scheduler_new::harness::WorkerReport;
use spider_scheduler_new::types::JobId;
use spider_scheduler_new::types::ResourceGroupId;
use spider_scheduler_new::types::TaskId;

/// The tick cadence every scenario runs the core at, in milliseconds.
const TICK_INTERVAL_MS: u64 = 1;

/// The inbound-queue poll timeout every scenario runs the core with, in milliseconds.
const STORAGE_POLL_TIMEOUT_MS: u64 = 5;

/// The per-lane inbound fetch bound for the finalization lanes.
const FINALIZE_TASK_CAPACITY: usize = 64;

/// The long-poll wait every worker asks the scheduler to hold a request for, in milliseconds.
const NEXT_TASK_WAIT_MS: u64 = 50;

/// The identity of the first general worker, past the range the pinned workers name themselves by.
const GENERAL_WORKER_ID_BASE: usize = 100;

/// The execution time of a worker that is not deliberately slowed down, in milliseconds.
const FAST_TASK_DURATION_MS: u64 = 1;

/// The deadline for a scenario running at reduced dimensions.
const SMALL_SCENARIO_TIMEOUT: Duration = Duration::from_mins(1);

/// The deadline for the full-scale scenario.
const FULL_SCALE_TIMEOUT: Duration = Duration::from_mins(5);

/// The number of entries a diagnostic message lists before it summarizes the rest.
const MAX_REPORTED_ENTRIES: usize = 16;

/// Verifies the coverage invariant end to end: general workers alone reach every resource group,
/// and every task is dispatched exactly once.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`Harness::start`]'s return values on failure.
/// * Forwards [`Harness::run_until_drained`]'s return values on failure.
/// * Forwards [`check_exactly_once`]'s return values on failure.
#[tokio::test(flavor = "multi_thread")]
async fn general_workers_alone_drain_every_resource_group() -> anyhow::Result<()> {
    let num_resource_groups = 4;
    let workers = (0..4)
        .map(|worker_index| general_worker(worker_index, FAST_TASK_DURATION_MS))
        .collect::<Vec<FakeWorkerConfig>>();
    let harness = Harness::start(HarnessConfig {
        core: core_config(64, 256, 4),
        storage: storage_config(num_resource_groups, 4, 16, false),
        workers,
    })
    .await?;

    let outcome = harness.run_until_drained(SMALL_SCENARIO_TIMEOUT).await?;

    check_exactly_once(&outcome.reports, &outcome.storage.expected_regular_tasks())?;
    assert_eq!(
        served_resource_groups(&outcome.reports),
        all_resource_groups(num_resource_groups)
    );
    assert!(
        outcome.storage.is_drained(),
        "the workload did not drain within the timeout"
    );

    Ok(())
}

/// Verifies that a pinned worker receives assignments from its own resource group and from no
/// other, while the workload as a whole is still dispatched exactly once.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`Harness::start`]'s return values on failure.
/// * Forwards [`Harness::run_until_drained`]'s return values on failure.
/// * Forwards [`check_exactly_once`]'s return values on failure.
/// * Forwards [`check_pinned_isolation`]'s return values on failure.
#[tokio::test(flavor = "multi_thread")]
async fn pinned_workers_receive_only_their_own_resource_group() -> anyhow::Result<()> {
    let num_resource_groups = 4;
    let num_jobs_per_group = 4;
    let num_tasks_per_job = 16;
    let workers = (0..4)
        .map(|group_index| pinned_worker(group_index, FAST_TASK_DURATION_MS))
        .collect::<Vec<FakeWorkerConfig>>();
    let harness = Harness::start(HarnessConfig {
        core: core_config(64, 256, 4),
        storage: storage_config(
            num_resource_groups,
            num_jobs_per_group,
            num_tasks_per_job,
            false,
        ),
        workers,
    })
    .await?;

    let outcome = harness.run_until_drained(SMALL_SCENARIO_TIMEOUT).await?;

    check_exactly_once(&outcome.reports, &outcome.storage.expected_regular_tasks())?;
    check_pinned_isolation(&outcome.reports)?;
    for report in &outcome.reports {
        assert_eq!(
            report.dispatches.len(),
            num_jobs_per_group * num_tasks_per_job,
            "worker {} did not receive its group's whole workload",
            report.execution_manager_id
        );
    }
    assert!(
        outcome.storage.is_drained(),
        "the workload did not drain within the timeout"
    );

    Ok(())
}

/// Verifies the case the hint scheme exists for: with pinned workers covering only some resource
/// groups, the groups without one are still fully drained by the general workers.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`Harness::start`]'s return values on failure.
/// * Forwards [`Harness::run_until_drained`]'s return values on failure.
/// * Forwards [`check_exactly_once`]'s return values on failure.
/// * Forwards [`check_pinned_isolation`]'s return values on failure.
#[tokio::test(flavor = "multi_thread")]
async fn unpinned_resource_groups_drain_through_general_workers() -> anyhow::Result<()> {
    let num_resource_groups = 6;
    let num_pinned_groups = 3;
    let mut workers: Vec<FakeWorkerConfig> = (0..num_pinned_groups)
        .map(|group_index| pinned_worker(group_index, FAST_TASK_DURATION_MS))
        .collect();
    workers.extend((0..3).map(|worker_index| general_worker(worker_index, FAST_TASK_DURATION_MS)));
    let harness = Harness::start(HarnessConfig {
        core: core_config(64, 384, 4),
        storage: storage_config(num_resource_groups, 4, 16, false),
        workers,
    })
    .await?;

    let outcome = harness.run_until_drained(SMALL_SCENARIO_TIMEOUT).await?;

    check_exactly_once(&outcome.reports, &outcome.storage.expected_regular_tasks())?;
    check_pinned_isolation(&outcome.reports)?;
    let unpinned_groups: HashSet<ResourceGroupId> = (num_pinned_groups..num_resource_groups)
        .map(resource_group_id)
        .collect();
    let general_groups: HashSet<ResourceGroupId> = outcome
        .reports
        .iter()
        .filter(|report| report.resource_group_id.is_none())
        .flat_map(|report| report.dispatches.iter())
        .map(|record| record.resource_group_id)
        .collect();
    assert_eq!(
        unpinned_groups
            .difference(&general_groups)
            .copied()
            .collect::<HashSet<ResourceGroupId>>(),
        HashSet::new(),
        "a resource group without a pinned worker was never served"
    );
    assert!(
        outcome.storage.is_drained(),
        "the workload did not drain within the timeout"
    );

    Ok(())
}

/// Verifies that one resource group draining far more slowly than the rest neither blocks the
/// others nor takes over the dispatch buffer.
///
/// The occupancy bound is read off the mint order rather than measured directly. A run of
/// consecutive assignments minted for the slow group is a run in which no other group was served,
/// and the slow worker consumes at most one assignment per `SLOW_TASK_DURATION_MS`, fifty times
/// the core's tick interval; a run of `n` such mints therefore drives the slow group's queue
/// occupancy to nearly `n`. Bounding the longest run by half the dispatch buffer
/// is bounding the group's peak occupancy by the same figure, which is the ceiling the admission
/// policy's `S < F` rule imposes when a single group is backlogged.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`Harness::start`]'s return values on failure.
/// * Forwards [`Harness::run_until_drained`]'s return values on failure.
/// * Forwards [`check_exactly_once`]'s return values on failure.
/// * Forwards [`check_pinned_isolation`]'s return values on failure.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_resource_group_neither_blocks_nor_takes_over_the_buffer() -> anyhow::Result<()> {
    const SLOW_TASK_DURATION_MS: u64 = 50;
    const DISPATCH_QUEUE_CAPACITY: usize = 32;

    let num_resource_groups = 4;
    let num_jobs_per_group = 2;
    let num_tasks_per_job = 24;
    let slow_group = resource_group_id(0);
    let mut workers = vec![pinned_worker(0, SLOW_TASK_DURATION_MS)];
    workers.extend((1..4).map(|group_index| pinned_worker(group_index, FAST_TASK_DURATION_MS)));
    let harness = Harness::start(HarnessConfig {
        core: core_config(DISPATCH_QUEUE_CAPACITY, 256, 4),
        storage: storage_config(
            num_resource_groups,
            num_jobs_per_group,
            num_tasks_per_job,
            false,
        ),
        workers,
    })
    .await?;

    let outcome = harness.run_until_drained(SMALL_SCENARIO_TIMEOUT).await?;

    check_exactly_once(&outcome.reports, &outcome.storage.expected_regular_tasks())?;
    check_pinned_isolation(&outcome.reports)?;
    for report in &outcome.reports {
        assert_eq!(
            report.dispatches.len(),
            num_jobs_per_group * num_tasks_per_job,
            "worker {} did not receive its group's whole workload",
            report.execution_manager_id
        );
    }
    let longest_run = longest_uninterrupted_mint_run(&mint_order(&outcome.reports), slow_group);
    assert!(
        longest_run <= DISPATCH_QUEUE_CAPACITY / 2,
        "the slow resource group held {longest_run} of the {DISPATCH_QUEUE_CAPACITY}-slot \
         dispatch buffer"
    );
    assert!(
        outcome.storage.is_drained(),
        "the workload did not drain within the timeout"
    );

    Ok(())
}

/// Verifies exactly-once dispatch at the design's scale target: 16 resource groups of 32 jobs of
/// 128 tasks, drained by a mix of pinned and general workers.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`Harness::start`]'s return values on failure.
/// * Forwards [`Harness::run_until_drained`]'s return values on failure.
/// * Forwards [`check_exactly_once`]'s return values on failure.
/// * Forwards [`check_pinned_isolation`]'s return values on failure.
#[tokio::test(flavor = "multi_thread")]
async fn the_full_scale_workload_is_dispatched_exactly_once() -> anyhow::Result<()> {
    let num_resource_groups = 16;
    let mut workers: Vec<FakeWorkerConfig> = (0..16)
        .map(|group_index| pinned_worker(group_index, FAST_TASK_DURATION_MS))
        .collect();
    workers.extend((0..16).map(|worker_index| general_worker(worker_index, FAST_TASK_DURATION_MS)));
    let harness = Harness::start(HarnessConfig {
        core: core_config(1024, 4096, 8),
        storage: storage_config(num_resource_groups, 32, 128, false),
        workers,
    })
    .await?;

    let outcome = harness.run_until_drained(FULL_SCALE_TIMEOUT).await?;

    check_exactly_once(&outcome.reports, &outcome.storage.expected_regular_tasks())?;
    check_pinned_isolation(&outcome.reports)?;
    assert_eq!(
        served_resource_groups(&outcome.reports),
        all_resource_groups(num_resource_groups)
    );
    assert_eq!(outcome.storage.total_regular_tasks(), 65536);
    assert!(
        outcome.storage.is_drained(),
        "the workload did not drain within the timeout"
    );

    Ok(())
}

/// Verifies that a job's finalization task -- commit for some jobs, cleanup for others -- is
/// dispatched exactly once, and only after every one of that job's regular tasks has been
/// dispatched.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`Harness::start`]'s return values on failure.
/// * Forwards [`Harness::run_until_drained`]'s return values on failure.
/// * Forwards [`check_exactly_once`]'s return values on failure.
/// * Forwards [`check_finalization_tasks`]'s return values on failure.
#[tokio::test(flavor = "multi_thread")]
async fn every_finalization_task_follows_its_job_and_is_dispatched_once() -> anyhow::Result<()> {
    let num_resource_groups = 4;
    let num_jobs_per_group = 4;
    let mut workers: Vec<FakeWorkerConfig> = (0..2)
        .map(|group_index| pinned_worker(group_index, FAST_TASK_DURATION_MS))
        .collect();
    workers.extend((0..2).map(|worker_index| general_worker(worker_index, FAST_TASK_DURATION_MS)));
    let harness = Harness::start(HarnessConfig {
        core: core_config(64, 256, 4),
        storage: storage_config(num_resource_groups, num_jobs_per_group, 16, true),
        workers,
    })
    .await?;

    let outcome = harness.run_until_drained(SMALL_SCENARIO_TIMEOUT).await?;

    let expected_finalizations = outcome.storage.expected_finalization_tasks();
    assert_eq!(
        expected_finalizations.len(),
        num_resource_groups * num_jobs_per_group
    );
    assert!(
        expected_finalizations
            .iter()
            .any(|(_, task_id)| TaskId::Cleanup == *task_id),
        "the scenario must exercise the cleanup-ready lane as well as the commit-ready one"
    );
    check_exactly_once(&outcome.reports, &outcome.storage.expected_regular_tasks())?;
    check_finalization_tasks(&outcome.reports, &expected_finalizations)?;
    assert!(
        outcome.storage.is_drained(),
        "the workload did not drain within the timeout"
    );

    Ok(())
}

/// Verifies that a session bump in the middle of a run does not cost the workload a task: storage
/// replays everything it has not seen completed, the core re-admits the replay, and the run still
/// drains.
///
/// Exactly-once cannot hold across a bump and is deliberately not asserted. Storage re-arms every
/// task it has not seen completed, so anything the pre-bump pipeline still held is dispatched a
/// second time. What that pipeline held is bounded by construction: at most
/// `ready_task_capacity` tasks buffered in the core, at most `dispatch_queue_capacity`
/// assignments published into the per-group queues, and one in-flight assignment per worker.
/// Nothing can be added to it in the meantime, because the first entries storage serves after the
/// bump carry the new session and are what makes the core flush. A task dispatched more than twice,
/// or more replayed tasks than that bound, would mean the flush failed to fence the old session
/// off.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`anyhow::Error`] if the session-bumping task could not be joined.
/// * Forwards [`Harness::start`]'s return values on failure.
/// * Forwards [`Harness::run_until_drained`]'s return values on failure.
/// * Forwards [`check_coverage`]'s return values on failure.
#[tokio::test(flavor = "multi_thread")]
async fn a_mid_run_session_bump_replays_every_unfinished_task() -> anyhow::Result<()> {
    const BUMP_DELAY: Duration = Duration::from_millis(200);
    const BUMP_SCENARIO_TASK_DURATION_MS: u64 = 3;
    const DISPATCH_QUEUE_CAPACITY: usize = 32;
    const READY_TASK_CAPACITY: usize = 64;

    let num_workers = 4;
    let mut workers: Vec<FakeWorkerConfig> = (0..2)
        .map(|group_index| pinned_worker(group_index, BUMP_SCENARIO_TASK_DURATION_MS))
        .collect();
    workers.extend(
        (0..2).map(|worker_index| general_worker(worker_index, BUMP_SCENARIO_TASK_DURATION_MS)),
    );
    let harness = Harness::start(HarnessConfig {
        core: core_config(DISPATCH_QUEUE_CAPACITY, READY_TASK_CAPACITY, 4),
        storage: storage_config(4, 4, 64, false),
        workers,
    })
    .await?;

    let storage = harness.storage();
    let bump_task = tokio::spawn(async move {
        tokio::time::sleep(BUMP_DELAY).await;
        let drained_before_bump = storage.is_drained();
        (drained_before_bump, storage.bump_session())
    });
    let outcome = harness.run_until_drained(SMALL_SCENARIO_TIMEOUT).await?;
    let (drained_before_bump, new_session_id) = bump_task.await?;

    assert!(
        !drained_before_bump,
        "the bump landed after the run drained"
    );
    assert_eq!(new_session_id, 1);
    check_coverage(&outcome.reports, &outcome.storage.expected_regular_tasks())?;
    let duplicated = duplicated_dispatches(&outcome.reports);
    let max_replayed = READY_TASK_CAPACITY + DISPATCH_QUEUE_CAPACITY + num_workers;
    assert!(
        duplicated.len() <= max_replayed,
        "{} tasks were dispatched more than once across a single session bump, which exceeds the \
         {max_replayed} the pre-bump pipeline could have held: {}",
        duplicated.len(),
        format_duplicates(&duplicated)
    );
    let dispatched_thrice: Vec<((JobId, TaskId), usize)> = duplicated
        .iter()
        .filter(|(_, num_dispatches)| 2 < *num_dispatches)
        .copied()
        .collect();
    assert_eq!(
        format_duplicates(&dispatched_thrice),
        "none",
        "a single session bump can replay a task at most once"
    );
    assert!(
        outcome.storage.is_drained(),
        "the workload did not drain within the timeout"
    );

    Ok(())
}

/// Asserts that the regular tasks the workers received are exactly the tasks the workload
/// describes, each of them once.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`anyhow::Error`] if a task was never dispatched, was dispatched more than once, or was
///   dispatched without belonging to the workload. The message names the offending tasks, since a
///   count alone is not diagnosable at the full scale.
fn check_exactly_once(
    reports: &[WorkerReport],
    expected: &HashSet<(JobId, TaskId)>,
) -> anyhow::Result<()> {
    let counts = regular_dispatch_counts(reports);
    let missing = sorted_tasks(
        expected
            .iter()
            .filter(|task| !counts.contains_key(*task))
            .copied(),
    );
    let unexpected = sorted_tasks(
        counts
            .keys()
            .filter(|task| !expected.contains(*task))
            .copied(),
    );
    let duplicated = duplicated_dispatches(reports);
    if missing.is_empty() && unexpected.is_empty() && duplicated.is_empty() {
        return Ok(());
    }

    let num_dispatched: usize = counts.values().sum();
    let num_expected = expected.len();
    let missing_report = format_tasks(&missing);
    let duplicated_report = format_duplicates(&duplicated);
    let unexpected_report = format_tasks(&unexpected);
    anyhow::bail!(
        "exactly-once dispatch violated: {num_dispatched} regular dispatches against \
         {num_expected} expected tasks\n  missing: {missing_report}\n  duplicated: \
         {duplicated_report}\n  unexpected: {unexpected_report}"
    );
}

/// Asserts that every task the workload describes was dispatched at least once and that nothing
/// outside the workload was dispatched.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`anyhow::Error`] if a task was never dispatched or a dispatched task does not belong to the
///   workload.
fn check_coverage(
    reports: &[WorkerReport],
    expected: &HashSet<(JobId, TaskId)>,
) -> anyhow::Result<()> {
    let counts = regular_dispatch_counts(reports);
    let missing = sorted_tasks(
        expected
            .iter()
            .filter(|task| !counts.contains_key(*task))
            .copied(),
    );
    let unexpected = sorted_tasks(
        counts
            .keys()
            .filter(|task| !expected.contains(*task))
            .copied(),
    );
    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }

    let num_expected = expected.len();
    let missing_report = format_tasks(&missing);
    let unexpected_report = format_tasks(&unexpected);
    anyhow::bail!(
        "dispatch coverage violated against {num_expected} expected tasks\n  missing: \
         {missing_report}\n  unexpected: {unexpected_report}"
    );
}

/// Asserts that no pinned worker received an assignment belonging to another resource group.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`anyhow::Error`] if a pinned worker received an assignment from a resource group other than
///   its own.
fn check_pinned_isolation(reports: &[WorkerReport]) -> anyhow::Result<()> {
    for report in reports {
        let Some(pinned_group) = report.resource_group_id else {
            continue;
        };
        let num_foreign = report
            .dispatches
            .iter()
            .filter(|record| record.resource_group_id != pinned_group)
            .count();
        if 0 == num_foreign {
            continue;
        }

        let worker_id = report.execution_manager_id;
        let foreign_groups: HashSet<ResourceGroupId> = report
            .dispatches
            .iter()
            .map(|record| record.resource_group_id)
            .filter(|rg_id| *rg_id != pinned_group)
            .collect();
        anyhow::bail!(
            "worker {worker_id} pinned to resource group {pinned_group} received {num_foreign} \
             assignments from resource groups {foreign_groups:?}"
        );
    }

    Ok(())
}

/// Asserts that every finalization task in `expected` was dispatched exactly once, after every
/// regular task of its job had been minted, and that no other finalization task was dispatched.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`anyhow::Error`] if a finalization task was never dispatched, was dispatched more than once,
///   was dispatched for a job the workload never finalized, carries a kind other than the one the
///   workload armed, or carries an assignment ID older than one of its job's regular tasks.
fn check_finalization_tasks(
    reports: &[WorkerReport],
    expected: &HashSet<(JobId, TaskId)>,
) -> anyhow::Result<()> {
    let mut num_finalizations: HashMap<(JobId, TaskId), usize> = HashMap::new();
    let mut finalization_mint: HashMap<JobId, u64> = HashMap::new();
    let mut last_regular_mint: HashMap<JobId, u64> = HashMap::new();
    for record in reports.iter().flat_map(|report| report.dispatches.iter()) {
        let mint = record.assignment_id.get();
        if matches!(record.task_id, TaskId::Index(_)) {
            let last = last_regular_mint.entry(record.job_id).or_default();
            *last = (*last).max(mint);
            continue;
        }
        *num_finalizations
            .entry((record.job_id, record.task_id))
            .or_default() += 1;
        let earliest = finalization_mint.entry(record.job_id).or_insert(mint);
        *earliest = (*earliest).min(mint);
    }

    let unexpected = sorted_tasks(
        num_finalizations
            .keys()
            .filter(|task| !expected.contains(*task))
            .copied(),
    );
    if !unexpected.is_empty() {
        let unexpected_report = format_tasks(&unexpected);
        anyhow::bail!("finalization tasks the workload never armed: {unexpected_report}");
    }

    for task in expected {
        let (job_id, task_id) = *task;
        let num_dispatched = num_finalizations.get(task).copied().unwrap_or_default();
        if 1 != num_dispatched {
            anyhow::bail!("job {job_id}'s {task_id} task was dispatched {num_dispatched} times");
        }
        let finalization_mint = finalization_mint.get(&job_id).copied().unwrap_or_default();
        let last_regular_mint = last_regular_mint.get(&job_id).copied().unwrap_or_default();
        if finalization_mint <= last_regular_mint {
            anyhow::bail!(
                "job {job_id}'s {task_id} task was minted at {finalization_mint}, ahead of its \
                 regular task minted at {last_regular_mint}"
            );
        }
    }

    Ok(())
}

/// # Returns
///
/// How many times each of the workload's regular tasks was dispatched, over every worker.
fn regular_dispatch_counts(reports: &[WorkerReport]) -> HashMap<(JobId, TaskId), usize> {
    let mut counts: HashMap<(JobId, TaskId), usize> = HashMap::new();
    for record in reports.iter().flat_map(|report| report.dispatches.iter()) {
        if !matches!(record.task_id, TaskId::Index(_)) {
            continue;
        }
        *counts.entry((record.job_id, record.task_id)).or_default() += 1;
    }

    counts
}

/// # Returns
///
/// Every regular task that was dispatched more than once, with its dispatch count, ordered by job
/// and task.
fn duplicated_dispatches(reports: &[WorkerReport]) -> Vec<((JobId, TaskId), usize)> {
    let counts = regular_dispatch_counts(reports);
    let mut duplicated: Vec<((JobId, TaskId), usize)> = counts
        .into_iter()
        .filter(|(_, num_dispatches)| 1 < *num_dispatches)
        .collect();
    duplicated.sort_unstable_by_key(|(task, _)| task_sort_key(task));

    duplicated
}

/// # Returns
///
/// Every dispatched assignment of every worker, ordered by the assignment ID the core minted it
/// with, which is the order the core published them in.
fn mint_order(reports: &[WorkerReport]) -> Vec<&DispatchRecord> {
    let mut records: Vec<&DispatchRecord> = reports
        .iter()
        .flat_map(|report| report.dispatches.iter())
        .collect();
    records.sort_unstable_by_key(|record| record.assignment_id.get());

    records
}

/// Measures how far `resource_group_id` ever got ahead of the rest of the workload.
///
/// The search is confined to the mints made before the last mint of any other resource group, so a
/// group's uncontested tail -- everything it publishes once the other groups are finished -- does
/// not count against it.
///
/// # Returns
///
/// The longest run of consecutive assignments minted for `resource_group_id` with no other
/// resource group's assignment in between.
fn longest_uninterrupted_mint_run(
    mint_order: &[&DispatchRecord],
    resource_group_id: ResourceGroupId,
) -> usize {
    let Some(last_other_index) = mint_order
        .iter()
        .rposition(|record| record.resource_group_id != resource_group_id)
    else {
        return mint_order.len();
    };

    let mut longest_run = 0;
    let mut current_run = 0;
    for record in &mint_order[..last_other_index] {
        if record.resource_group_id == resource_group_id {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 0;
        }
    }

    longest_run
}

/// # Returns
///
/// Every resource group that appears in at least one dispatched assignment.
fn served_resource_groups(reports: &[WorkerReport]) -> HashSet<ResourceGroupId> {
    reports
        .iter()
        .flat_map(|report| report.dispatches.iter())
        .map(|record| record.resource_group_id)
        .collect()
}

/// # Returns
///
/// The IDs of the `num_resource_groups` resource groups a workload of that width spans.
fn all_resource_groups(num_resource_groups: usize) -> HashSet<ResourceGroupId> {
    (0..num_resource_groups).map(resource_group_id).collect()
}

/// # Returns
///
/// `tasks`, ordered by job and then by task.
fn sorted_tasks(tasks: impl Iterator<Item = (JobId, TaskId)>) -> Vec<(JobId, TaskId)> {
    let mut tasks: Vec<(JobId, TaskId)> = tasks.collect();
    tasks.sort_unstable_by_key(task_sort_key);

    tasks
}

/// # Returns
///
/// A totally ordered key over a job's tasks, placing the regular tasks in index order ahead of the
/// finalization tasks.
fn task_sort_key(task: &(JobId, TaskId)) -> (u64, u8, u64) {
    let (job_id, task_id) = *task;
    let (kind, index) = match task_id {
        TaskId::Index(task_index) => (0, u64::try_from(task_index).unwrap_or(u64::MAX)),
        TaskId::Commit => (1, 0),
        TaskId::Cleanup => (2, 0),
    };

    (job_id.get(), kind, index)
}

/// # Returns
///
/// `tasks` rendered as a diagnostic, listing at most [`MAX_REPORTED_ENTRIES`] of them.
fn format_tasks(tasks: &[(JobId, TaskId)]) -> String {
    if tasks.is_empty() {
        return "none".to_owned();
    }

    let num_tasks = tasks.len();
    let listed = tasks
        .iter()
        .take(MAX_REPORTED_ENTRIES)
        .map(|(job_id, task_id)| format!("(job {job_id}, task {task_id})"))
        .collect::<Vec<String>>()
        .join(", ");

    format!("{num_tasks} [{listed}{}]", format_elision(num_tasks))
}

/// # Returns
///
/// `duplicated` rendered as a diagnostic, listing at most [`MAX_REPORTED_ENTRIES`] of the tasks
/// with the number of times each was dispatched.
fn format_duplicates(duplicated: &[((JobId, TaskId), usize)]) -> String {
    if duplicated.is_empty() {
        return "none".to_owned();
    }

    let num_tasks = duplicated.len();
    let listed = duplicated
        .iter()
        .take(MAX_REPORTED_ENTRIES)
        .map(|((job_id, task_id), num_dispatches)| {
            format!("(job {job_id}, task {task_id}) x{num_dispatches}")
        })
        .collect::<Vec<String>>()
        .join(", ");

    format!("{num_tasks} [{listed}{}]", format_elision(num_tasks))
}

/// # Returns
///
/// The trailing note a diagnostic listing `num_entries` entries needs, which is empty unless the
/// listing was truncated.
fn format_elision(num_entries: usize) -> String {
    let Some(num_elided) = num_entries.checked_sub(MAX_REPORTED_ENTRIES) else {
        return String::new();
    };
    if 0 == num_elided {
        return String::new();
    }

    format!(", and {num_elided} more")
}

/// # Returns
///
/// A core configuration running the tick loop at the suite's common cadence with the given buffer
/// dimensions.
const fn core_config(
    dispatch_queue_capacity: usize,
    ready_task_capacity: usize,
    active_job_list_capacity: usize,
) -> CoreConfig {
    CoreConfig {
        dispatch_queue_capacity: non_zero_usize(dispatch_queue_capacity),
        active_job_list_capacity: non_zero_usize(active_job_list_capacity),
        ready_task_capacity: non_zero_usize(ready_task_capacity),
        commit_ready_task_capacity: non_zero_usize(FINALIZE_TASK_CAPACITY),
        cleanup_ready_task_capacity: non_zero_usize(FINALIZE_TASK_CAPACITY),
        storage_poll_timeout_ms: STORAGE_POLL_TIMEOUT_MS,
        tick_interval_ms: non_zero_u64(TICK_INTERVAL_MS),
    }
}

/// # Returns
///
/// A workload of the given dimensions.
const fn storage_config(
    num_resource_groups: usize,
    num_jobs_per_group: usize,
    num_tasks_per_job: usize,
    emit_finalizations: bool,
) -> FakeStorageConfig {
    FakeStorageConfig {
        num_resource_groups,
        num_jobs_per_group,
        num_tasks_per_job,
        emit_commit_ready: emit_finalizations,
        emit_cleanup_ready: emit_finalizations,
    }
}

/// # Returns
///
/// A worker pinned to the resource group at `group_index`, identifying itself by that index.
fn pinned_worker(group_index: usize, task_duration_ms: u64) -> FakeWorkerConfig {
    FakeWorkerConfig {
        execution_manager_id: dense_index(group_index),
        resource_group_id: Some(resource_group_id(group_index)),
        task_duration_ms,
        next_task_wait_ms: NEXT_TASK_WAIT_MS,
    }
}

/// # Returns
///
/// The `worker_index`-th worker that accepts assignments from any resource group.
fn general_worker(worker_index: usize, task_duration_ms: u64) -> FakeWorkerConfig {
    FakeWorkerConfig {
        execution_manager_id: dense_index(GENERAL_WORKER_ID_BASE + worker_index),
        resource_group_id: None,
        task_duration_ms,
        next_task_wait_ms: NEXT_TASK_WAIT_MS,
    }
}

/// # Returns
///
/// The ID the workload gives the resource group at `group_index`.
fn resource_group_id(group_index: usize) -> ResourceGroupId {
    ResourceGroupId::from(dense_index(group_index))
}

/// # Returns
///
/// `index` as the raw value an ID wraps.
///
/// # Panics
///
/// Panics if `index` exceeds the range of a raw ID.
fn dense_index(index: usize) -> u64 {
    u64::try_from(index).expect("dense index exceeds the raw ID range")
}

/// # Returns
///
/// `value` as a non-zero size.
///
/// # Panics
///
/// Panics if `value` is zero.
const fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("configuration value must be non-zero")
}

/// # Returns
///
/// `value` as a non-zero count.
///
/// # Panics
///
/// Panics if `value` is zero.
const fn non_zero_u64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).expect("configuration value must be non-zero")
}
