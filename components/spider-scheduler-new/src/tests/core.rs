//! Unit tests for the core's tick: the rotation across resource groups, the active list it keeps,
//! and the state a session bump discards.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::bail;

use super::CoreFixture;
use super::DEFAULT_SESSION_ID;
use super::PENDING_POLL_TIMEOUT_MS;
use super::SHORT_POLL_TIMEOUT_MS;
use super::TICK_DEADLINE;
use super::drain_reader;
use super::make_assignment;
use super::make_config;
use super::make_idle_storage;
use crate::error::CoreError;
use crate::error::MakeAssignmentError;
use crate::harness::FakeStorage;
use crate::harness::FakeStorageConfig;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskId;

/// The number of active jobs a resource group may hold.
const ACTIVE_JOB_LIST_CAPACITY: usize = 4;

/// The first resource group a test seeds.
const RG_A: ResourceGroupId = ResourceGroupId::from(0);

/// The second resource group a test seeds.
const RG_B: ResourceGroupId = ResourceGroupId::from(1);

/// The third resource group a test seeds.
const RG_C: ResourceGroupId = ResourceGroupId::from(2);

#[tokio::test]
async fn the_rotation_arm_persists_across_ticks() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 2;

    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            PENDING_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    );
    for (index, rg_id) in [RG_A, RG_B, RG_C].into_iter().enumerate() {
        fixture.seed_job(rg_id, JobId::from(u64::try_from(index)?), 8)?;
    }

    fixture.core.tick().await?;
    assert_eq!(
        [
            fixture.queue_len(RG_A),
            fixture.queue_len(RG_B),
            fixture.queue_len(RG_C)
        ],
        [1, 1, 0]
    );
    assert_eq!(fixture.core.last_served_rg, Some(RG_B));

    for rg_id in [RG_A, RG_B, RG_C] {
        drain_reader(&fixture.reader(rg_id)).await;
    }

    fixture.core.tick().await?;
    assert_eq!(
        [
            fixture.queue_len(RG_A),
            fixture.queue_len(RG_B),
            fixture.queue_len(RG_C)
        ],
        [1, 0, 1]
    );
    assert_eq!(fixture.core.last_served_rg, Some(RG_A));
    Ok(())
}

#[tokio::test]
async fn dropping_an_exhausted_group_does_not_skip_the_group_moved_into_its_slot()
-> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 1;

    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            PENDING_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    );
    fixture.seed_job(RG_A, JobId::from(0), 8)?;
    fixture.activate_group(RG_B);
    fixture.seed_job(RG_C, JobId::from(2), 8)?;

    // The arm starts on the group that has nothing to schedule, so the tick's single assignment is
    // won by whichever group the removal moves into the vacated slot.
    fixture.core.last_served_rg = Some(RG_A);
    fixture.core.tick().await?;

    assert_eq!(
        [
            fixture.queue_len(RG_A),
            fixture.queue_len(RG_B),
            fixture.queue_len(RG_C)
        ],
        [0, 0, 1]
    );
    assert_eq!(fixture.core.last_served_rg, Some(RG_C));
    assert_eq!(fixture.core.active_rg_list.len(), 2);
    Ok(())
}

#[tokio::test]
async fn an_exhausted_group_stays_active_until_its_dispatch_queue_drains() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 4;

    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            PENDING_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    );
    fixture.activate_group(RG_A);
    fixture.preload_queue(RG_A, 1)?;

    fixture.core.tick().await?;
    assert_eq!(fixture.core.active_rg_list.len(), 1);
    assert!(is_active(&fixture, RG_A));

    assert_eq!(drain_reader(&fixture.reader(RG_A)).await.len(), 1);
    fixture.core.tick().await?;
    assert_eq!(fixture.core.active_rg_list.len(), 0);
    assert!(!is_active(&fixture, RG_A));
    Ok(())
}

#[tokio::test]
async fn dispatching_and_retirement_run_while_a_storage_poll_is_in_flight() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 4;
    // The lone group's admission threshold caps it at half the dispatch buffer per tick.
    const NUM_TASKS_PER_TICK: usize = DISPATCH_QUEUE_CAPACITY / 2;
    const NUM_TASKS: usize = 2 * NUM_TASKS_PER_TICK;

    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            PENDING_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    );
    fixture.seed_job(RG_A, JobId::from(0), NUM_TASKS)?;
    fixture.seed_job(RG_B, JobId::from(1), 0)?;

    // The first tick has no poll to collect; every later one finds the poll it started still in
    // flight, because the idle storage never answers before its timeout.
    fixture.core.tick().await?;
    assert_eq!(fixture.queue_len(RG_A), NUM_TASKS_PER_TICK);
    assert_eq!(fixture.core.job_registry.len(), 2);

    assert_eq!(
        drain_reader(&fixture.reader(RG_A)).await.len(),
        NUM_TASKS_PER_TICK
    );
    fixture.core.tick().await?;

    assert_eq!(fixture.queue_len(RG_A), NUM_TASKS_PER_TICK);
    assert_eq!(fixture.core.global_task_set, HashSet::new());
    assert_eq!(fixture.core.job_registry.len(), 1);
    assert_eq!(fixture.core.active_rg_list.len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_session_bump_clears_the_dedup_set_and_the_finalized_job_table() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 8;
    const SENTINEL_JOB_ID: JobId = JobId::from(4096);

    let storage = FakeStorage::new(FakeStorageConfig {
        num_resource_groups: 1,
        num_jobs_per_group: 1,
        num_tasks_per_job: 2,
        emit_commit_ready: false,
        emit_cleanup_ready: false,
    });
    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            SHORT_POLL_TIMEOUT_MS,
        ),
        storage.clone(),
    );
    tick_until!(fixture.core, fixture.queue_len(RG_A) == 2);

    fixture
        .core
        .global_task_set
        .insert((SENTINEL_JOB_ID, TaskId::Index(0)));
    fixture.core.finalized_jobs.insert(SENTINEL_JOB_ID);

    let new_session_id = storage.bump_session();
    tick_until!(
        fixture.core,
        fixture.session_manager.current() == new_session_id
    );

    assert!(
        !fixture
            .core
            .global_task_set
            .contains(&(SENTINEL_JOB_ID, TaskId::Index(0)))
    );
    assert!(!fixture.core.finalized_jobs.contains(&SENTINEL_JOB_ID));
    Ok(())
}

#[tokio::test]
async fn a_session_bump_readmits_the_tasks_storage_replays() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 1;

    let storage = FakeStorage::new(FakeStorageConfig {
        num_resource_groups: 1,
        num_jobs_per_group: 1,
        num_tasks_per_job: 2,
        emit_commit_ready: false,
        emit_cleanup_ready: false,
    });
    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            SHORT_POLL_TIMEOUT_MS,
        ),
        storage.clone(),
    );
    tick_until!(fixture.core, fixture.queue_len(RG_A) == 1);

    // The buffer holds one assignment, so the job's second task is still buffered in the core and
    // still in the dedup set when the bump lands.
    let published = drain_reader(&fixture.reader(RG_A)).await;
    assert_eq!(published.len(), 1);
    assert_eq!(published[0].session_id, DEFAULT_SESSION_ID);
    assert_eq!(fixture.core.global_task_set.len(), 1);

    let new_session_id = storage.bump_session();
    let mut replayed = Vec::new();
    let deadline = tokio::time::Instant::now() + TICK_DEADLINE;
    while replayed.len() < 2 {
        fixture.core.tick().await?;
        // A poll issued before the bump still lands under the old session, so the assignments it
        // produces are stale by construction and are not part of the replay.
        replayed.extend(
            drain_reader(&fixture.reader(RG_A))
                .await
                .into_iter()
                .filter(|assignment| assignment.session_id == new_session_id),
        );
        if tokio::time::Instant::now() > deadline {
            bail!("storage's replayed tasks were not re-admitted: {replayed:?}");
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    let replayed_task_ids: HashSet<TaskId> = replayed
        .iter()
        .map(|assignment| assignment.task_id)
        .collect();
    assert_eq!(
        replayed_task_ids,
        HashSet::from([TaskId::Index(0), TaskId::Index(1)])
    );
    Ok(())
}

#[tokio::test]
async fn a_rescheduled_assignment_is_readmitted() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 8;
    const LOST_TASK_ID: TaskId = TaskId::Index(9);

    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            SHORT_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    );
    let lost = make_assignment(
        RG_A,
        JobId::from(0),
        LOST_TASK_ID,
        fixture.session_manager.current(),
    );
    fixture.reschedule_queue_writer.send(lost)?;

    tick_until!(fixture.core, fixture.queue_len(RG_A) == 1);

    let redispatched = drain_reader(&fixture.reader(RG_A)).await;
    assert_eq!(redispatched.len(), 1);
    assert_eq!(redispatched[0].task_id, LOST_TASK_ID);
    assert_eq!(redispatched[0].job_id, JobId::from(0));
    Ok(())
}

#[tokio::test]
async fn a_closed_dispatch_queue_fails_the_tick() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 4;

    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            PENDING_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    );
    fixture.seed_job(RG_A, JobId::from(0), 4)?;
    fixture.close_dispatch_queue(RG_A);

    let CoreError::FatalPublication(err) = fixture
        .core
        .tick()
        .await
        .expect_err("a closed dispatch queue fails the tick")
    else {
        bail!("the tick failed with something other than a fatal publication failure");
    };
    assert_eq!(err, MakeAssignmentError::DispatchQueueClosed);
    Ok(())
}

#[tokio::test]
async fn a_closed_broadcast_queue_fails_the_tick() -> anyhow::Result<()> {
    const DISPATCH_QUEUE_CAPACITY: usize = 4;

    let mut fixture = CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            PENDING_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    );
    fixture.seed_job(RG_A, JobId::from(0), 4)?;
    fixture.close_broadcast_queue();

    let CoreError::FatalPublication(err) = fixture
        .core
        .tick()
        .await
        .expect_err("a closed broadcast queue fails the tick")
    else {
        bail!("the tick failed with something other than a fatal publication failure");
    };
    assert_eq!(err, MakeAssignmentError::BroadcastQueueClosed);
    Ok(())
}

/// # Returns
///
/// Whether the core still holds `rg_id` on its active resource group list.
///
/// # Panics
///
/// Panics if the core has no scheduling unit for `rg_id`.
fn is_active(fixture: &CoreFixture, rg_id: ResourceGroupId) -> bool {
    let unit_index = *fixture
        .core
        .rg_index
        .get(&rg_id)
        .expect("the core holds a scheduling unit for the group");
    fixture.core.rg_units[unit_index].is_active
}
