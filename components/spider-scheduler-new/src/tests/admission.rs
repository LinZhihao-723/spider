//! Unit tests for the admission policy.
//!
//! The policy is a dynamic threshold -- a group is admitted more work only while its own queue
//! occupancy is below the buffer's current free space -- plus the quantum-1 rotation that lets
//! every group creep toward the same falling threshold. Both halves are load-bearing: a threshold
//! read correctly but applied to one group at a time produces a staircase rather than an
//! equilibrium, so these tests assert the resulting occupancies, not the code path that produced
//! them.

use super::CoreFixture;
use super::PENDING_POLL_TIMEOUT_MS;
use super::make_config;
use super::make_idle_storage;
use crate::types::JobId;
use crate::types::ResourceGroupId;

/// The dispatch buffer capacity `B`, matching the design document's worked example.
const DISPATCH_QUEUE_CAPACITY: usize = 256;

/// The number of active jobs a resource group may hold.
const ACTIVE_JOB_LIST_CAPACITY: usize = 4;

/// The number of backlogged resource groups `N` in the equilibrium test.
const NUM_BACKLOGGED_GROUPS: usize = 5;

/// The number of ready tasks each group is seeded with, more than any single tick may publish.
const NUM_TASKS_PER_JOB: usize = 512;

/// How far a group's occupancy may sit from the equilibrium share before the rotation is considered
/// broken. The staircase this test exists to catch misses by an order of magnitude: batch filling
/// five groups against `B = 256` yields `64, 64, 64, 64, 0` with no free space at all.
const SHARE_TOLERANCE: usize = 6;

#[tokio::test]
async fn one_tick_leaves_every_backlogged_group_at_the_dynamic_threshold() -> anyhow::Result<()> {
    let expected_share = DISPATCH_QUEUE_CAPACITY / (NUM_BACKLOGGED_GROUPS + 1);
    let mut fixture = new_fixture();
    for index in 0..NUM_BACKLOGGED_GROUPS {
        let raw_id = u64::try_from(index)?;
        fixture.seed_job(
            ResourceGroupId::from(raw_id),
            JobId::from(raw_id),
            NUM_TASKS_PER_JOB,
        )?;
    }

    fixture.core.tick().await?;

    let occupancies = occupancies_of(&fixture, NUM_BACKLOGGED_GROUPS);
    let occupancy: usize = occupancies.iter().sum();
    let free = DISPATCH_QUEUE_CAPACITY - occupancy;
    for (index, group_occupancy) in occupancies.iter().enumerate() {
        assert!(
            group_occupancy.abs_diff(expected_share) <= SHARE_TOLERANCE,
            "group {index} holds {group_occupancy} assignments, expected about {expected_share}: \
             {occupancies:?}"
        );
    }
    assert!(
        free.abs_diff(expected_share) <= SHARE_TOLERANCE,
        "the tick left {free} free, expected about {expected_share}: {occupancies:?}"
    );

    let published: usize =
        NUM_BACKLOGGED_GROUPS * NUM_TASKS_PER_JOB - fixture.core.global_task_set.len();
    assert_eq!(published, occupancy);
    Ok(())
}

#[tokio::test]
async fn no_group_is_batch_filled_while_another_waits() -> anyhow::Result<()> {
    let mut fixture = new_fixture();
    for index in 0..NUM_BACKLOGGED_GROUPS {
        let raw_id = u64::try_from(index)?;
        fixture.seed_job(
            ResourceGroupId::from(raw_id),
            JobId::from(raw_id),
            NUM_TASKS_PER_JOB,
        )?;
    }

    fixture.core.tick().await?;

    let occupancies = occupancies_of(&fixture, NUM_BACKLOGGED_GROUPS);
    let most = *occupancies
        .iter()
        .max()
        .expect("the tick served at least one group");
    let least = *occupancies
        .iter()
        .min()
        .expect("the tick served at least one group");
    assert!(
        most - least <= 2,
        "the rotation is not interleaved at quantum 1: {occupancies:?}"
    );
    assert!(
        most < DISPATCH_QUEUE_CAPACITY / 2,
        "a single group took half the dispatch buffer: {occupancies:?}"
    );
    assert!(
        least > 0,
        "a backlogged group was left at zero: {occupancies:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_newly_active_group_is_admitted_against_a_backlogged_incumbent() -> anyhow::Result<()> {
    const INCUMBENT_OCCUPANCY: usize = 100;

    let incumbent_rg_id = ResourceGroupId::from(0);
    let newcomer_rg_id = ResourceGroupId::from(1);
    let mut fixture = new_fixture();
    fixture.seed_job(incumbent_rg_id, JobId::from(0), NUM_TASKS_PER_JOB)?;
    fixture.seed_job(newcomer_rg_id, JobId::from(1), NUM_TASKS_PER_JOB)?;
    fixture.preload_queue(incumbent_rg_id, INCUMBENT_OCCUPANCY)?;

    fixture.core.tick().await?;

    let occupancies = occupancies_of(&fixture, 2);
    let free = DISPATCH_QUEUE_CAPACITY - occupancies.iter().sum::<usize>();
    assert!(
        occupancies[1] >= DISPATCH_QUEUE_CAPACITY / 8,
        "the newly active group was starved by the incumbent: {occupancies:?}"
    );
    assert!(
        occupancies[0] - occupancies[1] < INCUMBENT_OCCUPANCY,
        "the incumbent's head start grew instead of shrinking: {occupancies:?}"
    );
    assert!(
        free >= DISPATCH_QUEUE_CAPACITY / 8,
        "the tick left only {free} free for the next group to arrive: {occupancies:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_lone_group_takes_no_more_than_half_the_dispatch_buffer() -> anyhow::Result<()> {
    let rg_id = ResourceGroupId::from(0);
    let mut fixture = new_fixture();
    fixture.seed_job(rg_id, JobId::from(0), NUM_TASKS_PER_JOB)?;

    fixture.core.tick().await?;

    let occupancy = fixture.queue_len(rg_id);
    assert!(
        occupancy <= DISPATCH_QUEUE_CAPACITY / 2,
        "the only active group holds {occupancy} of {DISPATCH_QUEUE_CAPACITY} assignments"
    );
    assert!(
        occupancy >= DISPATCH_QUEUE_CAPACITY / 4,
        "the only active group holds only {occupancy} of {DISPATCH_QUEUE_CAPACITY} assignments"
    );
    Ok(())
}

/// # Returns
///
/// A newly created fixture over an idle storage, so a tick's decisions come from the tasks the test
/// seeded and from nothing else.
fn new_fixture() -> CoreFixture {
    CoreFixture::new(
        make_config(
            DISPATCH_QUEUE_CAPACITY,
            ACTIVE_JOB_LIST_CAPACITY,
            PENDING_POLL_TIMEOUT_MS,
        ),
        make_idle_storage(),
    )
}

/// # Returns
///
/// The number of assignments queued for each of the first `num_groups` resource groups, indexed by
/// resource group ID.
fn occupancies_of(fixture: &CoreFixture, num_groups: usize) -> Vec<usize> {
    (0..num_groups)
        .map(|index| {
            let raw_id = u64::try_from(index).expect("a group index fits a raw ID");
            fixture.queue_len(ResourceGroupId::from(raw_id))
        })
        .collect()
}
