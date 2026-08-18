//! Unit tests for the job registry: what a key resolves to, and when it stops resolving.

use std::collections::VecDeque;

use super::make_job_entry;
use crate::job_registry::DOWNGRADE_LIVES;
use crate::job_registry::JobEntry;
use crate::job_registry::JobKey;
use crate::job_registry::JobRegistry;
use crate::job_registry::UpsertOutcome;
use crate::types::JobId;
use crate::types::TaskIndex;

/// The job every single-job test registers.
const JOB_ID: JobId = JobId::from(3);

#[test]
fn upsert_registers_a_new_job_and_appends_to_an_existing_one() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();

    let job_key = make_job_entry(&mut registry, JOB_ID, 2)?;
    assert_eq!(job_id_of(&mut registry, job_key), Some(JOB_ID));
    assert_eq!(registry.len(), 1);

    let outcome = registry.upsert(JOB_ID, vec![7, 8]);
    assert!(matches!(outcome, UpsertOutcome::Exist));
    assert_eq!(registry.len(), 1);

    let entry = entry_of(&mut registry, job_key);
    let mut dispatched: Vec<TaskIndex> = Vec::new();
    while let Some(task_index) = entry.pop_next_task() {
        dispatched.push(task_index);
    }
    assert_eq!(dispatched, vec![0, 1, 7, 8]);
    Ok(())
}

#[test]
fn upsert_keeps_the_scheduling_position_of_an_existing_job() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let job_key = make_job_entry(&mut registry, JOB_ID, 1)?;

    let outcome = registry.upsert(JOB_ID, vec![1]);
    let UpsertOutcome::Exist = outcome else {
        anyhow::bail!("re-registering a job must not hand out a second key to place");
    };

    // The entry the registry appended to must be the one the scheduling unit's key resolves to.
    let entry = entry_of(&mut registry, job_key);
    assert_eq!(entry.pop_next_task(), Some(0));
    assert_eq!(entry.pop_next_task(), Some(1));
    assert_eq!(entry.pop_next_task(), None);
    Ok(())
}

#[test]
fn remove_by_job_id_hands_back_the_entry_and_drops_the_registration() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let job_key = make_job_entry(&mut registry, JOB_ID, 2)?;

    let mut removed = registry
        .remove_by_job_id(JOB_ID)
        .expect("the registered job is removed by its finalization");
    assert_eq!(removed.job_id(), JOB_ID);
    assert_eq!(removed.take_ready_tasks(), VecDeque::from(vec![0, 1]));
    assert_eq!(registry.len(), 0);

    assert_eq!(job_id_of(&mut registry, job_key), None);
    assert!(registry.remove_by_job_id(JOB_ID).is_none());
    Ok(())
}

#[test]
fn a_removed_jobs_key_never_resolves_to_a_later_job() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let stale_key = make_job_entry(&mut registry, JOB_ID, 2)?;
    registry
        .remove_by_job_id(JOB_ID)
        .expect("the registered job is removed by its finalization");

    // The freed slot is offered to the next job registered, which is exactly the case a plain
    // index could not distinguish from the removed one.
    let next_job_id = JobId::from(JOB_ID.get() + 1);
    let next_key = make_job_entry(&mut registry, next_job_id, 1)?;
    assert_eq!(job_id_of(&mut registry, next_key), Some(next_job_id));
    assert_eq!(job_id_of(&mut registry, stale_key), None);
    Ok(())
}

#[test]
fn inserting_tasks_restores_the_downgrade_budget() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let job_key = make_job_entry(&mut registry, JOB_ID, 1)?;
    assert_eq!(
        entry_of(&mut registry, job_key).downgrade_counter(),
        DOWNGRADE_LIVES
    );

    for _ in 0..=DOWNGRADE_LIVES {
        entry_of(&mut registry, job_key).decrement_downgrade_counter();
    }
    assert_eq!(entry_of(&mut registry, job_key).downgrade_counter(), 0);

    assert!(matches!(
        registry.upsert(JOB_ID, vec![1]),
        UpsertOutcome::Exist
    ));
    assert_eq!(
        entry_of(&mut registry, job_key).downgrade_counter(),
        DOWNGRADE_LIVES
    );

    let entry = entry_of(&mut registry, job_key);
    entry.decrement_downgrade_counter();
    entry.reset_downgrade_counter();
    assert_eq!(entry.downgrade_counter(), DOWNGRADE_LIVES);
    Ok(())
}

#[test]
fn take_ready_tasks_empties_a_job_without_removing_it() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let job_key = make_job_entry(&mut registry, JOB_ID, 3)?;

    let entry = entry_of(&mut registry, job_key);
    assert_eq!(entry.take_ready_tasks(), VecDeque::from(vec![0, 1, 2]));
    assert!(!entry.has_ready_task());
    assert_eq!(entry.pop_next_task(), None);
    assert_eq!(entry.take_ready_tasks(), VecDeque::new());

    assert_eq!(job_id_of(&mut registry, job_key), Some(JOB_ID));
    assert_eq!(registry.len(), 1);
    Ok(())
}

#[test]
fn remove_and_clear_drop_the_registered_jobs() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let job_key = make_job_entry(&mut registry, JOB_ID, 1)?;
    let other_job_id = JobId::from(JOB_ID.get() + 1);
    let other_job_key = make_job_entry(&mut registry, other_job_id, 1)?;
    assert_eq!(registry.len(), 2);

    let removed = registry
        .remove(job_key)
        .expect("the registered job is removed by its retirement");
    assert_eq!(removed.job_id(), JOB_ID);
    assert_eq!(registry.len(), 1);
    assert!(registry.remove(job_key).is_none());

    // Removing one job leaves every other key resolving as it did.
    assert_eq!(job_id_of(&mut registry, other_job_key), Some(other_job_id));

    registry.clear();
    assert_eq!(registry.len(), 0);
    assert_eq!(job_id_of(&mut registry, other_job_key), None);
    Ok(())
}

/// # Returns
///
/// The ID of the job `job_key` refers to, or [`None`] if the key no longer resolves.
fn job_id_of(registry: &mut JobRegistry, job_key: JobKey) -> Option<JobId> {
    registry.get_mut(job_key).map(|entry| entry.job_id())
}

/// # Returns
///
/// The entry `job_key` refers to.
///
/// # Panics
///
/// Panics if the key no longer resolves.
fn entry_of(registry: &mut JobRegistry, job_key: JobKey) -> &mut JobEntry {
    registry
        .get_mut(job_key)
        .expect("the job is still registered")
}
