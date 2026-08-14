//! Unit tests for the job registry and the entries it shares with the scheduling units.

use std::collections::VecDeque;

use super::make_job_entry;
use crate::core::DOWNGRADE_LIVES;
use crate::error::JobEntryError;
use crate::job_registry::JobRegistry;
use crate::job_registry::UpsertOutcome;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskIndex;

/// The resource group every job in this module belongs to.
const RG_ID: ResourceGroupId = ResourceGroupId::from(7);

/// The job every single-job test registers.
const JOB_ID: JobId = JobId::from(3);

#[test]
fn upsert_registers_a_new_job_and_appends_to_an_existing_one() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();

    let entry = make_job_entry(&mut registry, JOB_ID, RG_ID, 2)?;
    assert_eq!(entry.job_id(), JOB_ID);
    assert_eq!(entry.rg_id(), RG_ID);
    assert_eq!(registry.len(), 1);

    let outcome = registry.upsert(JOB_ID, RG_ID, vec![7, 8]);
    assert!(matches!(outcome, UpsertOutcome::Exist));
    assert_eq!(registry.len(), 1);

    let mut dispatched: Vec<TaskIndex> = Vec::new();
    while let Some(task_index) = entry.get_next_task()? {
        dispatched.push(task_index);
    }
    assert_eq!(dispatched, vec![0, 1, 7, 8]);
    Ok(())
}

#[test]
fn upsert_keeps_the_scheduling_position_of_an_existing_job() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let entry = make_job_entry(&mut registry, JOB_ID, RG_ID, 1)?;

    let outcome = registry.upsert(JOB_ID, RG_ID, vec![1]);
    let UpsertOutcome::Exist = outcome else {
        anyhow::bail!("re-registering a job must not hand out a second entry to place");
    };

    // The entry the registry appended to must be the one the scheduling unit already holds.
    assert_eq!(entry.get_next_task()?, Some(0));
    assert_eq!(entry.get_next_task()?, Some(1));
    assert_eq!(entry.get_next_task()?, None);
    Ok(())
}

#[test]
fn finalize_and_remove_marks_the_entry_and_drops_the_registration() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let entry = make_job_entry(&mut registry, JOB_ID, RG_ID, 2)?;
    assert!(!entry.is_finalized());

    let removed = registry
        .finalize_and_remove(JOB_ID)
        .expect("the registered job is removed by its finalization");
    assert_eq!(removed.job_id(), JOB_ID);
    assert!(removed.is_finalized());
    assert!(entry.is_finalized());
    assert_eq!(registry.len(), 0);

    assert!(registry.finalize_and_remove(JOB_ID).is_none());
    Ok(())
}

#[test]
fn get_next_task_reports_a_finalized_job() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let entry = make_job_entry(&mut registry, JOB_ID, RG_ID, 2)?;
    assert_eq!(entry.get_next_task()?, Some(0));

    registry
        .finalize_and_remove(JOB_ID)
        .expect("the registered job is removed by its finalization");

    assert_eq!(entry.get_next_task(), Err(JobEntryError::Finalized));
    assert!(entry.has_ready_task());
    Ok(())
}

#[test]
fn inserting_tasks_restores_the_downgrade_budget() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let entry = make_job_entry(&mut registry, JOB_ID, RG_ID, 1)?;
    assert_eq!(entry.downgrade_counter(), DOWNGRADE_LIVES);

    for _ in 0..=DOWNGRADE_LIVES {
        entry.decrement_downgrade_counter();
    }
    assert_eq!(entry.downgrade_counter(), 0);

    assert!(matches!(
        registry.upsert(JOB_ID, RG_ID, vec![1]),
        UpsertOutcome::Exist
    ));
    assert_eq!(entry.downgrade_counter(), DOWNGRADE_LIVES);

    entry.decrement_downgrade_counter();
    entry.reset_downgrade_counter();
    assert_eq!(entry.downgrade_counter(), DOWNGRADE_LIVES);
    Ok(())
}

#[test]
fn take_ready_tasks_empties_a_job_without_finalizing_it() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let entry = make_job_entry(&mut registry, JOB_ID, RG_ID, 3)?;

    assert_eq!(entry.take_ready_tasks(), VecDeque::from(vec![0, 1, 2]));
    assert!(!entry.has_ready_task());
    assert_eq!(entry.get_next_task()?, None);
    assert_eq!(entry.take_ready_tasks(), VecDeque::new());
    Ok(())
}

#[test]
fn remove_and_clear_drop_the_registered_jobs() -> anyhow::Result<()> {
    let mut registry = JobRegistry::new();
    let entry = make_job_entry(&mut registry, JOB_ID, RG_ID, 1)?;
    let other_job_id = JobId::from(JOB_ID.get() + 1);
    make_job_entry(&mut registry, other_job_id, RG_ID, 1)?;
    assert_eq!(registry.len(), 2);

    let removed = registry
        .remove(JOB_ID)
        .expect("the registered job is removed by its retirement");
    assert_eq!(removed.job_id(), JOB_ID);
    assert_eq!(registry.len(), 1);
    assert!(registry.remove(JOB_ID).is_none());

    // Retirement drops the registry's co-ownership only; the entry a scheduling unit still holds
    // keeps working.
    assert!(!entry.is_finalized());
    assert_eq!(entry.get_next_task()?, Some(0));

    registry.clear();
    assert_eq!(registry.len(), 0);
    Ok(())
}
