//! Unit tests for a resource group's scheduling unit: what it picks, when it refuses, and how a
//! job moves between the active list, the pending queue, and retirement.

use super::DEFAULT_SESSION_ID;
use super::make_job_entry;
use super::make_unit;
use crate::core::TaskAssignmentIdIssuer;
use crate::dispatch_queue::GlobalDispatchQueue;
use crate::error::MakeAssignmentError;
use crate::job_registry::DOWNGRADE_LIVES;
use crate::job_registry::JobKey;
use crate::job_registry::JobRegistry;
use crate::resource_group::ResourceGroupTable;
use crate::scheduling_unit::RgSchedulingUnit;
use crate::types::FinalizeKind;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskId;
use crate::types::TaskIndex;

/// The resource group every test in this module schedules for.
const RG_ID: ResourceGroupId = ResourceGroupId::from(4);

/// The first job a test registers.
const JOB_A: JobId = JobId::from(10);

/// The second job a test registers.
const JOB_B: JobId = JobId::from(11);

/// The third job a test registers.
const JOB_C: JobId = JobId::from(12);

/// The free space passed to an assignment attempt that is not meant to reach the threshold.
const FREE_SPACE: usize = 32;

/// One scheduling unit and the structures it publishes into.
struct UnitFixture {
    unit: RgSchedulingUnit,
    registry: JobRegistry,
    jobs_to_retire: Vec<JobKey>,
    global_queue: GlobalDispatchQueue,
    id_issuer: TaskAssignmentIdIssuer,
    rg_table: ResourceGroupTable,
}

impl UnitFixture {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created fixture whose unit holds no job.
    fn new(active_job_list_capacity: usize) -> Self {
        let rg_table = ResourceGroupTable::new();
        Self {
            unit: make_unit(
                &rg_table,
                RG_ID,
                DEFAULT_SESSION_ID,
                active_job_list_capacity,
            ),
            registry: JobRegistry::new(),
            jobs_to_retire: Vec::new(),
            global_queue: GlobalDispatchQueue::new(),
            id_issuer: TaskAssignmentIdIssuer::new(),
            rg_table,
        }
    }

    /// Registers a job of `num_tasks` buffered ready tasks and gives it its scheduling position.
    ///
    /// # Returns
    ///
    /// The key of the registered job.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`make_job_entry`]'s return values on failure.
    fn add_job(&mut self, job_id: JobId, num_tasks: usize) -> anyhow::Result<JobKey> {
        let job_key = make_job_entry(&mut self.registry, job_id, num_tasks)?;
        self.unit.place_new_job(job_key);
        Ok(job_key)
    }

    /// Attempts one assignment against `free` free space, as one turn of the core's decision loop
    /// would.
    ///
    /// # Returns
    ///
    /// The job and task the published assignment carries, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`RgSchedulingUnit::try_make_assignment`]'s return values on failure.
    fn try_make(&mut self, free: usize) -> Result<(JobId, TaskId), MakeAssignmentError> {
        let Self {
            unit,
            registry,
            jobs_to_retire,
            global_queue,
            id_issuer,
            ..
        } = self;
        unit.try_make_assignment(
            free,
            DEFAULT_SESSION_ID,
            id_issuer,
            global_queue,
            registry,
            jobs_to_retire,
        )
    }

    /// Appends `task_indices` to the ready tasks of the job `job_key` refers to.
    ///
    /// # Panics
    ///
    /// Panics if the job has been removed from the registry.
    fn insert_tasks(&mut self, job_key: JobKey, task_indices: Vec<TaskIndex>) {
        self.registry
            .get_mut(job_key)
            .expect("the job is still registered")
            .insert_tasks(task_indices);
    }

    /// # Returns
    ///
    /// The number of further chances the job `job_key` refers to has to be refilled.
    ///
    /// # Panics
    ///
    /// Panics if the job has been removed from the registry.
    fn downgrade_counter(&mut self, job_key: JobKey) -> u32 {
        self.registry
            .get_mut(job_key)
            .expect("the job is still registered")
            .downgrade_counter()
    }

    /// # Returns
    ///
    /// Whether the job `job_key` refers to still has a buffered ready task.
    ///
    /// # Panics
    ///
    /// Panics if the job has been removed from the registry.
    fn has_ready_task(&mut self, job_key: JobKey) -> bool {
        self.registry
            .get_mut(job_key)
            .expect("the job is still registered")
            .has_ready_task()
    }

    /// Closes the group's dispatch queue, so that the channel rejects the next publication.
    fn close_dispatch_queue(&self) {
        self.rg_table
            .get_or_create(RG_ID, DEFAULT_SESSION_ID)
            .sender
            .close();
    }
}

#[test]
fn finalization_tasks_are_dispatched_ahead_of_regular_ones() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(2);
    fixture.add_job(JOB_A, 2)?;
    fixture.unit.push_finalization(JOB_B, FinalizeKind::Commit);
    fixture.unit.push_finalization(JOB_C, FinalizeKind::Cleanup);

    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_B, TaskId::Commit)));
    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_C, TaskId::Cleanup)));
    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_A, TaskId::Index(0))));
    Ok(())
}

#[test]
fn an_empty_unit_reports_no_task() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(2);
    assert!(!fixture.unit.has_schedulable_task());
    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::NoTask)
    );

    // A job with nothing buffered still counts as a schedulable position, but yields no task.
    fixture.add_job(JOB_A, 0)?;
    assert!(fixture.unit.has_schedulable_task());
    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::NoTask)
    );
    Ok(())
}

#[test]
fn the_admission_threshold_binds_at_the_free_space_boundary() -> anyhow::Result<()> {
    const OCCUPANCY: usize = 3;

    let mut fixture = UnitFixture::new(2);
    fixture.add_job(JOB_A, 8)?;
    for task_index in 0..OCCUPANCY {
        assert_eq!(
            fixture.try_make(FREE_SPACE),
            Ok((JOB_A, TaskId::Index(task_index)))
        );
    }
    assert_eq!(fixture.unit.dispatch_queue_size(), OCCUPANCY);

    assert_eq!(
        fixture.try_make(OCCUPANCY),
        Err(MakeAssignmentError::DispatchQueueFull)
    );
    assert_eq!(fixture.unit.dispatch_queue_size(), OCCUPANCY);

    assert_eq!(
        fixture.try_make(OCCUPANCY + 1),
        Ok((JOB_A, TaskId::Index(OCCUPANCY)))
    );
    Ok(())
}

#[test]
fn an_exhausted_active_job_is_replaced_by_a_pending_one() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(1);
    let job_a = fixture.add_job(JOB_A, 1)?;
    fixture.add_job(JOB_B, 1)?;
    assert_eq!(fixture.unit.active_jobs.len(), 1);
    assert_eq!(fixture.unit.pending_jobs.len(), 1);

    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_A, TaskId::Index(0))));
    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_B, TaskId::Index(0))));

    assert_eq!(fixture.unit.active_jobs.len(), 1);
    assert_eq!(fixture.unit.pending_jobs.len(), 0);
    assert_eq!(fixture.downgrade_counter(job_a), 0);
    assert_eq!(fixture.jobs_to_retire.as_slice(), &[]);
    Ok(())
}

#[test]
fn promotion_discards_a_removed_pending_job() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(1);
    fixture.add_job(JOB_A, 1)?;
    fixture.add_job(JOB_B, 1)?;
    fixture.add_job(JOB_C, 1)?;
    fixture
        .registry
        .remove_by_job_id(JOB_B)
        .expect("the registered job is removed by its finalization");

    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_A, TaskId::Index(0))));
    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_C, TaskId::Index(0))));
    assert_eq!(fixture.unit.pending_jobs.len(), 0);
    assert_eq!(fixture.jobs_to_retire.as_slice(), &[]);
    Ok(())
}

#[test]
fn a_removed_active_job_is_swapped_out_for_a_pending_one() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(1);
    fixture.add_job(JOB_A, 1)?;
    fixture.add_job(JOB_B, 1)?;
    fixture
        .registry
        .remove_by_job_id(JOB_A)
        .expect("the registered job is removed by its finalization");

    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_B, TaskId::Index(0))));
    assert_eq!(fixture.unit.active_jobs.len(), 1);
    assert_eq!(fixture.unit.pending_jobs.len(), 0);
    Ok(())
}

#[test]
fn a_job_that_stops_producing_tasks_is_downgraded_and_then_retired() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(1);
    let job_a = fixture.add_job(JOB_A, 1)?;
    assert_eq!(fixture.downgrade_counter(job_a), DOWNGRADE_LIVES);

    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_A, TaskId::Index(0))));
    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::NoTask)
    );
    assert_eq!(fixture.downgrade_counter(job_a), 0);
    assert_eq!(fixture.unit.active_jobs.len(), 0);
    assert_eq!(fixture.jobs_to_retire.as_slice(), &[]);

    fixture.unit.apply_downgrades(&mut fixture.registry);
    assert_eq!(fixture.unit.pending_jobs.len(), 1);
    assert_eq!(fixture.downgrade_counter(job_a), DOWNGRADE_LIVES);

    let mut jobs_to_retire = Vec::new();
    fixture
        .unit
        .promote_pending_jobs(&mut fixture.registry, &mut jobs_to_retire);
    assert_eq!(jobs_to_retire.as_slice(), &[]);
    assert_eq!(fixture.unit.active_jobs.len(), 0);
    assert_eq!(fixture.downgrade_counter(job_a), 0);

    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::NoTask)
    );
    assert_eq!(fixture.jobs_to_retire.as_slice(), &[job_a]);
    assert_eq!(fixture.unit.pending_jobs.len(), 0);
    assert!(!fixture.unit.has_schedulable_task());
    Ok(())
}

#[test]
fn an_arriving_task_restores_a_downgraded_job() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(1);
    let job_a = fixture.add_job(JOB_A, 1)?;

    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_A, TaskId::Index(0))));
    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::NoTask)
    );
    fixture.unit.apply_downgrades(&mut fixture.registry);

    fixture.insert_tasks(job_a, vec![1]);
    let mut jobs_to_retire = Vec::new();
    fixture
        .unit
        .promote_pending_jobs(&mut fixture.registry, &mut jobs_to_retire);
    assert_eq!(jobs_to_retire.as_slice(), &[]);
    assert_eq!(fixture.unit.active_jobs.len(), 1);

    assert_eq!(fixture.try_make(FREE_SPACE), Ok((JOB_A, TaskId::Index(1))));
    assert_eq!(fixture.downgrade_counter(job_a), DOWNGRADE_LIVES);
    Ok(())
}

#[test]
fn a_rejected_publication_leaves_the_finalization_buffered() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(1);
    fixture.add_job(JOB_A, 1)?;
    fixture.unit.push_finalization(JOB_B, FinalizeKind::Commit);
    fixture.close_dispatch_queue();

    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::DispatchQueueClosed)
    );
    assert_eq!(fixture.unit.num_buffered_finalizations(), (1, 0));

    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::DispatchQueueClosed)
    );
    assert_eq!(fixture.unit.num_buffered_finalizations(), (1, 0));
    assert!(fixture.unit.has_schedulable_task());
    Ok(())
}

#[test]
fn a_rejected_publication_leaves_the_regular_task_buffered() -> anyhow::Result<()> {
    let mut fixture = UnitFixture::new(1);
    let job_a = fixture.add_job(JOB_A, 1)?;
    fixture.close_dispatch_queue();

    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::DispatchQueueClosed)
    );
    assert!(fixture.has_ready_task(job_a));

    assert_eq!(
        fixture.try_make(FREE_SPACE),
        Err(MakeAssignmentError::DispatchQueueClosed)
    );
    assert!(fixture.has_ready_task(job_a));
    assert!(fixture.unit.has_schedulable_task());
    Ok(())
}
