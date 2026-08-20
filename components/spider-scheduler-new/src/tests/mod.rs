//! Crate-level unit tests.
//!
//! The tests drive the core's tick and the structures it decides with directly: no gRPC server, no
//! workers, and no harness. Whatever an execution manager would do -- draining a group's queue,
//! consuming a hint -- a test body does by hand through the same `pub(crate)` entry points the
//! dispatch service uses.

/// Drives ticks on `core` until `predicate` holds, failing the calling test if it does not hold
/// within [`TICK_DEADLINE`].
///
/// A macro rather than an async function so that the predicate is an expression rather than a
/// closure, and may therefore borrow whatever the tick mutates.
macro_rules! tick_until {
    ($core:expr, $predicate:expr) => {{
        let deadline = tokio::time::Instant::now() + $crate::tests::TICK_DEADLINE;
        loop {
            $core.tick().await?;
            if $predicate {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                ::anyhow::bail!(
                    "the core did not reach the expected state within {:?}",
                    $crate::tests::TICK_DEADLINE
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    }};
}

mod admission;
mod core;
mod dispatch_queue;
mod job_registry;
mod metrics;
mod scheduling_unit;

use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::time::Duration;

use anyhow::bail;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio_util::sync::CancellationToken;

use crate::config::CoreConfig;
use crate::core::Core;
use crate::dispatch_queue::GlobalDispatchQueue;
use crate::harness::FakeStorage;
use crate::harness::FakeStorageConfig;
use crate::job_registry::JobKey;
use crate::job_registry::JobRegistry;
use crate::job_registry::UpsertOutcome;
use crate::resource_group::ResourceGroupTable;
use crate::resource_group::RgDispatchQueueReader;
use crate::resource_group::RgDispatchQueueWriter;
use crate::scheduling_unit::RgSchedulingUnit;
use crate::session::SessionManager;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskAssignment;
use crate::types::TaskAssignmentId;
use crate::types::TaskId;
use crate::types::TaskIndex;

/// The session a test that never bumps the session runs under.
const DEFAULT_SESSION_ID: SessionId = 0;

/// A storage poll timeout no test outlives, so every tick after the first observes a poll that is
/// still in flight.
const PENDING_POLL_TIMEOUT_MS: u64 = 60_000;

/// The storage poll timeout of a test that must observe completed polls.
const SHORT_POLL_TIMEOUT_MS: u64 = 10;

/// The longest a test waits for the ticks it drives to reach the state it expects.
const TICK_DEADLINE: Duration = Duration::from_secs(10);

/// A core wired to an in-memory storage and to the dispatch structures a test inspects, driven by
/// manual [`Core::tick`] calls.
struct CoreFixture {
    core: Core<FakeStorage>,
    rg_table: ResourceGroupTable,
    session_manager: SessionManager,
    reschedule_queue_writer: UnboundedSender<TaskAssignment>,
    active_job_list_capacity: usize,
}

impl CoreFixture {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created fixture whose core holds no buffered task and no active resource group.
    fn new(config: CoreConfig, storage: FakeStorage) -> Self {
        let rg_table = ResourceGroupTable::new();
        let global_queue = GlobalDispatchQueue::new();
        let session_manager = SessionManager::new(DEFAULT_SESSION_ID);
        let (reschedule_queue_writer, reschedule_queue_reader) = unbounded_channel();
        let core = Core::new(
            config,
            storage,
            rg_table.clone(),
            global_queue,
            session_manager.clone(),
            reschedule_queue_reader,
            CancellationToken::new(),
        );
        Self {
            core,
            rg_table,
            session_manager,
            reschedule_queue_writer,
            active_job_list_capacity: config.active_job_list_capacity.get(),
        }
    }

    /// Puts `rg_id` on the core's active resource group list, appending its scheduling unit built
    /// against the group's dispatch queue endpoints if the core has none.
    ///
    /// # Returns
    ///
    /// The position of the group's scheduling unit in the core's unit arena.
    fn activate_group(&mut self, rg_id: ResourceGroupId) -> usize {
        if let Some(unit_index) = self.core.rg_index.get(&rg_id) {
            return *unit_index;
        }

        let mut unit = make_unit(
            &self.rg_table,
            rg_id,
            self.session_manager.current(),
            self.active_job_list_capacity,
        );
        unit.is_active = true;
        let unit_index = self.core.rg_units.len();
        self.core.rg_units.push(unit);
        self.core.rg_index.insert(rg_id, unit_index);
        self.core.active_rg_list.push(unit_index);
        unit_index
    }

    /// Registers a job of `num_tasks` buffered ready tasks against `rg_id`, exactly as a completed
    /// inbound poll carrying that job would, and activates the group.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`make_job_entry`]'s return values on failure.
    fn seed_job(
        &mut self,
        rg_id: ResourceGroupId,
        job_id: JobId,
        num_tasks: usize,
    ) -> anyhow::Result<()> {
        for task_index in 0..num_tasks {
            self.core
                .global_task_set
                .insert((job_id, TaskId::Index(task_index)));
        }
        let job_key = make_job_entry(&mut self.core.job_registry, job_id, num_tasks)?;
        let unit_index = self.activate_group(rg_id);
        self.core.rg_units[unit_index].place_new_job(job_key);
        Ok(())
    }

    /// Puts `num_assignments` assignments straight into `rg_id`'s dispatch queue, standing in for
    /// the occupancy a group carries over from earlier ticks.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`anyhow::Error`] if the group's dispatch queue rejects an assignment.
    fn preload_queue(&self, rg_id: ResourceGroupId, num_assignments: usize) -> anyhow::Result<()> {
        let endpoints = self
            .rg_table
            .get_or_create(rg_id, self.session_manager.current());
        for index in 0..num_assignments {
            let assignment = make_assignment(
                rg_id,
                JobId::from(u64::MAX),
                TaskId::Index(index),
                self.session_manager.current(),
            );
            if endpoints.sender.try_send(assignment).is_err() {
                bail!(
                    "the dispatch queue of resource group {rg_id} rejected a preloaded assignment"
                );
            }
        }
        Ok(())
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for `rg_id`.
    fn queue_len(&self, rg_id: ResourceGroupId) -> usize {
        writer_of(&self.rg_table, rg_id, self.session_manager.current()).queue_len()
    }

    /// # Returns
    ///
    /// The read side of `rg_id`'s dispatch queue in the session the core currently serves, which a
    /// test hands to [`drain_reader`] to play a pinned execution manager.
    fn reader(&self, rg_id: ResourceGroupId) -> RgDispatchQueueReader {
        reader_of(&self.rg_table, rg_id, self.session_manager.current())
    }
}

/// # Returns
///
/// A core config with the given capacities and storage poll timeout, whose remaining tunables never
/// throttle a test.
///
/// # Panics
///
/// Panics if either capacity is zero.
fn make_config(
    dispatch_queue_capacity: usize,
    active_job_list_capacity: usize,
    storage_poll_timeout_ms: u64,
) -> CoreConfig {
    CoreConfig {
        dispatch_queue_capacity: NonZeroUsize::new(dispatch_queue_capacity)
            .expect("the dispatch queue capacity is non-zero"),
        active_job_list_capacity: NonZeroUsize::new(active_job_list_capacity)
            .expect("the active job list capacity is non-zero"),
        ready_task_capacity: NonZeroUsize::new(16_384).expect("16384 is non-zero"),
        commit_ready_task_capacity: NonZeroUsize::new(64).expect("64 is non-zero"),
        cleanup_ready_task_capacity: NonZeroUsize::new(64).expect("64 is non-zero"),
        storage_poll_timeout_ms,
        tick_interval_ms: NonZeroU64::new(1).expect("1 is non-zero"),
    }
}

/// # Returns
///
/// A fake storage whose workload is empty, so its polls never deliver an entry and never complete
/// before their timeout.
fn make_idle_storage() -> FakeStorage {
    FakeStorage::new(FakeStorageConfig {
        num_resource_groups: 0,
        num_jobs_per_group: 0,
        num_tasks_per_job: 0,
        emit_commit_ready: false,
        emit_cleanup_ready: false,
    })
}

/// # Returns
///
/// A newly created, inactive scheduling unit publishing into `rg_id`'s dispatch queue.
fn make_unit(
    rg_table: &ResourceGroupTable,
    rg_id: ResourceGroupId,
    session_id: SessionId,
    active_job_list_capacity: usize,
) -> RgSchedulingUnit {
    RgSchedulingUnit::new(
        rg_id,
        &rg_table.get_or_create(rg_id, session_id),
        active_job_list_capacity,
    )
}

/// Registers a job of `num_tasks` buffered ready tasks, whose task indices are `0..num_tasks`.
///
/// # Returns
///
/// The key of the registered job, which still needs a scheduling position.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`anyhow::Error`] if the job is already registered.
fn make_job_entry(
    registry: &mut JobRegistry,
    job_id: JobId,
    num_tasks: usize,
) -> anyhow::Result<JobKey> {
    let task_indices: Vec<TaskIndex> = (0..num_tasks).collect();
    let UpsertOutcome::New(job_key) = registry.upsert(job_id, task_indices) else {
        bail!("job {job_id} is already registered");
    };
    Ok(job_key)
}

/// # Returns
///
/// An assignment of `task_id` carrying an ID no assignment the core publishes can collide with.
fn make_assignment(
    rg_id: ResourceGroupId,
    job_id: JobId,
    task_id: TaskId,
    session_id: SessionId,
) -> TaskAssignment {
    TaskAssignment {
        id: TaskAssignmentId::from(u64::MAX),
        resource_group_id: rg_id,
        job_id,
        task_id,
        session_id,
    }
}

/// # Returns
///
/// The read side of `rg_id`'s dispatch queue, creating the group in `session_id` if the table has
/// no entry for it.
fn reader_of(
    rg_table: &ResourceGroupTable,
    rg_id: ResourceGroupId,
    session_id: SessionId,
) -> RgDispatchQueueReader {
    rg_table.get_dispatch_queue_reader(rg_id, session_id)
}

/// # Returns
///
/// The write side of `rg_id`'s dispatch queue, creating the group in `session_id` if the table has
/// no entry for it.
fn writer_of(
    rg_table: &ResourceGroupTable,
    rg_id: ResourceGroupId,
    session_id: SessionId,
) -> RgDispatchQueueWriter {
    rg_table.get_or_create(rg_id, session_id).writer()
}

/// Takes every assignment currently queued behind `reader`, playing a pinned execution manager,
/// which leaves the group's hint counter untouched.
///
/// # Returns
///
/// The assignments taken, in dispatch order.
async fn drain_reader(reader: &RgDispatchQueueReader) -> Vec<TaskAssignment> {
    let mut assignments = Vec::new();
    while let Some(assignment) = reader.recv_pinned(Duration::ZERO).await {
        assignments.push(assignment);
    }
    assignments
}
