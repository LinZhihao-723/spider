//! The core's registry of jobs that currently hold buffered ready tasks.
//!
//! A registered job is co-owned by the registry and by exactly one scheduling position: a resource
//! group's active job list, its pending job queue, or a per-tick downgrade buffer. The registry is
//! core-private and single-threaded, which is what permits the `Rc`-based sharing used here.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::core::DOWNGRADE_LIVES;
use crate::error::JobEntryError;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskIndex;

/// The result of registering a batch of ready tasks against a job.
pub enum UpsertOutcome {
    /// The job was already registered, so its entry keeps the scheduling position it already holds.
    Exist,

    /// The job was registered by this call, so its entry still needs a scheduling position.
    New(SharedJobEntry),
}

/// A job entry shared between the registry and the one scheduling position that holds it.
#[derive(Clone)]
pub struct SharedJobEntry(Rc<SharedJobEntryInner>);

impl SharedJobEntry {
    /// # Returns
    ///
    /// The job this entry belongs to.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn job_id(&self) -> JobId {
        self.0.inner.borrow().job_id
    }

    /// # Returns
    ///
    /// The resource group that owns the job.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn rg_id(&self) -> ResourceGroupId {
        self.0.inner.borrow().rg_id
    }

    /// Reads the flag through a [`Cell`], so it is callable while any borrow of the entry is live.
    ///
    /// # Returns
    ///
    /// Whether the job has reached commit-ready or cleanup-ready.
    pub fn is_finalized(&self) -> bool {
        self.0.finalized.get()
    }

    /// Marks the job as finalized, after which no further regular task may be scheduled from it.
    ///
    /// Writes the flag through a [`Cell`], so it is callable while any borrow of the entry is live.
    pub fn finalize(&self) {
        self.0.finalized.set(true);
    }

    /// Appends newly arrived ready tasks and restores the job's full downgrade budget.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn insert_tasks(&self, task_indices: Vec<TaskIndex>) {
        let mut entry = self.0.inner.borrow_mut();
        entry.ready_tasks.extend(task_indices);
        entry.downgrade_counter = DOWNGRADE_LIVES;
    }

    /// Takes the job's next buffered ready task.
    ///
    /// # Returns
    ///
    /// * `Some` with the next ready task on success, if the job has one buffered.
    /// * `None` on success, if the job has no buffered ready task left.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`JobEntryError::Finalized`] if the job is finalized, so no further regular task may be
    ///   scheduled from it.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn get_next_task(&self) -> Result<Option<TaskIndex>, JobEntryError> {
        if self.is_finalized() {
            return Err(JobEntryError::Finalized);
        }
        Ok(self.0.inner.borrow_mut().ready_tasks.pop_front())
    }

    /// Returns a task taken by [`Self::get_next_task`] to the head of the job's ready tasks.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn restore_task(&self, task_index: TaskIndex) {
        self.0.inner.borrow_mut().ready_tasks.push_front(task_index);
    }

    /// Empties the job's buffered ready tasks.
    ///
    /// # Returns
    ///
    /// The ready tasks the job still held.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn take_ready_tasks(&self) -> VecDeque<TaskIndex> {
        std::mem::take(&mut self.0.inner.borrow_mut().ready_tasks)
    }

    /// # Returns
    ///
    /// Whether the job has at least one buffered ready task.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn has_ready_task(&self) -> bool {
        !self.0.inner.borrow().ready_tasks.is_empty()
    }

    /// # Returns
    ///
    /// The number of further chances the job has to be refilled before it is retired.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn downgrade_counter(&self) -> u32 {
        self.0.inner.borrow().downgrade_counter
    }

    /// Consumes one of the job's remaining chances to be refilled.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn decrement_downgrade_counter(&self) {
        let mut entry = self.0.inner.borrow_mut();
        entry.downgrade_counter = entry.downgrade_counter.saturating_sub(1);
    }

    /// Restores the job's full downgrade budget.
    ///
    /// # Panics
    ///
    /// Panics if a conflicting borrow of the entry is live (see design §8.3).
    pub fn reset_downgrade_counter(&self) {
        self.0.inner.borrow_mut().downgrade_counter = DOWNGRADE_LIVES;
    }

    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created entry for `job_id` holding `task_indices`.
    fn new(job_id: JobId, rg_id: ResourceGroupId, task_indices: Vec<TaskIndex>) -> Self {
        Self(Rc::new(SharedJobEntryInner {
            finalized: Cell::new(false),
            inner: RefCell::new(JobEntry {
                job_id,
                rg_id,
                ready_tasks: task_indices.into(),
                downgrade_counter: DOWNGRADE_LIVES,
            }),
        }))
    }
}

/// The core's map from job to buffered scheduling state.
#[derive(Default)]
pub struct JobRegistry {
    jobs: HashMap<JobId, SharedJobEntry>,
}

impl JobRegistry {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `task_indices` as ready tasks of `job_id`, creating the job's entry if it has
    /// none.
    ///
    /// # Returns
    ///
    /// * [`UpsertOutcome::New`] with the created entry, if the job was not registered.
    /// * [`UpsertOutcome::Exist`] otherwise.
    pub fn upsert(
        &mut self,
        job_id: JobId,
        rg_id: ResourceGroupId,
        task_indices: Vec<TaskIndex>,
    ) -> UpsertOutcome {
        if let Some(entry) = self.jobs.get(&job_id) {
            entry.insert_tasks(task_indices);
            return UpsertOutcome::Exist;
        }

        let entry = SharedJobEntry::new(job_id, rg_id, task_indices);
        self.jobs.insert(job_id, entry.clone());
        UpsertOutcome::New(entry)
    }

    /// Marks `job_id` finalized and drops the registry's co-ownership of its entry.
    ///
    /// # Returns
    ///
    /// The removed entry, or [`None`] if the job was not registered.
    pub fn finalize_and_remove(&mut self, job_id: JobId) -> Option<SharedJobEntry> {
        let entry = self.jobs.remove(&job_id)?;
        entry.finalize();
        Some(entry)
    }

    /// Drops the registry's co-ownership of `job_id`'s entry.
    ///
    /// # Returns
    ///
    /// The removed entry, or [`None`] if the job was not registered.
    pub fn remove(&mut self, job_id: JobId) -> Option<SharedJobEntry> {
        self.jobs.remove(&job_id)
    }

    /// Drops every registered job.
    pub fn clear(&mut self) {
        self.jobs.clear();
    }

    /// # Returns
    ///
    /// The number of registered jobs.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }
}

/// The buffered scheduling state of a single job.
struct JobEntry {
    job_id: JobId,
    rg_id: ResourceGroupId,
    ready_tasks: VecDeque<TaskIndex>,
    downgrade_counter: u32,
}

/// The `Rc`-shared body of a [`SharedJobEntry`].
struct SharedJobEntryInner {
    // Outside the `RefCell` so that `finalize` and `is_finalized` cannot panic: `Cell` mutates
    // through `&self` with no borrow tracking. Both are called from positions where a conflicting
    // guard may plausibly be live -- `finalize_and_remove` while step 2 processes a batch, and the
    // finalized check while step 4 scans `pending_jobs`.
    finalized: Cell<bool>,
    inner: RefCell<JobEntry>,
}
