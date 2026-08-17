//! The core's registry of jobs that currently hold buffered ready tasks.
//!
//! The registry is a generational arena and is the sole owner of every job entry. A scheduling
//! position -- a resource group's active job list, its pending job queue, or a per-tick downgrade
//! buffer -- holds a [`JobKey`] and resolves it against the arena when it needs the entry. A key
//! held past its entry's removal fails to resolve, which is how a scheduling position discovers
//! that its job is gone.

use std::collections::HashMap;
use std::collections::VecDeque;

use slotmap::SlotMap;

use crate::core::DOWNGRADE_LIVES;
use crate::types::JobId;
use crate::types::TaskIndex;

slotmap::new_key_type! {
    /// A generational handle to a job entry owned by a [`JobRegistry`].
    pub struct JobKey;
}

/// The result of registering a batch of ready tasks against a job.
pub enum UpsertOutcome {
    /// The job was already registered, so its entry keeps the scheduling position it already holds.
    Exist,

    /// The job was registered by this call, so its entry still needs a scheduling position.
    New(JobKey),
}

/// The buffered scheduling state of a single job.
pub struct JobEntry {
    job_id: JobId,
    ready_tasks: VecDeque<TaskIndex>,
    downgrade_counter: u32,
}

impl JobEntry {
    /// # Returns
    ///
    /// The job this entry belongs to.
    pub const fn job_id(&self) -> JobId {
        self.job_id
    }

    /// Appends newly arrived ready tasks and restores the job's full downgrade budget.
    pub fn insert_tasks(&mut self, task_indices: Vec<TaskIndex>) {
        self.ready_tasks.extend(task_indices);
        self.downgrade_counter = DOWNGRADE_LIVES;
    }

    /// Takes the job's next buffered ready task.
    ///
    /// # Returns
    ///
    /// * `Some` with the next ready task, if the job has one buffered.
    /// * `None` if the job has no buffered ready task left.
    pub fn get_next_task(&mut self) -> Option<TaskIndex> {
        self.ready_tasks.pop_front()
    }

    /// Returns a task taken by [`Self::get_next_task`] to the head of the job's ready tasks.
    pub fn restore_task(&mut self, task_index: TaskIndex) {
        self.ready_tasks.push_front(task_index);
    }

    /// Empties the job's buffered ready tasks.
    ///
    /// # Returns
    ///
    /// The ready tasks the job still held.
    pub fn take_ready_tasks(&mut self) -> VecDeque<TaskIndex> {
        std::mem::take(&mut self.ready_tasks)
    }

    /// # Returns
    ///
    /// Whether the job has at least one buffered ready task.
    pub fn has_ready_task(&self) -> bool {
        !self.ready_tasks.is_empty()
    }

    /// # Returns
    ///
    /// The number of further chances the job has to be refilled before it is retired.
    pub const fn downgrade_counter(&self) -> u32 {
        self.downgrade_counter
    }

    /// Consumes one of the job's remaining chances to be refilled.
    pub const fn decrement_downgrade_counter(&mut self) {
        self.downgrade_counter = self.downgrade_counter.saturating_sub(1);
    }

    /// Restores the job's full downgrade budget.
    pub const fn reset_downgrade_counter(&mut self) {
        self.downgrade_counter = DOWNGRADE_LIVES;
    }

    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created entry for `job_id` holding `task_indices`.
    fn new(job_id: JobId, task_indices: Vec<TaskIndex>) -> Self {
        Self {
            job_id,
            ready_tasks: task_indices.into(),
            downgrade_counter: DOWNGRADE_LIVES,
        }
    }
}

/// The core's arena of job entries, indexed by job ID.
#[derive(Default)]
pub struct JobRegistry {
    jobs: SlotMap<JobKey, JobEntry>,
    by_job_id: HashMap<JobId, JobKey>,
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
    /// * [`UpsertOutcome::New`] with the key of the created entry, if the job was not registered.
    /// * [`UpsertOutcome::Exist`] otherwise.
    pub fn upsert(&mut self, job_id: JobId, task_indices: Vec<TaskIndex>) -> UpsertOutcome {
        if let Some(entry) = self
            .by_job_id
            .get(&job_id)
            .and_then(|key| self.jobs.get_mut(*key))
        {
            entry.insert_tasks(task_indices);
            return UpsertOutcome::Exist;
        }

        let key = self.jobs.insert(JobEntry::new(job_id, task_indices));
        self.by_job_id.insert(job_id, key);
        UpsertOutcome::New(key)
    }

    /// # Returns
    ///
    /// The entry `key` refers to, or [`None`] if the job has been removed from the registry.
    pub fn get_mut(&mut self, key: JobKey) -> Option<&mut JobEntry> {
        self.jobs.get_mut(key)
    }

    /// Removes `job_id`'s entry from the registry, after which every key to it fails to resolve.
    ///
    /// # Returns
    ///
    /// The removed entry, or [`None`] if the job was not registered.
    pub fn remove_by_job_id(&mut self, job_id: JobId) -> Option<JobEntry> {
        let key = self.by_job_id.remove(&job_id)?;
        self.jobs.remove(key)
    }

    /// Removes the entry `key` refers to, after which every other key to it fails to resolve.
    ///
    /// # Returns
    ///
    /// The removed entry, or [`None`] if the job has already been removed.
    pub fn remove(&mut self, key: JobKey) -> Option<JobEntry> {
        let entry = self.jobs.remove(key)?;
        self.by_job_id.remove(&entry.job_id);
        Some(entry)
    }

    /// Drops every registered job.
    pub fn clear(&mut self) {
        self.jobs.clear();
        self.by_job_id.clear();
    }

    /// # Returns
    ///
    /// The number of registered jobs.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }
}
