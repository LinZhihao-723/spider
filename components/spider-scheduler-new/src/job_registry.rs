//! Registry of currently running jobs, each holding a queue of ready tasks to schedule from.

use std::collections::HashMap;
use std::collections::VecDeque;

use slotmap::SlotMap;

use crate::types::JobId;
use crate::types::TaskIndex;

/// The number of chances a job that produced no task gets before it is retired.
pub const DOWNGRADE_LIVES: u32 = 1;

slotmap::new_key_type! {
    /// A generational handle to a job entry owned by a [`JobRegistry`].
    pub struct JobKey;
}

/// The result of inserting a batch of ready tasks for a job.
pub enum UpsertOutcome {
    /// The job was already registered.
    Exist,

    /// The job was registered by this call, along with a key to access its entry.
    New(JobKey),
}

/// The scheduling state of a single job, maintaining a FIFO queue for its ready tasks.
pub struct JobEntry {
    job_id: JobId,
    ready_task_queue: VecDeque<TaskIndex>,
    downgrade_counter: u32,
}

impl JobEntry {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created entry for `job_id` holding `task_indices`.
    fn new(job_id: JobId, task_indices: Vec<TaskIndex>) -> Self {
        Self {
            job_id,
            ready_task_queue: task_indices.into(),
            downgrade_counter: DOWNGRADE_LIVES,
        }
    }

    /// # Returns
    ///
    /// The job ID of this entry.
    pub const fn job_id(&self) -> JobId {
        self.job_id
    }

    /// Appends newly arrived ready tasks and restores the job's full downgrade budget.
    pub fn insert_tasks(&mut self, task_indices: Vec<TaskIndex>) {
        self.ready_task_queue.extend(task_indices);
        self.downgrade_counter = DOWNGRADE_LIVES;
    }

    /// Reads the job's next ready task from the queue without taking it.
    ///
    /// # Returns
    ///
    /// The next ready task, or [`None`] if the queue is empty.
    pub fn peek_next_task(&self) -> Option<TaskIndex> {
        self.ready_task_queue.front().copied()
    }

    /// Pops the job's next ready task from the queue.
    ///
    /// # Returns
    ///
    /// The next ready task, or [`None`] if the queue is empty.
    pub fn pop_next_task(&mut self) -> Option<TaskIndex> {
        self.ready_task_queue.pop_front()
    }

    /// Empties the job's ready task queue.
    ///
    /// # Returns
    ///
    /// The ready tasks the job still held.
    pub fn take_ready_tasks(&mut self) -> VecDeque<TaskIndex> {
        std::mem::take(&mut self.ready_task_queue)
    }

    /// # Returns
    ///
    /// Whether the job has at least one ready task.
    pub fn has_ready_task(&self) -> bool {
        !self.ready_task_queue.is_empty()
    }

    /// # Returns
    ///
    /// The number of further chances the job has to be refilled before it is downgraded.
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
}

/// The core's registry of job entries, supporting two access methods:
///
/// * Job ID: Lookup through the job ID-to-key mapping.
/// * Job key: Direct lookup using the key returned when the job is created.
#[derive(Default)]
pub struct JobRegistry {
    entries: SlotMap<JobKey, JobEntry>,
    id_to_key: HashMap<JobId, JobKey>,
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
            .id_to_key
            .get(&job_id)
            .and_then(|key| self.entries.get_mut(*key))
        {
            entry.insert_tasks(task_indices);
            return UpsertOutcome::Exist;
        }

        let key = self.entries.insert(JobEntry::new(job_id, task_indices));
        self.id_to_key.insert(job_id, key);
        UpsertOutcome::New(key)
    }

    /// # Returns
    ///
    /// The entry `key` refers to, or [`None`] if the job has been removed from the registry.
    pub fn get_mut(&mut self, key: JobKey) -> Option<&mut JobEntry> {
        self.entries.get_mut(key)
    }

    /// Removes `job_id`'s entry from the registry, after which every key to it fails to resolve.
    ///
    /// # Returns
    ///
    /// The removed entry, or [`None`] if the job was not registered.
    pub fn remove_by_job_id(&mut self, job_id: JobId) -> Option<JobEntry> {
        let key = self.id_to_key.remove(&job_id)?;
        self.entries.remove(key)
    }

    /// Removes the entry `key` refers to, after which every other key to it fails to resolve.
    ///
    /// # Returns
    ///
    /// The removed entry, or [`None`] if the job has already been removed.
    pub fn remove(&mut self, key: JobKey) -> Option<JobEntry> {
        let entry = self.entries.remove(key)?;
        self.id_to_key.remove(&entry.job_id);
        Some(entry)
    }

    /// Drops every registered job.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.id_to_key.clear();
    }

    /// # Returns
    ///
    /// The number of registered jobs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
