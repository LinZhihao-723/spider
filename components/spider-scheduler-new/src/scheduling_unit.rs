//! The core-private scheduling state of a single resource group.
//!
//! The unit owns everything the core decides with for one group -- its job lists, its pending
//! finalizations, and the write side of its dispatch queue. The jobs themselves are owned by the
//! job registry, so every scheduling position here holds a [`JobKey`] and resolves it against the
//! registry the core hands in; a key that fails to resolve is a job that has been removed.

use std::collections::VecDeque;

use crate::core::TaskAssignmentIdIssuer;
use crate::dispatch_queue::GlobalDispatchQueue;
use crate::error::MakeAssignmentError;
use crate::job_registry::JobKey;
use crate::job_registry::JobRegistry;
use crate::resource_group::RgDispatchQueueEndpoints;
use crate::resource_group::RgDispatchQueueWriter;
use crate::types::FinalizeKind;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskAssignment;
use crate::types::TaskId;
use crate::types::TaskIndex;

/// The scheduling state of one resource group.
pub struct RgSchedulingUnit {
    /// The resource group this unit schedules for.
    pub rg_id: ResourceGroupId,

    /// The jobs assignments are currently drawn from, rotated over by [`Self::rr_arm`].
    pub active_jobs: Vec<JobKey>,

    /// The jobs waiting for a slot in [`Self::active_jobs`].
    pub pending_jobs: VecDeque<JobKey>,

    /// The index into [`Self::active_jobs`] the next regular task is drawn from.
    pub rr_arm: usize,

    /// Whether the group is currently on the core's active resource group list.
    pub is_active: bool,

    finalize_queue: VecDeque<(JobId, FinalizeKind)>,
    num_buffered_commits: usize,
    num_buffered_cleanups: usize,
    writer: RgDispatchQueueWriter,
    downgrade_buffer: Vec<JobKey>,
    active_job_list_capacity: usize,
}

impl RgSchedulingUnit {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created, inactive unit publishing into `endpoints`.
    pub fn new(
        rg_id: ResourceGroupId,
        endpoints: &RgDispatchQueueEndpoints,
        active_job_list_capacity: usize,
    ) -> Self {
        Self {
            rg_id,
            active_jobs: Vec::with_capacity(active_job_list_capacity),
            pending_jobs: VecDeque::new(),
            rr_arm: 0,
            is_active: false,
            finalize_queue: VecDeque::new(),
            num_buffered_commits: 0,
            num_buffered_cleanups: 0,
            writer: endpoints.writer(),
            downgrade_buffer: Vec::new(),
            active_job_list_capacity,
        }
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for the group.
    pub fn dispatch_queue_size(&self) -> usize {
        self.writer.queue_len()
    }

    /// # Returns
    ///
    /// Whether the group holds anything an assignment could still be drawn from.
    pub fn has_schedulable_task(&self) -> bool {
        !self.finalize_queue.is_empty()
            || !self.active_jobs.is_empty()
            || !self.pending_jobs.is_empty()
    }

    /// # Returns
    ///
    /// A tuple containing:
    ///
    /// * The number of commit tasks the group has buffered.
    /// * The number of cleanup tasks the group has buffered.
    pub const fn num_buffered_finalizations(&self) -> (usize, usize) {
        (self.num_buffered_commits, self.num_buffered_cleanups)
    }

    /// Records that `job_id` has reached the finalization named by `kind`.
    pub fn push_finalization(&mut self, job_id: JobId, kind: FinalizeKind) {
        self.finalize_queue.push_back((job_id, kind));
        self.count_finalization(kind);
    }

    /// Gives a newly registered job its scheduling position in this group.
    pub fn place_new_job(&mut self, job_key: JobKey) {
        if self.active_jobs.len() < self.active_job_list_capacity {
            self.active_jobs.push(job_key);
        } else {
            self.pending_jobs.push_back(job_key);
        }
    }

    /// Tops the active job list up to capacity from the pending job queue.
    ///
    /// Pending jobs that yield nothing spend a downgrade life, and are collected into
    /// `jobs_to_retire` once they have none left.
    pub fn promote_pending_jobs(
        &mut self,
        job_registry: &mut JobRegistry,
        jobs_to_retire: &mut Vec<JobKey>,
    ) {
        while self.active_jobs.len() < self.active_job_list_capacity {
            let Some(job_key) = self.pop_promotable_job(job_registry, jobs_to_retire) else {
                break;
            };
            self.active_jobs.push(job_key);
        }
    }

    /// Publishes at most one assignment for this group.
    ///
    /// `free` is the tick's remaining free space in the whole dispatch buffer, read but not
    /// modified here -- the caller decrements it once the assignment is published.
    ///
    /// The unit and the job arena are borrowed mutably at the same time, which is why this is a
    /// method on the unit rather than on the core: the caller destructures the core's fields so
    /// that the two borrows name different fields and the borrow checker accepts them.
    ///
    /// # Returns
    ///
    /// The job and task the published assignment carries, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`MakeAssignmentError::NoTask`] if the group has nothing left to schedule.
    /// * [`MakeAssignmentError::DispatchQueueFull`] if the group's queue occupancy has reached the
    ///   admission threshold.
    /// * Forwards [`Self::publish`]'s return values on failure.
    pub fn try_make_assignment(
        &mut self,
        free: usize,
        session_id: SessionId,
        id_issuer: &TaskAssignmentIdIssuer,
        global_queue: &GlobalDispatchQueue,
        job_registry: &mut JobRegistry,
        jobs_to_retire: &mut Vec<JobKey>,
    ) -> Result<(JobId, TaskId), MakeAssignmentError> {
        if !self.has_schedulable_task() {
            return Err(MakeAssignmentError::NoTask);
        }
        if self.dispatch_queue_size() >= free {
            return Err(MakeAssignmentError::DispatchQueueFull);
        }

        // The task is only taken out of the structure that buffered it once its publication has
        // succeeded: the core removes it from the dedup set only on success, so a task dropped by a
        // rejected publication would be in neither place and could never be re-admitted.
        if let Some((job_id, kind)) = self.peek_finalization() {
            let task_id = TaskId::from(kind);
            self.publish(job_id, task_id, session_id, id_issuer, global_queue)?;
            self.commit_finalization();
            return Ok((job_id, task_id));
        }

        let (job_key, job_id, task_index) = self
            .peek_regular_task(job_registry, jobs_to_retire)
            .ok_or(MakeAssignmentError::NoTask)?;
        let task_id = TaskId::Index(task_index);
        self.publish(job_id, task_id, session_id, id_issuer, global_queue)?;
        Self::commit_regular_task(job_key, job_registry);
        Ok((job_id, task_id))
    }

    /// Returns every job buffered for downgrade to the head of the pending job queue, with its
    /// downgrade budget restored.
    pub fn apply_downgrades(&mut self, job_registry: &mut JobRegistry) {
        for job_key in std::mem::take(&mut self.downgrade_buffer) {
            if let Some(entry) = job_registry.get_mut(job_key) {
                entry.reset_downgrade_counter();
            }
            self.pending_jobs.push_front(job_key);
        }
    }

    /// Reads the group's next owed finalization without taking it.
    ///
    /// # Returns
    ///
    /// The job to finalize and how, or [`None`] if the group owes no finalization.
    fn peek_finalization(&self) -> Option<(JobId, FinalizeKind)> {
        self.finalize_queue.front().copied()
    }

    /// Takes the finalization read by [`Self::peek_finalization`] off the group's finalize queue.
    ///
    /// The call is a no-op if the group owes no finalization.
    fn commit_finalization(&mut self) {
        if let Some((_, kind)) = self.finalize_queue.pop_front() {
            self.discount_finalization(kind);
        }
    }

    /// Adds one buffered finalization of `kind` to the group's running counts.
    const fn count_finalization(&mut self, kind: FinalizeKind) {
        match kind {
            FinalizeKind::Commit => self.num_buffered_commits += 1,
            FinalizeKind::Cleanup => self.num_buffered_cleanups += 1,
        }
    }

    /// Takes one buffered finalization of `kind` off the group's running counts.
    const fn discount_finalization(&mut self, kind: FinalizeKind) {
        match kind {
            FinalizeKind::Commit => {
                self.num_buffered_commits = self.num_buffered_commits.saturating_sub(1);
            }
            FinalizeKind::Cleanup => {
                self.num_buffered_cleanups = self.num_buffered_cleanups.saturating_sub(1);
            }
        }
    }

    /// Finds the next regular task to dispatch without taking it out of the job that buffers it,
    /// rotating the arm and refilling the active job list from the pending job queue as jobs run
    /// dry.
    ///
    /// # Returns
    ///
    /// A tuple on success, containing:
    ///
    /// * The key of the job the task was found in.
    /// * That job's ID.
    /// * The task index to dispatch.
    ///
    /// [`None`] is returned if no active or pending job yields a task.
    fn peek_regular_task(
        &mut self,
        job_registry: &mut JobRegistry,
        jobs_to_retire: &mut Vec<JobKey>,
    ) -> Option<(JobKey, JobId, TaskIndex)> {
        let mut remaining_visits = self.active_jobs.len();
        loop {
            if self.active_jobs.is_empty() {
                let job_key = self.pop_promotable_job(job_registry, jobs_to_retire)?;
                self.rr_arm = 0;
                self.active_jobs.push(job_key);
                remaining_visits = 1;
            } else if 0 == remaining_visits {
                return None;
            }
            remaining_visits -= 1;

            if self.rr_arm >= self.active_jobs.len() {
                self.rr_arm = 0;
            }
            let job_key = self.active_jobs[self.rr_arm];
            let Some(entry) = job_registry.get_mut(job_key) else {
                if self.swap_in_pending_job(job_registry, jobs_to_retire) {
                    remaining_visits += 1;
                }
                continue;
            };
            let Some(task_index) = entry.peek_next_task() else {
                entry.decrement_downgrade_counter();
                if 0 == entry.downgrade_counter() {
                    self.downgrade_buffer.push(job_key);
                    if self.swap_in_pending_job(job_registry, jobs_to_retire) {
                        remaining_visits += 1;
                    }
                } else {
                    self.rr_arm += 1;
                }
                continue;
            };
            let job_id = entry.job_id();
            self.rr_arm += 1;
            return Some((job_key, job_id, task_index));
        }
    }

    /// Takes the task read by [`Self::peek_regular_task`] out of the job `job_key` refers to.
    ///
    /// The call is a no-op if the job has been removed from the registry.
    fn commit_regular_task(job_key: JobKey, job_registry: &mut JobRegistry) {
        if let Some(entry) = job_registry.get_mut(job_key) {
            entry.pop_next_task();
        }
    }

    /// Replaces the active job the arm points at with the next promotable pending job.
    ///
    /// # Returns
    ///
    /// Whether a pending job took the vacated slot. When none did, the slot itself is removed.
    fn swap_in_pending_job(
        &mut self,
        job_registry: &mut JobRegistry,
        jobs_to_retire: &mut Vec<JobKey>,
    ) -> bool {
        if let Some(job_key) = self.pop_promotable_job(job_registry, jobs_to_retire) {
            self.active_jobs[self.rr_arm] = job_key;
            true
        } else {
            self.active_jobs.swap_remove(self.rr_arm);
            false
        }
    }

    /// Pops pending jobs until one with a buffered ready task is found, examining each queued job
    /// at most once.
    ///
    /// A key that no longer resolves belongs to a job that has been removed from the registry and
    /// is discarded outright; a job that yields nothing spends a downgrade life and goes to the
    /// back of the queue, or is collected into `jobs_to_retire` if it has none left.
    ///
    /// # Returns
    ///
    /// The key of the promotable job, or [`None`] if no pending job yields a task.
    fn pop_promotable_job(
        &mut self,
        job_registry: &mut JobRegistry,
        jobs_to_retire: &mut Vec<JobKey>,
    ) -> Option<JobKey> {
        let mut remaining_visits = self.pending_jobs.len();
        while 0 != remaining_visits {
            remaining_visits -= 1;
            let job_key = self.pending_jobs.pop_front()?;
            let Some(entry) = job_registry.get_mut(job_key) else {
                continue;
            };
            if entry.has_ready_task() {
                return Some(job_key);
            }
            if 0 == entry.downgrade_counter() {
                jobs_to_retire.push(job_key);
            } else {
                entry.decrement_downgrade_counter();
                self.pending_jobs.push_back(job_key);
            }
        }
        None
    }

    /// Publishes one assignment into the group's dispatch queue and, when the group's coverage
    /// requires it, a hint into the global dispatch queue.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`MakeAssignmentError::DispatchQueueClosed`] if the group's queue is closed.
    fn publish(
        &self,
        job_id: JobId,
        task_id: TaskId,
        session_id: SessionId,
        id_issuer: &TaskAssignmentIdIssuer,
        global_queue: &GlobalDispatchQueue,
    ) -> Result<(), MakeAssignmentError> {
        let assignment = TaskAssignment {
            id: id_issuer.next(),
            resource_group_id: self.rg_id,
            job_id,
            task_id,
            session_id,
        };
        self.writer
            .try_send(assignment)
            .map_err(|_| MakeAssignmentError::DispatchQueueClosed)?;

        // The assignment above must already be in the queue before the hint count is compared
        // against the queue size: reversing the two lets a general execution manager consume a hint
        // for an assignment that is not yet visible.
        //
        // `S` must likewise be sampled before `H`, per the memory-ordering argument in design §8.1.
        let dispatch_queue_size = self.writer.queue_len();
        if self.writer.living_hint() >= dispatch_queue_size {
            return Ok(());
        }

        self.writer.increment_living_hint();
        if !global_queue.try_send(self.writer.hint()) {
            // The hint channel is unbounded, so this is unreachable unless it was closed. The
            // counter must still be restored, or the group would be credited with a pop attempt
            // nobody will make.
            self.writer.decrement_living_hint();
            tracing::error!(
                rg_id = ? self.rg_id,
                "Failed to publish a dispatch hint."
            );
        }
        Ok(())
    }
}
