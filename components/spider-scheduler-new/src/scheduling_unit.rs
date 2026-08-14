//! The core-private scheduling state of a single resource group.
//!
//! The unit owns everything the core decides with for one group -- its job lists, its pending
//! finalizations, and the write side of its dispatch queue -- so the decision loop can hold a
//! mutable borrow of one group without touching any other.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;

use crate::core::TaskAssignmentIdIssuer;
use crate::dispatch_queue::GlobalDispatchQueue;
use crate::error::JobEntryError;
use crate::error::MakeAssignmentError;
use crate::job_registry::SharedJobEntry;
use crate::resource_group::RgDispatchQueueEndpoints;
use crate::resource_group::RgDispatchQueueReader;
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
    pub active_jobs: Vec<SharedJobEntry>,

    /// The jobs waiting for a slot in [`Self::active_jobs`].
    pub pending_jobs: VecDeque<SharedJobEntry>,

    /// The index into [`Self::active_jobs`] the next regular task is drawn from.
    pub rr_arm: usize,

    /// Whether the group is currently on the core's active resource group list.
    pub is_active: bool,

    finalize_queue: VecDeque<(JobId, FinalizeKind)>,
    num_buffered_commits: usize,
    num_buffered_cleanups: usize,
    dispatch_queue_sender: async_channel::Sender<TaskAssignment>,
    reader: RgDispatchQueueReader,
    downgrade_buffer: Vec<SharedJobEntry>,
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
        endpoints: RgDispatchQueueEndpoints,
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
            dispatch_queue_sender: endpoints.sender,
            reader: endpoints.reader,
            downgrade_buffer: Vec::new(),
            active_job_list_capacity,
        }
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for the group.
    pub fn dispatch_queue_size(&self) -> usize {
        self.reader.len()
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
    pub fn place_new_job(&mut self, entry: SharedJobEntry) {
        if self.active_jobs.len() < self.active_job_list_capacity {
            self.active_jobs.push(entry);
        } else {
            self.pending_jobs.push_back(entry);
        }
    }

    /// Tops the active job list up to capacity from the pending job queue.
    ///
    /// Pending jobs that yield nothing spend a downgrade life, and are collected into
    /// `jobs_to_retire` once they have none left.
    pub fn promote_pending_jobs(&mut self, jobs_to_retire: &mut Vec<JobId>) {
        while self.active_jobs.len() < self.active_job_list_capacity {
            let Some(entry) = self.pop_promotable_job(jobs_to_retire) else {
                break;
            };
            self.active_jobs.push(entry);
        }
    }

    /// Publishes at most one assignment for this group.
    ///
    /// `free` is the tick's remaining free space in the whole dispatch buffer, read but not
    /// modified here -- the caller decrements it once the assignment is published.
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
    /// * [`MakeAssignmentError::DispatchQueueClosed`] if the group's dispatch queue is closed.
    pub fn try_make_assignment(
        &mut self,
        free: usize,
        session_id: SessionId,
        id_issuer: &TaskAssignmentIdIssuer,
        global_queue: &GlobalDispatchQueue,
        jobs_to_retire: &mut Vec<JobId>,
    ) -> Result<(JobId, TaskId), MakeAssignmentError> {
        if !self.has_schedulable_task() {
            return Err(MakeAssignmentError::NoTask);
        }
        if self.dispatch_queue_size() >= free {
            return Err(MakeAssignmentError::DispatchQueueFull);
        }

        let taken = if let Some((job_id, kind)) = self.pop_finalization() {
            TakenTask::Finalization(job_id, kind)
        } else {
            let (entry, task_index) = self
                .take_regular_task(jobs_to_retire)
                .ok_or(MakeAssignmentError::NoTask)?;
            TakenTask::Regular(entry, task_index)
        };
        let (job_id, task_id) = match &taken {
            TakenTask::Finalization(job_id, kind) => (*job_id, TaskId::from(*kind)),
            TakenTask::Regular(entry, task_index) => (entry.job_id(), TaskId::Index(*task_index)),
        };

        // The task is already out of the structure that held it, so a rejected publication has to
        // put it back: the core removes it from the dedup set only on success, and an entry that is
        // in neither place can never be re-admitted.
        if let Err(e) = self.publish(job_id, task_id, session_id, id_issuer, global_queue) {
            self.restore(taken);
            return Err(e);
        }
        Ok((job_id, task_id))
    }

    /// Returns every job buffered for downgrade to the head of the pending job queue, with its
    /// downgrade budget restored.
    pub fn apply_downgrades(&mut self) {
        for entry in std::mem::take(&mut self.downgrade_buffer) {
            entry.reset_downgrade_counter();
            self.pending_jobs.push_front(entry);
        }
    }

    /// Takes the group's next owed finalization.
    ///
    /// # Returns
    ///
    /// The job to finalize and how, or [`None`] if the group owes no finalization.
    fn pop_finalization(&mut self) -> Option<(JobId, FinalizeKind)> {
        let (job_id, kind) = self.finalize_queue.pop_front()?;
        self.discount_finalization(kind);
        Some((job_id, kind))
    }

    /// Returns a task whose publication was rejected to the structure it was taken from.
    fn restore(&mut self, taken: TakenTask) {
        match taken {
            TakenTask::Finalization(job_id, kind) => {
                self.finalize_queue.push_front((job_id, kind));
                self.count_finalization(kind);
            }
            TakenTask::Regular(entry, task_index) => entry.restore_task(task_index),
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

    /// Draws the next regular task from the active job list, rotating the arm and refilling the
    /// list from the pending job queue as jobs run dry.
    ///
    /// # Returns
    ///
    /// The job entry the task was drawn from and the task index to dispatch, or [`None`] if no
    /// active or pending job yields a task.
    fn take_regular_task(
        &mut self,
        jobs_to_retire: &mut Vec<JobId>,
    ) -> Option<(SharedJobEntry, TaskIndex)> {
        let mut remaining_visits = self.active_jobs.len();
        loop {
            if self.active_jobs.is_empty() {
                let entry = self.pop_promotable_job(jobs_to_retire)?;
                self.rr_arm = 0;
                self.active_jobs.push(entry);
                remaining_visits = 1;
            } else if 0 == remaining_visits {
                return None;
            }
            remaining_visits -= 1;

            if self.rr_arm >= self.active_jobs.len() {
                self.rr_arm = 0;
            }
            let entry = self.active_jobs[self.rr_arm].clone();
            match entry.get_next_task() {
                Ok(Some(task_index)) => {
                    self.rr_arm += 1;
                    return Some((entry, task_index));
                }
                Ok(None) => {
                    entry.decrement_downgrade_counter();
                    if 0 == entry.downgrade_counter() {
                        self.downgrade_buffer.push(entry);
                        if self.swap_in_pending_job(jobs_to_retire) {
                            remaining_visits += 1;
                        }
                    } else {
                        self.rr_arm += 1;
                    }
                }
                Err(JobEntryError::Finalized) => {
                    if self.swap_in_pending_job(jobs_to_retire) {
                        remaining_visits += 1;
                    }
                }
            }
        }
    }

    /// Replaces the active job the arm points at with the next promotable pending job.
    ///
    /// # Returns
    ///
    /// Whether a pending job took the vacated slot. When none did, the slot itself is removed.
    fn swap_in_pending_job(&mut self, jobs_to_retire: &mut Vec<JobId>) -> bool {
        if let Some(entry) = self.pop_promotable_job(jobs_to_retire) {
            self.active_jobs[self.rr_arm] = entry;
            true
        } else {
            self.active_jobs.swap_remove(self.rr_arm);
            false
        }
    }

    /// Pops pending jobs until one with a buffered ready task is found, examining each queued job
    /// at most once.
    ///
    /// A job found finalized is discarded outright; one that yields nothing spends a downgrade life
    /// and goes to the back of the queue, or is collected into `jobs_to_retire` if it has none
    /// left.
    ///
    /// # Returns
    ///
    /// The promotable job, or [`None`] if no pending job yields a task.
    fn pop_promotable_job(&mut self, jobs_to_retire: &mut Vec<JobId>) -> Option<SharedJobEntry> {
        let mut remaining_visits = self.pending_jobs.len();
        while 0 != remaining_visits {
            remaining_visits -= 1;
            let entry = self.pending_jobs.pop_front()?;
            if entry.is_finalized() {
                continue;
            }
            if entry.has_ready_task() {
                return Some(entry);
            }
            if 0 == entry.downgrade_counter() {
                jobs_to_retire.push(entry.job_id());
            } else {
                entry.decrement_downgrade_counter();
                self.pending_jobs.push_back(entry);
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
        self.dispatch_queue_sender
            .try_send(assignment)
            .map_err(|_| MakeAssignmentError::DispatchQueueClosed)?;

        // The assignment above must already be in the queue before the hint count is compared
        // against the queue size: reversing the two lets a general execution manager consume a hint
        // for an assignment that is not yet visible.
        //
        // `S` must likewise be sampled before `H`, per the memory-ordering argument in design §8.1.
        let dispatch_queue_size = self.dispatch_queue_size();
        let living_hint = self.reader.living_hint();
        if living_hint.load(Ordering::Acquire) >= dispatch_queue_size {
            return Ok(());
        }

        living_hint.fetch_add(1, Ordering::Release);
        if !global_queue.try_send(self.reader.clone()) {
            // The hint channel is unbounded, so this is unreachable unless it was closed. The
            // counter must still be restored, or the group would be credited with a pop attempt
            // nobody will make.
            living_hint.fetch_sub(1, Ordering::AcqRel);
            tracing::error!(
                rg_id = ? self.rg_id,
                "Failed to publish a dispatch hint."
            );
        }
        Ok(())
    }
}

/// A task taken out of the structure that buffered it, still owed a publication.
enum TakenTask {
    /// A finalization owed by the group, taken off the head of its finalize queue.
    Finalization(JobId, FinalizeKind),

    /// A regular task, taken out of the job entry it belongs to.
    Regular(SharedJobEntry, TaskIndex),
}
