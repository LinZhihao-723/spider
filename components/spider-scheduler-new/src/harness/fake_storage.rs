//! An in-memory stand-in for the storage-owned inbound queue.

use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::StorageClientError;
use crate::storage_client::SchedulerStorageClient;
use crate::types::InboundEntry;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskId;

/// The shape of the workload a [`FakeStorage`] serves.
///
/// The workload is densely numbered: resource group IDs are `0..num_resource_groups`, job IDs are
/// `0..num_resource_groups * num_jobs_per_group` laid out group by group, and each job's regular
/// tasks are [`TaskId::Index`] over `0..num_tasks_per_job`.
#[derive(Clone, Debug)]
pub struct FakeStorageConfig {
    /// The number of resource groups the workload spans.
    pub num_resource_groups: usize,

    /// The number of jobs each resource group owns.
    pub num_jobs_per_group: usize,

    /// The number of regular tasks each job owns.
    pub num_tasks_per_job: usize,

    /// Emits a commit-ready entry for a job once all of its regular tasks have been reported
    /// complete.
    pub emit_commit_ready: bool,

    /// Emits a cleanup-ready entry for a job once all of its regular tasks have been reported
    /// complete.
    ///
    /// A job finalizes exactly once, so when commit-ready entries are emitted as well the
    /// workload's jobs alternate between the two lanes rather than appearing in both.
    pub emit_cleanup_ready: bool,
}

/// An in-memory inbound queue serving a deterministic workload to the prototype core.
///
/// Regular tasks are handed out one resource group at a time in rotation, so the order the core
/// observes carries no per-group bias of storage's own and admission fairness measures the
/// scheduler rather than the workload source.
///
/// Every clone shares the same state, so the harness may hand clones to the core, the gRPC service,
/// and the test body alike.
#[derive(Clone, Debug)]
pub struct FakeStorage {
    config: FakeStorageConfig,
    state: Arc<Mutex<FakeStorageState>>,
}

impl FakeStorage {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A fake storage armed with every regular task the configuration describes, serving session 0.
    #[must_use]
    pub fn new(config: FakeStorageConfig) -> Self {
        let state = FakeStorageState::new(&config);
        Self {
            config,
            state: Arc::new(Mutex::new(state)),
        }
    }

    /// # Returns
    ///
    /// Every regular task the configuration will ever emit, against which a run's dispatches are
    /// checked for exactly-once delivery.
    #[must_use]
    pub fn expected_regular_tasks(&self) -> HashSet<(JobId, TaskId)> {
        let num_tasks_per_job = self.config.num_tasks_per_job;
        (0..num_jobs(&self.config))
            .flat_map(|job_index| {
                let job_id = job_id_of(job_index);
                (0..num_tasks_per_job).map(move |task_index| (job_id, TaskId::Index(task_index)))
            })
            .collect()
    }

    /// # Returns
    ///
    /// Every finalization task the configuration will ever emit, against which a run's dispatches
    /// are checked for exactly-once delivery.
    #[must_use]
    pub fn expected_finalization_tasks(&self) -> HashSet<(JobId, TaskId)> {
        (0..num_jobs(&self.config))
            .filter_map(|job_index| {
                finalize_task_id_of(&self.config, job_index)
                    .map(|task_id| (job_id_of(job_index), task_id))
            })
            .collect()
    }

    /// # Returns
    ///
    /// The number of regular tasks the configuration will ever emit.
    #[must_use]
    pub const fn total_regular_tasks(&self) -> usize {
        num_jobs(&self.config) * self.config.num_tasks_per_job
    }

    /// # Returns
    ///
    /// Whether every task of the configured workload has been reported complete: every regular
    /// task, plus every finalization task the configuration emits.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    #[must_use]
    pub fn is_drained(&self) -> bool {
        let num_tasks_per_job = self.config.num_tasks_per_job;
        let config = &self.config;
        let state = self.lock_state();

        state
            .num_completed_regular
            .iter()
            .all(|num_completed| *num_completed >= num_tasks_per_job)
            && state
                .finalize_completed
                .iter()
                .enumerate()
                .all(|(job_index, completed)| {
                    *completed || finalize_task_id_of(config, job_index).is_none()
                })
    }

    /// Reports a task as executed, so the commit-ready lane can advance.
    ///
    /// A task outside the configured workload, and a task reported more than once, are both
    /// ignored.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    pub fn complete_task(&self, job_id: JobId, task_id: TaskId) {
        let Some(job_index) = job_index_of(&self.config, job_id) else {
            return;
        };
        self.lock_state()
            .record_completion(&self.config, job_index, task_id);
    }

    /// Bumps the session and re-arms every task that has not been reported complete, mirroring a
    /// storage restart that replays its unfinished work.
    ///
    /// # Returns
    ///
    /// The session ID every subsequent poll is served under.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    #[must_use]
    pub fn bump_session(&self) -> SessionId {
        let mut state = self.lock_state();
        state.session_id += 1;
        state.rearm(&self.config);
        state.session_id
    }

    /// Takes up to `max_items` regular tasks, rotating across resource groups.
    ///
    /// # Returns
    ///
    /// A tuple containing the current session ID and the tasks taken.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn take_ready(&self, max_items: usize) -> (SessionId, Vec<InboundEntry>) {
        let mut state = self.lock_state();
        let entries = state.take_regular(max_items);
        (state.session_id, entries)
    }

    /// Takes up to `max_items` commit-ready jobs.
    ///
    /// # Returns
    ///
    /// A tuple containing the current session ID and the commit tasks taken.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn take_commit_ready(&self, max_items: usize) -> (SessionId, Vec<InboundEntry>) {
        let mut state = self.lock_state();
        let entries = state.take_finalizations(FinalizeLane::Commit, max_items);
        (state.session_id, entries)
    }

    /// Takes up to `max_items` cleanup-ready jobs.
    ///
    /// # Returns
    ///
    /// A tuple containing the current session ID and the cleanup tasks taken.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn take_cleanup_ready(&self, max_items: usize) -> (SessionId, Vec<InboundEntry>) {
        let mut state = self.lock_state();
        let entries = state.take_finalizations(FinalizeLane::Cleanup, max_items);
        (state.session_id, entries)
    }

    /// # Returns
    ///
    /// A guard over the state shared by every clone of this storage.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn lock_state(&self) -> MutexGuard<'_, FakeStorageState> {
        self.state
            .lock()
            .expect("fake storage state mutex is poisoned")
    }
}

#[async_trait]
impl SchedulerStorageClient for FakeStorage {
    async fn poll_ready(
        &self,
        max_items: usize,
        wait: Duration,
    ) -> Result<(SessionId, Vec<InboundEntry>), StorageClientError> {
        let (session_id, entries) = self.take_ready(max_items);
        if !entries.is_empty() {
            return Ok((session_id, entries));
        }

        tokio::time::sleep(wait).await;
        Ok(self.take_ready(max_items))
    }

    async fn poll_commit_ready(
        &self,
        max_items: usize,
        wait: Duration,
    ) -> Result<(SessionId, Vec<InboundEntry>), StorageClientError> {
        let (session_id, entries) = self.take_commit_ready(max_items);
        if !entries.is_empty() {
            return Ok((session_id, entries));
        }

        tokio::time::sleep(wait).await;
        Ok(self.take_commit_ready(max_items))
    }

    async fn poll_cleanup_ready(
        &self,
        max_items: usize,
        wait: Duration,
    ) -> Result<(SessionId, Vec<InboundEntry>), StorageClientError> {
        let (session_id, entries) = self.take_cleanup_ready(max_items);
        if !entries.is_empty() {
            return Ok((session_id, entries));
        }

        tokio::time::sleep(wait).await;
        Ok(self.take_cleanup_ready(max_items))
    }
}

/// The inbound-queue lane a finalization task is served through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FinalizeLane {
    /// The commit-ready lane.
    Commit,

    /// The cleanup-ready lane.
    Cleanup,
}

/// The state shared by every clone of a [`FakeStorage`].
#[derive(Debug)]
struct FakeStorageState {
    session_id: SessionId,
    pending_regular: Vec<VecDeque<InboundEntry>>,
    regular_arm: usize,
    pending_commit: VecDeque<InboundEntry>,
    pending_cleanup: VecDeque<InboundEntry>,
    completed_regular: HashSet<(JobId, TaskId)>,
    num_completed_regular: Vec<usize>,
    finalize_emitted: Vec<bool>,
    finalize_completed: Vec<bool>,
}

impl FakeStorageState {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A state with nothing completed and every task of the configuration armed.
    fn new(config: &FakeStorageConfig) -> Self {
        let num_jobs = num_jobs(config);
        let mut state = Self {
            session_id: 0,
            pending_regular: Vec::new(),
            regular_arm: 0,
            pending_commit: VecDeque::new(),
            pending_cleanup: VecDeque::new(),
            completed_regular: HashSet::new(),
            num_completed_regular: vec![0; num_jobs],
            finalize_emitted: vec![false; num_jobs],
            finalize_completed: vec![false; num_jobs],
        };
        state.rearm(config);
        state
    }

    /// Re-arms every task that has not been reported complete.
    fn rearm(&mut self, config: &FakeStorageConfig) {
        let pending_regular = build_pending_regular(config, &self.completed_regular);
        self.pending_regular = pending_regular;
        self.regular_arm = 0;
        self.pending_commit.clear();
        self.pending_cleanup.clear();
        self.finalize_emitted.fill(false);
        for job_index in 0..self.finalize_emitted.len() {
            self.arm_finalization(config, job_index);
        }
    }

    /// Takes up to `max_items` regular tasks, visiting the resource groups in rotation and
    /// resuming where the previous call left off.
    ///
    /// # Returns
    ///
    /// The tasks taken, which is empty once every armed task has been handed out.
    fn take_regular(&mut self, max_items: usize) -> Vec<InboundEntry> {
        let num_groups = self.pending_regular.len();
        if 0 == num_groups {
            return Vec::new();
        }

        let mut entries = Vec::new();
        let mut num_groups_skipped = 0;
        while entries.len() < max_items && num_groups_skipped < num_groups {
            if let Some(entry) = self.pending_regular[self.regular_arm].pop_front() {
                entries.push(entry);
                num_groups_skipped = 0;
            } else {
                num_groups_skipped += 1;
            }
            self.regular_arm = (self.regular_arm + 1) % num_groups;
        }
        entries
    }

    /// Takes up to `max_items` finalization tasks from `lane`.
    ///
    /// # Returns
    ///
    /// The finalization tasks taken, which is empty while no job served by `lane` has finished its
    /// regular tasks.
    fn take_finalizations(&mut self, lane: FinalizeLane, max_items: usize) -> Vec<InboundEntry> {
        let pending = match lane {
            FinalizeLane::Commit => &mut self.pending_commit,
            FinalizeLane::Cleanup => &mut self.pending_cleanup,
        };
        let num_taken = max_items.min(pending.len());
        pending.drain(..num_taken).collect()
    }

    /// Records `task_id` of the job at `job_index` as executed, ignoring a repeated report.
    fn record_completion(&mut self, config: &FakeStorageConfig, job_index: usize, task_id: TaskId) {
        match task_id {
            TaskId::Index(_) => {
                if !self
                    .completed_regular
                    .insert((job_id_of(job_index), task_id))
                {
                    return;
                }
                self.num_completed_regular[job_index] += 1;
                self.arm_finalization(config, job_index);
            }
            TaskId::Commit | TaskId::Cleanup => self.finalize_completed[job_index] = true,
        }
    }

    /// Arms the finalization task of the job at `job_index` if the configuration gives that job one
    /// and the job has finished every regular task without its finalization having been armed
    /// already.
    fn arm_finalization(&mut self, config: &FakeStorageConfig, job_index: usize) {
        let Some(task_id) = finalize_task_id_of(config, job_index) else {
            return;
        };
        if self.finalize_emitted[job_index]
            || self.finalize_completed[job_index]
            || self.num_completed_regular[job_index] < config.num_tasks_per_job
        {
            return;
        }

        self.finalize_emitted[job_index] = true;
        let entry = InboundEntry {
            resource_group_id: resource_group_id_of(config, job_index),
            job_id: job_id_of(job_index),
            task_id,
        };
        if TaskId::Cleanup == task_id {
            self.pending_cleanup.push_back(entry);
        } else {
            self.pending_commit.push_back(entry);
        }
    }
}

/// # Returns
///
/// The finalization task the configuration gives the job at `job_index`, or [`None`] if the
/// configuration emits no finalization for it.
const fn finalize_task_id_of(config: &FakeStorageConfig, job_index: usize) -> Option<TaskId> {
    match (config.emit_commit_ready, config.emit_cleanup_ready) {
        (true, true) => {
            if job_index.is_multiple_of(2) {
                Some(TaskId::Commit)
            } else {
                Some(TaskId::Cleanup)
            }
        }
        (true, false) => Some(TaskId::Commit),
        (false, true) => Some(TaskId::Cleanup),
        (false, false) => None,
    }
}

/// Builds the per-resource-group queues of regular tasks, skipping the ones already completed.
///
/// # Returns
///
/// One queue per resource group, indexed by resource group index.
fn build_pending_regular(
    config: &FakeStorageConfig,
    completed_regular: &HashSet<(JobId, TaskId)>,
) -> Vec<VecDeque<InboundEntry>> {
    (0..config.num_resource_groups)
        .map(|group_index| {
            let resource_group_id = ResourceGroupId::from(index_to_raw_id(group_index));
            (0..config.num_jobs_per_group)
                .flat_map(|job_index_in_group| {
                    let job_id =
                        job_id_of(group_index * config.num_jobs_per_group + job_index_in_group);
                    (0..config.num_tasks_per_job).map(move |task_index| InboundEntry {
                        resource_group_id,
                        job_id,
                        task_id: TaskId::Index(task_index),
                    })
                })
                .filter(|entry| !completed_regular.contains(&(entry.job_id, entry.task_id)))
                .collect()
        })
        .collect()
}

/// # Returns
///
/// The number of jobs the configuration describes.
const fn num_jobs(config: &FakeStorageConfig) -> usize {
    config.num_resource_groups * config.num_jobs_per_group
}

/// # Returns
///
/// The ID of the job at `job_index`.
fn job_id_of(job_index: usize) -> JobId {
    JobId::from(index_to_raw_id(job_index))
}

/// # Returns
///
/// The ID of the resource group owning the job at `job_index`.
fn resource_group_id_of(config: &FakeStorageConfig, job_index: usize) -> ResourceGroupId {
    let group_index = job_index
        .checked_div(config.num_jobs_per_group)
        .unwrap_or(0);
    ResourceGroupId::from(index_to_raw_id(group_index))
}

/// # Returns
///
/// The index of `job_id` in the configured workload, or `None` if it names no configured job.
fn job_index_of(config: &FakeStorageConfig, job_id: JobId) -> Option<usize> {
    usize::try_from(job_id.get())
        .ok()
        .filter(|job_index| *job_index < num_jobs(config))
}

/// # Returns
///
/// `index` as the raw value an ID wraps.
///
/// # Panics
///
/// Panics if `index` exceeds the range of a raw ID.
fn index_to_raw_id(index: usize) -> u64 {
    u64::try_from(index).expect("dense index exceeds the raw ID range")
}
