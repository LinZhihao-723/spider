//! An in-memory stand-in for the storage-owned inbound queue.

use std::collections::HashSet;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;

use crate::bench::JobProgress;
use crate::bench::JobProgressMap;
use crate::bench::results::duration_to_nanos;
use crate::error::StorageClientError;
use crate::storage_client::SchedulerStorageClient;
use crate::types::InboundEntry;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskId;
use crate::types::TaskIndex;

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

/// How a [`FakeStorage`] paces the workload it serves.
///
/// Every field defaults to the unpaced behaviour a correctness scenario expects: the whole workload
/// is armed at once, a poll response is bounded only by the caller's `max_items`, and nothing is
/// timestamped.
#[derive(Clone, Debug, Default)]
pub struct ReleaseConfig {
    /// The largest number of regular tasks one poll response may carry, applied on top of the
    /// caller's `max_items`. [`None`] leaves a response bounded by `max_items` alone.
    pub inbound_wave_size: Option<NonZeroUsize>,

    /// The number of jobs each resource group contributes to one batch. [`None`] arms every job of
    /// the workload at once.
    ///
    /// Batch `n + 1` is armed once every job of batch `n` has had all of its regular tasks
    /// reported complete. That release rule is the benchmark contract's own interpretation of
    /// "jobs are created in batches" rather than something storage dictates, so it is stated
    /// here where a reader of the configuration meets it.
    ///
    /// A workload whose jobs own no regular task can never report a batch complete and is armed in
    /// one batch whatever this field says.
    pub jobs_per_batch: Option<NonZeroUsize>,

    /// Where the storage records the instant it first emits a task of each job. [`None`] records
    /// nothing.
    pub first_seen_recorder: Option<FirstSeenRecorder>,
}

/// Records into a shared progress map when a [`FakeStorage`] first emits a task of a job, which is
/// where a job's end-to-end time starts.
///
/// The map is the same one the dispatch service records completions into, so a job's two endpoints
/// are measured against one clock: timestamps are nanoseconds since the run origin both sides were
/// built with.
#[derive(Clone, Debug)]
pub struct FirstSeenRecorder {
    progress: Arc<JobProgressMap>,
    run_origin: Instant,
}

impl FirstSeenRecorder {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created recorder writing into `progress`, timestamping against `run_origin`.
    #[must_use]
    pub const fn new(progress: Arc<JobProgressMap>, run_origin: Instant) -> Self {
        Self {
            progress,
            run_origin,
        }
    }

    /// # Returns
    ///
    /// The progress map this recorder writes into.
    #[must_use]
    pub const fn progress(&self) -> &Arc<JobProgressMap> {
        &self.progress
    }

    /// Records the job as first seen now, keeping the earliest observation.
    fn note(&self, resource_group_id: ResourceGroupId, job_id: JobId) {
        let nanos = duration_to_nanos(self.run_origin.elapsed());
        self.progress
            .entry(job_id)
            .or_insert_with(|| JobProgress::new(resource_group_id, nanos))
            .note_first_seen(nanos);
    }
}

/// An in-memory inbound queue serving a deterministic workload to the prototype core.
///
/// Regular tasks are handed out one resource group at a time in rotation, so the order the core
/// observes carries no per-group bias of storage's own and admission fairness measures the
/// scheduler rather than the workload source. How much of the workload is armed, and how much of it
/// one response may carry, is set by the [`ReleaseConfig`].
///
/// Completion reports are recorded without taking the lock a poll takes, because the benchmark
/// reports one completion per dispatched task and would otherwise serialize the whole workload
/// behind the queue that generates it.
///
/// Every clone shares the same state, so the harness may hand clones to the core, the gRPC service,
/// and the test body alike.
#[derive(Clone, Debug)]
pub struct FakeStorage {
    config: FakeStorageConfig,
    release: ReleaseConfig,
    shared: Arc<Shared>,
}

impl FakeStorage {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A fake storage armed with every regular task the configuration describes, serving session 0.
    #[must_use]
    pub fn new(config: FakeStorageConfig) -> Self {
        Self::with_release_config(config, ReleaseConfig::default())
    }

    /// Factory function.
    ///
    /// # Returns
    ///
    /// A fake storage serving `config`'s workload at the pace `release` sets, under session 0.
    #[must_use]
    pub fn with_release_config(config: FakeStorageConfig, release: ReleaseConfig) -> Self {
        let batch_plan = BatchPlan::new(&config, &release);
        let completions = Completions::new(&config);
        let regular = RegularState::new(&config, batch_plan, &completions);
        let finalize = FinalizeState::new(&config, &completions);

        Self {
            config,
            release,
            shared: Arc::new(Shared {
                session_id: AtomicU64::new(0),
                batch_plan,
                completions,
                regular: Mutex::new(regular),
                finalize: Mutex::new(finalize),
            }),
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
    /// The number of finalization tasks the configuration will ever emit.
    #[must_use]
    pub const fn total_finalization_tasks(&self) -> usize {
        if self.config.emit_commit_ready || self.config.emit_cleanup_ready {
            num_jobs(&self.config)
        } else {
            0
        }
    }

    /// # Returns
    ///
    /// The number of tasks the configuration will ever emit, regular and finalization alike.
    #[must_use]
    pub const fn total_tasks(&self) -> usize {
        self.total_regular_tasks() + self.total_finalization_tasks()
    }

    /// # Returns
    ///
    /// The number of tasks reported complete so far, regular and finalization alike.
    #[must_use]
    pub fn num_completed_tasks(&self) -> usize {
        let completions = &self.shared.completions;

        completions.num_completed_regular.load(Ordering::Relaxed)
            + completions
                .num_completed_finalizations
                .load(Ordering::Relaxed)
    }

    /// # Returns
    ///
    /// Where the storage records the instant it first emits a task of each job, or [`None`] if it
    /// records nothing.
    #[must_use]
    pub const fn first_seen_recorder(&self) -> Option<&FirstSeenRecorder> {
        self.release.first_seen_recorder.as_ref()
    }

    /// # Returns
    ///
    /// Whether every task of the configured workload has been reported complete: every regular
    /// task, plus every finalization task the configuration emits.
    #[must_use]
    pub fn is_drained(&self) -> bool {
        let completions = &self.shared.completions;

        completions.num_completed_regular.load(Ordering::Relaxed) >= self.total_regular_tasks()
            && completions
                .num_completed_finalizations
                .load(Ordering::Relaxed)
                >= self.total_finalization_tasks()
    }

    /// Reports a task as executed, so the commit-ready lane and the job batches can advance.
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
        match task_id {
            TaskId::Index(task_index) => {
                if !self
                    .shared
                    .completions
                    .record_regular(&self.config, job_index, task_index)
                {
                    return;
                }
                self.lock_finalize()
                    .arm(&self.config, &self.shared.completions, job_index);
                self.advance_batch();
            }
            TaskId::Commit | TaskId::Cleanup => self
                .shared
                .completions
                .record_finalization(&self.config, job_index),
        }
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
        let mut regular = self.lock_regular();
        let session_id = self.shared.session_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.lock_finalize()
            .rearm(&self.config, &self.shared.completions);
        regular.rearm(
            &self.config,
            self.shared.batch_plan,
            &self.shared.completions,
        );
        drop(regular);

        session_id
    }

    /// Takes up to `max_items` regular tasks, rotating across resource groups and never exceeding
    /// the configured inbound wave size.
    ///
    /// # Returns
    ///
    /// A tuple containing the current session ID and the tasks taken.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn take_ready(&self, max_items: usize) -> (SessionId, Vec<InboundEntry>) {
        let max_items = self
            .release
            .inbound_wave_size
            .map_or(max_items, |wave_size| max_items.min(wave_size.get()));
        let mut state = self.lock_regular();
        let session_id = self.shared.session_id.load(Ordering::Relaxed);
        let (entries, first_seen) = state.take_regular(&self.config, max_items);
        drop(state);

        if let Some(recorder) = &self.release.first_seen_recorder {
            for entry in first_seen {
                recorder.note(entry.resource_group_id, entry.job_id);
            }
        }

        (session_id, entries)
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
        let entries = self.lock_finalize().take(FinalizeLane::Commit, max_items);

        (self.shared.session_id.load(Ordering::Relaxed), entries)
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
        let entries = self.lock_finalize().take(FinalizeLane::Cleanup, max_items);

        (self.shared.session_id.load(Ordering::Relaxed), entries)
    }

    /// Arms the next batch of jobs if the batch now current has no job left to complete.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn advance_batch(&self) {
        let completions = &self.shared.completions;
        if self.shared.batch_plan.num_batches <= 1
            || 1 != completions
                .num_jobs_left_in_batch
                .fetch_sub(1, Ordering::AcqRel)
        {
            return;
        }
        self.lock_regular()
            .arm_next_batch(&self.config, self.shared.batch_plan, completions);
    }

    /// # Returns
    ///
    /// A guard over the regular lane's state, shared by every clone of this storage.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn lock_regular(&self) -> MutexGuard<'_, RegularState> {
        self.shared
            .regular
            .lock()
            .expect("fake storage regular state mutex is poisoned")
    }

    /// # Returns
    ///
    /// A guard over the finalization lanes' state, shared by every clone of this storage.
    ///
    /// # Panics
    ///
    /// Panics if the shared state's mutex is poisoned.
    fn lock_finalize(&self) -> MutexGuard<'_, FinalizeState> {
        self.shared
            .finalize
            .lock()
            .expect("fake storage finalize state mutex is poisoned")
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

/// The number of regular tasks one word of the completion bitmap tracks.
const COMPLETION_BITS_PER_WORD: usize = u64::BITS as usize;

/// The inbound-queue lane a finalization task is served through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FinalizeLane {
    /// The commit-ready lane.
    Commit,

    /// The cleanup-ready lane.
    Cleanup,
}

/// The batches a workload's jobs are armed in.
///
/// A batch takes the same slice of job indices from every resource group, so batch `n` covers
/// `jobs_per_batch` jobs of each group.
#[derive(Clone, Copy, Debug)]
struct BatchPlan {
    num_resource_groups: usize,
    num_jobs_per_group: usize,
    jobs_per_batch: usize,
    num_batches: usize,
}

impl BatchPlan {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// The plan `release` asks for over `config`'s jobs, which is a single batch when `release`
    /// does not batch.
    fn new(config: &FakeStorageConfig, release: &ReleaseConfig) -> Self {
        let num_jobs_per_group = config.num_jobs_per_group;
        let jobs_per_batch = release
            .jobs_per_batch
            .filter(|_| 0 != config.num_tasks_per_job)
            .map_or(num_jobs_per_group, NonZeroUsize::get)
            .clamp(1, num_jobs_per_group.max(1));

        Self {
            num_resource_groups: config.num_resource_groups,
            num_jobs_per_group,
            jobs_per_batch,
            num_batches: num_jobs_per_group.div_ceil(jobs_per_batch),
        }
    }

    /// # Returns
    ///
    /// The half-open range of per-group job indices batch `batch_index` covers.
    fn job_range(&self, batch_index: usize) -> (usize, usize) {
        let start = batch_index
            .saturating_mul(self.jobs_per_batch)
            .min(self.num_jobs_per_group);
        let end = start
            .saturating_add(self.jobs_per_batch)
            .min(self.num_jobs_per_group);

        (start, end)
    }

    /// # Returns
    ///
    /// The number of jobs batch `batch_index` covers across every resource group.
    fn num_jobs_in(&self, batch_index: usize) -> usize {
        let (start, end) = self.job_range(batch_index);

        (end - start) * self.num_resource_groups
    }
}

/// The state shared by every clone of a [`FakeStorage`].
///
/// The two lanes hold their own locks and the completion tallies hold none, so the report of an
/// executed task never waits on the generation of the next poll response.
#[derive(Debug)]
struct Shared {
    session_id: AtomicU64,
    batch_plan: BatchPlan,
    completions: Completions,
    regular: Mutex<RegularState>,
    finalize: Mutex<FinalizeState>,
}

/// The armed regular tasks, one queue per resource group.
#[derive(Debug)]
struct RegularState {
    pending: Vec<VecDeque<InboundEntry>>,
    arm: usize,
    num_armed_batches: usize,
    first_seen_noted: Vec<bool>,
}

impl RegularState {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A state with the plan's first batch armed and nothing emitted yet.
    fn new(config: &FakeStorageConfig, batch_plan: BatchPlan, completions: &Completions) -> Self {
        let mut state = Self {
            pending: (0..config.num_resource_groups)
                .map(|_| VecDeque::new())
                .collect(),
            arm: 0,
            num_armed_batches: 0,
            first_seen_noted: vec![false; num_jobs(config)],
        };
        state.arm_next_batch(config, batch_plan, completions);

        state
    }

    /// Takes up to `max_items` regular tasks, visiting the resource groups in rotation and
    /// resuming where the previous call left off.
    ///
    /// # Returns
    ///
    /// A tuple containing the tasks taken, which is empty once every armed task has been handed
    /// out, and the subset of them that is the first task of its job ever emitted.
    fn take_regular(
        &mut self,
        config: &FakeStorageConfig,
        max_items: usize,
    ) -> (Vec<InboundEntry>, Vec<InboundEntry>) {
        let num_groups = self.pending.len();
        if 0 == num_groups {
            return (Vec::new(), Vec::new());
        }

        let mut entries = Vec::new();
        let mut first_seen = Vec::new();
        let mut num_groups_skipped = 0;
        while entries.len() < max_items && num_groups_skipped < num_groups {
            if let Some(entry) = self.pending[self.arm].pop_front() {
                if let Some(job_index) = job_index_of(config, entry.job_id)
                    && !self.first_seen_noted[job_index]
                {
                    self.first_seen_noted[job_index] = true;
                    first_seen.push(entry);
                }
                entries.push(entry);
                num_groups_skipped = 0;
            } else {
                num_groups_skipped += 1;
            }
            self.arm = (self.arm + 1) % num_groups;
        }

        (entries, first_seen)
    }

    /// Arms the batch following the ones already armed, if the plan has one left.
    fn arm_next_batch(
        &mut self,
        config: &FakeStorageConfig,
        batch_plan: BatchPlan,
        completions: &Completions,
    ) {
        let batch_index = self.num_armed_batches;
        if batch_index >= batch_plan.num_batches {
            return;
        }

        // The tally must name the incoming batch before any of its tasks can be taken, or a task
        // completed straight out of the first poll would be counted against the outgoing one.
        completions
            .num_jobs_left_in_batch
            .store(batch_plan.num_jobs_in(batch_index), Ordering::Release);
        self.num_armed_batches = batch_index + 1;
        arm_batch(
            config,
            batch_plan,
            batch_index,
            completions,
            &mut self.pending,
        );
    }

    /// Re-arms every task of the armed batches that has not been reported complete.
    fn rearm(
        &mut self,
        config: &FakeStorageConfig,
        batch_plan: BatchPlan,
        completions: &Completions,
    ) {
        for queue in &mut self.pending {
            queue.clear();
        }
        self.arm = 0;
        for batch_index in 0..self.num_armed_batches {
            arm_batch(
                config,
                batch_plan,
                batch_index,
                completions,
                &mut self.pending,
            );
        }
    }
}

/// The armed finalization tasks of both lanes.
#[derive(Debug)]
struct FinalizeState {
    pending_commit: VecDeque<InboundEntry>,
    pending_cleanup: VecDeque<InboundEntry>,
    emitted: Vec<bool>,
}

impl FinalizeState {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A state armed with the finalization of every job that has already finished its regular
    /// tasks.
    fn new(config: &FakeStorageConfig, completions: &Completions) -> Self {
        let mut state = Self {
            pending_commit: VecDeque::new(),
            pending_cleanup: VecDeque::new(),
            emitted: vec![false; num_jobs(config)],
        };
        state.rearm(config, completions);

        state
    }

    /// Takes up to `max_items` finalization tasks from `lane`.
    ///
    /// # Returns
    ///
    /// The finalization tasks taken, which is empty while no job served by `lane` has finished its
    /// regular tasks.
    fn take(&mut self, lane: FinalizeLane, max_items: usize) -> Vec<InboundEntry> {
        let pending = match lane {
            FinalizeLane::Commit => &mut self.pending_commit,
            FinalizeLane::Cleanup => &mut self.pending_cleanup,
        };
        let num_taken = max_items.min(pending.len());

        pending.drain(..num_taken).collect()
    }

    /// Arms the finalization task of the job at `job_index` if the configuration gives that job one
    /// and the job has finished every regular task without its finalization having been armed
    /// already.
    fn arm(&mut self, config: &FakeStorageConfig, completions: &Completions, job_index: usize) {
        let Some(task_id) = finalize_task_id_of(config, job_index) else {
            return;
        };
        if self.emitted[job_index]
            || completions.is_finalization_completed(job_index)
            || completions.num_completed_regular_of(job_index) < config.num_tasks_per_job
        {
            return;
        }

        self.emitted[job_index] = true;
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

    /// Re-arms the finalization of every job that has finished its regular tasks without having
    /// reported its finalization complete.
    fn rearm(&mut self, config: &FakeStorageConfig, completions: &Completions) {
        self.pending_commit.clear();
        self.pending_cleanup.clear();
        self.emitted.fill(false);
        for job_index in 0..self.emitted.len() {
            self.arm(config, completions, job_index);
        }
    }
}

/// The tally of what the workload has reported complete, held in atomics so that a completion
/// report never blocks and never blocks a poll.
#[derive(Debug)]
struct Completions {
    num_words_per_job: usize,
    regular_bits: Box<[AtomicU64]>,
    num_completed_regular_per_job: Box<[AtomicUsize]>,
    num_completed_regular: AtomicUsize,
    finalization_completed: Box<[AtomicBool]>,
    num_completed_finalizations: AtomicUsize,
    num_jobs_left_in_batch: AtomicUsize,
}

impl Completions {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A tally sized for `config`'s workload with nothing reported complete.
    fn new(config: &FakeStorageConfig) -> Self {
        let num_jobs = num_jobs(config);
        let num_words_per_job = config.num_tasks_per_job.div_ceil(COMPLETION_BITS_PER_WORD);

        Self {
            num_words_per_job,
            regular_bits: (0..num_jobs * num_words_per_job)
                .map(|_| AtomicU64::new(0))
                .collect(),
            num_completed_regular_per_job: (0..num_jobs).map(|_| AtomicUsize::new(0)).collect(),
            num_completed_regular: AtomicUsize::new(0),
            finalization_completed: (0..num_jobs).map(|_| AtomicBool::new(false)).collect(),
            num_completed_finalizations: AtomicUsize::new(0),
            num_jobs_left_in_batch: AtomicUsize::new(0),
        }
    }

    /// Records the regular task at `task_index` of the job at `job_index` as executed.
    ///
    /// # Returns
    ///
    /// Whether this report was the one that finished the job's regular tasks, which is `false` for
    /// a task outside the workload and for a repeated report.
    fn record_regular(
        &self,
        config: &FakeStorageConfig,
        job_index: usize,
        task_index: TaskIndex,
    ) -> bool {
        if task_index >= config.num_tasks_per_job {
            return false;
        }
        let (word_index, mask) = self.bit_of(job_index, task_index);
        if 0 != self.regular_bits[word_index].fetch_or(mask, Ordering::Relaxed) & mask {
            return false;
        }
        self.num_completed_regular.fetch_add(1, Ordering::Relaxed);

        config.num_tasks_per_job
            == 1 + self.num_completed_regular_per_job[job_index].fetch_add(1, Ordering::Relaxed)
    }

    /// Records the finalization of the job at `job_index` as executed, ignoring the report for a
    /// job the configuration gives no finalization and ignoring a repeated report.
    fn record_finalization(&self, config: &FakeStorageConfig, job_index: usize) {
        if finalize_task_id_of(config, job_index).is_none()
            || self.finalization_completed[job_index].swap(true, Ordering::Relaxed)
        {
            return;
        }
        self.num_completed_finalizations
            .fetch_add(1, Ordering::Relaxed);
    }

    /// # Returns
    ///
    /// Whether the regular task at `task_index` of the job at `job_index` has been reported
    /// complete.
    fn is_regular_completed(&self, job_index: usize, task_index: TaskIndex) -> bool {
        let (word_index, mask) = self.bit_of(job_index, task_index);

        0 != self.regular_bits[word_index].load(Ordering::Relaxed) & mask
    }

    /// # Returns
    ///
    /// Whether the finalization of the job at `job_index` has been reported complete.
    fn is_finalization_completed(&self, job_index: usize) -> bool {
        self.finalization_completed[job_index].load(Ordering::Relaxed)
    }

    /// # Returns
    ///
    /// The number of regular tasks the job at `job_index` has reported complete.
    fn num_completed_regular_of(&self, job_index: usize) -> usize {
        self.num_completed_regular_per_job[job_index].load(Ordering::Relaxed)
    }

    /// # Returns
    ///
    /// A tuple containing the index of the bitmap word holding the regular task at `task_index` of
    /// the job at `job_index`, and that task's mask within the word.
    const fn bit_of(&self, job_index: usize, task_index: TaskIndex) -> (usize, u64) {
        let word_index = job_index * self.num_words_per_job + task_index / COMPLETION_BITS_PER_WORD;
        let mask = 1_u64 << (task_index % COMPLETION_BITS_PER_WORD);

        (word_index, mask)
    }
}

/// Appends every regular task of batch `batch_index` that has not been reported complete to its
/// resource group's queue.
fn arm_batch(
    config: &FakeStorageConfig,
    batch_plan: BatchPlan,
    batch_index: usize,
    completions: &Completions,
    pending: &mut [VecDeque<InboundEntry>],
) {
    let (first_job_in_group, end_job_in_group) = batch_plan.job_range(batch_index);
    for (group_index, queue) in pending.iter_mut().enumerate() {
        let resource_group_id = ResourceGroupId::from(index_to_raw_id(group_index));
        for job_index_in_group in first_job_in_group..end_job_in_group {
            let job_index = group_index * config.num_jobs_per_group + job_index_in_group;
            let job_id = job_id_of(job_index);
            queue.extend((0..config.num_tasks_per_job).filter_map(|task_index| {
                (!completions.is_regular_completed(job_index, task_index)).then_some(InboundEntry {
                    resource_group_id,
                    job_id,
                    task_id: TaskId::Index(task_index),
                })
            }));
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::time::Instant;

    use dashmap::DashMap;

    use crate::bench::JobProgressMap;
    use crate::harness::fake_storage::FakeStorage;
    use crate::harness::fake_storage::FakeStorageConfig;
    use crate::harness::fake_storage::FirstSeenRecorder;
    use crate::harness::fake_storage::ReleaseConfig;
    use crate::harness::fake_storage::index_to_raw_id;
    use crate::types::JobId;
    use crate::types::TaskId;

    /// # Returns
    ///
    /// A workload of the given dimensions emitting no finalization.
    const fn workload(
        num_resource_groups: usize,
        num_jobs_per_group: usize,
        num_tasks_per_job: usize,
    ) -> FakeStorageConfig {
        FakeStorageConfig {
            num_resource_groups,
            num_jobs_per_group,
            num_tasks_per_job,
            emit_commit_ready: false,
            emit_cleanup_ready: false,
        }
    }

    /// # Returns
    ///
    /// `value` as a non-zero size.
    ///
    /// # Panics
    ///
    /// Panics if `value` is zero.
    fn non_zero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test size is non-zero")
    }

    /// # Returns
    ///
    /// The instant the storage recorded as its first emission of a task of the job at `job_index`.
    ///
    /// # Panics
    ///
    /// Panics if the job has never been recorded as first seen.
    fn first_seen_nanos_of(progress: &JobProgressMap, job_index: usize) -> u64 {
        let job_id = JobId::from(index_to_raw_id(job_index));

        progress
            .get(&job_id)
            .expect("the job was recorded as first seen")
            .to_record(job_id)
            .first_seen_nanos
    }

    /// Reports every task the storage emitted as executed.
    fn complete_all(storage: &FakeStorage, max_items: usize) -> usize {
        let (_, entries) = storage.take_ready(max_items);
        for entry in &entries {
            storage.complete_task(entry.job_id, entry.task_id);
        }

        entries.len()
    }

    #[test]
    fn the_default_release_config_bounds_a_poll_by_max_items_alone() {
        let storage = FakeStorage::new(workload(2, 2, 8));
        let (_, entries) = storage.take_ready(usize::MAX);
        assert_eq!(entries.len(), 32);
        assert_eq!(storage.total_tasks(), 32);
    }

    #[test]
    fn a_poll_response_never_exceeds_the_inbound_wave_size() {
        let storage = FakeStorage::with_release_config(
            workload(2, 2, 8),
            ReleaseConfig {
                inbound_wave_size: Some(non_zero(6)),
                ..ReleaseConfig::default()
            },
        );

        let mut num_taken = 0;
        for _ in 0..6 {
            let (_, entries) = storage.take_ready(usize::MAX);
            assert!(entries.len() <= 6);
            num_taken += entries.len();
        }
        assert_eq!(num_taken, 32);
    }

    #[test]
    fn a_batch_is_armed_only_once_its_predecessor_has_fully_completed() {
        let storage = FakeStorage::with_release_config(
            workload(2, 4, 2),
            ReleaseConfig {
                jobs_per_batch: Some(non_zero(2)),
                ..ReleaseConfig::default()
            },
        );

        let (_, first_batch) = storage.take_ready(usize::MAX);
        assert_eq!(first_batch.len(), 8);
        let (_, before_completion) = storage.take_ready(usize::MAX);
        assert_eq!(before_completion, Vec::new());

        for entry in first_batch.iter().take(7) {
            storage.complete_task(entry.job_id, entry.task_id);
        }
        let (_, still_held_back) = storage.take_ready(usize::MAX);
        assert_eq!(still_held_back, Vec::new());

        let last = first_batch[7];
        storage.complete_task(last.job_id, last.task_id);
        let (_, second_batch) = storage.take_ready(usize::MAX);
        assert_eq!(second_batch.len(), 8);
        assert!(!storage.is_drained());

        for entry in &second_batch {
            storage.complete_task(entry.job_id, entry.task_id);
        }
        assert!(storage.is_drained());
        assert_eq!(storage.num_completed_tasks(), 16);
    }

    #[test]
    fn every_batch_of_a_batched_workload_is_eventually_armed() {
        let storage = FakeStorage::with_release_config(
            workload(3, 9, 4),
            ReleaseConfig {
                inbound_wave_size: Some(non_zero(5)),
                jobs_per_batch: Some(non_zero(2)),
                ..ReleaseConfig::default()
            },
        );

        let mut num_completed = 0;
        for _ in 0..128 {
            num_completed += complete_all(&storage, usize::MAX);
        }
        assert_eq!(num_completed, storage.total_tasks());
        assert!(storage.is_drained());
    }

    #[test]
    fn the_storage_records_when_it_first_emits_a_task_of_each_job() {
        let progress = Arc::new(DashMap::new());
        let storage = FakeStorage::with_release_config(
            workload(2, 2, 4),
            ReleaseConfig {
                inbound_wave_size: Some(non_zero(2)),
                first_seen_recorder: Some(FirstSeenRecorder::new(
                    Arc::clone(&progress),
                    Instant::now(),
                )),
                ..ReleaseConfig::default()
            },
        );

        let (_, first_wave) = storage.take_ready(usize::MAX);
        assert_eq!(first_wave.len(), 2);
        assert_eq!(progress.len(), 2);
        let first_wave_jobs = [0, 2];
        let earliest = first_wave_jobs.map(|job_index| first_seen_nanos_of(&progress, job_index));

        while !storage.take_ready(usize::MAX).1.is_empty() {}
        assert_eq!(progress.len(), 4);
        for (job_index, first_seen_nanos) in first_wave_jobs.into_iter().zip(earliest) {
            assert_eq!(first_seen_nanos_of(&progress, job_index), first_seen_nanos);
        }
    }

    #[test]
    fn a_repeated_or_unknown_completion_report_is_ignored() {
        let storage = FakeStorage::new(workload(1, 1, 2));
        storage.complete_task(JobId::from(0), TaskId::Index(0));
        storage.complete_task(JobId::from(0), TaskId::Index(0));
        storage.complete_task(JobId::from(0), TaskId::Index(7));
        storage.complete_task(JobId::from(9), TaskId::Index(1));
        assert_eq!(storage.num_completed_tasks(), 1);
        assert!(!storage.is_drained());

        storage.complete_task(JobId::from(0), TaskId::Index(1));
        assert_eq!(storage.num_completed_tasks(), 2);
        assert!(storage.is_drained());
    }

    #[test]
    fn a_session_bump_re_arms_only_the_incomplete_tasks_of_the_armed_batches() {
        let storage = FakeStorage::with_release_config(
            workload(1, 4, 2),
            ReleaseConfig {
                jobs_per_batch: Some(non_zero(2)),
                ..ReleaseConfig::default()
            },
        );

        let (_, armed) = storage.take_ready(usize::MAX);
        assert_eq!(armed.len(), 4);
        storage.complete_task(armed[0].job_id, armed[0].task_id);

        assert_eq!(storage.bump_session(), 1);
        let (session_id, replayed) = storage.take_ready(usize::MAX);
        assert_eq!(session_id, 1);
        assert_eq!(replayed.len(), 3);
    }
}
