//! The prototype scheduler core: the single-threaded, tick-based decision loop.
//!
//! Each tick collects the inbound queue's polling results, folds them into the core's job registry
//! and resource group scheduling units, refills the dispatch buffer under the admission policy, and
//! retires the jobs that ran dry. The core owns all of that state outright -- job entries in a
//! generational arena, resource group scheduling units in an append-only `Vec` -- so nothing it
//! holds across an await point is thread-bound and the loop's future is `Send`.
//!
//! Every tick is timed step by step into a [`TickSample`]. Being single-threaded, the core owns the
//! resulting log outright and hands it back when the loop stops.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tokio::select;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

use crate::bench::results::TickSample;
use crate::bench::results::TickStep;
use crate::bench::results::TickTimer;
use crate::config::CoreConfig;
use crate::dispatch_queue::GlobalDispatchQueue;
use crate::error::CoreError;
use crate::error::MakeAssignmentError;
use crate::error::StorageClientError;
use crate::job_registry::JobKey;
use crate::job_registry::JobRegistry;
use crate::job_registry::UpsertOutcome;
use crate::resource_group::ResourceGroupTable;
use crate::scheduling_unit::RgSchedulingUnit;
use crate::session::SessionManager;
use crate::storage_client::SchedulerStorageClient;
use crate::types::FinalizeKind;
use crate::types::InboundEntry;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskAssignment;
use crate::types::TaskAssignmentId;
use crate::types::TaskId;
use crate::types::TaskIndex;

/// Single-source ID issuer for creating globally unique IDs for task assignments.
#[derive(Clone, Debug, Default)]
pub struct TaskAssignmentIdIssuer {
    next_id: Arc<AtomicU64>,
}

impl TaskAssignmentIdIssuer {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created [`TaskAssignmentIdIssuer`] starting from zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// # Returns
    ///
    /// The next unused task assignment ID.
    #[must_use]
    pub fn next(&self) -> TaskAssignmentId {
        TaskAssignmentId::from(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

/// The prototype scheduler core.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client the inbound queue is polled through.
pub struct Core<StorageClientType: SchedulerStorageClient> {
    pub(crate) global_task_set: HashSet<(JobId, TaskId)>,
    pub(crate) finalized_jobs: HashSet<JobId>,
    pub(crate) job_registry: JobRegistry,

    /// The scheduling units of every resource group the core has seen this session.
    ///
    /// Append-only within a session: a group is never removed individually, so a position in this
    /// vector is stable until [`Self::apply_session_bump`] flushes the whole of it.
    pub(crate) rg_units: Vec<RgSchedulingUnit>,

    pub(crate) rg_index: HashMap<ResourceGroupId, usize>,
    pub(crate) active_rg_list: Vec<usize>,
    pub(crate) last_served_rg: Option<ResourceGroupId>,

    config: CoreConfig,
    rg_table: ResourceGroupTable,
    global_queue: GlobalDispatchQueue,
    session_manager: SessionManager,
    id_issuer: TaskAssignmentIdIssuer,
    inbound_queue_reader: AsyncInboundQueueReader<StorageClientType>,
    reschedule_queue_reader: UnboundedReceiver<TaskAssignment>,
    cancellation_token: CancellationToken,
    tick_samples: Vec<TickSample>,
}

impl<StorageClientType: SchedulerStorageClient> Core<StorageClientType> {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created core with no buffered task and no in-flight inbound poll.
    #[must_use]
    pub fn new(
        config: CoreConfig,
        storage_client: StorageClientType,
        rg_table: ResourceGroupTable,
        global_queue: GlobalDispatchQueue,
        session_manager: SessionManager,
        reschedule_queue_reader: UnboundedReceiver<TaskAssignment>,
        cancellation_token: CancellationToken,
    ) -> Self {
        Self {
            global_task_set: HashSet::new(),
            finalized_jobs: HashSet::new(),
            job_registry: JobRegistry::new(),
            rg_units: Vec::new(),
            rg_index: HashMap::new(),
            active_rg_list: Vec::new(),
            last_served_rg: None,
            config,
            rg_table,
            global_queue,
            session_manager,
            id_issuer: TaskAssignmentIdIssuer::new(),
            inbound_queue_reader: AsyncInboundQueueReader::new(storage_client),
            reschedule_queue_reader,
            cancellation_token,
            tick_samples: Vec::new(),
        }
    }

    /// Reserves room for `num_ticks` tick samples.
    ///
    /// A benchmark run knows how many ticks to expect from its duration and tick interval, and
    /// reserving up front keeps the sample log from growing while the loop is being measured.
    ///
    /// # Returns
    ///
    /// The core, with its sample log able to hold `num_ticks` samples without growing.
    #[must_use]
    pub fn with_tick_sample_capacity(mut self, num_ticks: usize) -> Self {
        self.tick_samples.reserve(num_ticks);

        self
    }

    /// # Returns
    ///
    /// The samples of every tick executed so far, in tick order.
    #[must_use]
    pub fn tick_samples(&self) -> &[TickSample] {
        &self.tick_samples
    }

    /// Runs the scheduling loop until the cancellation token is triggered.
    ///
    /// Each iteration executes one [`Self::tick`] and then sleeps for the remainder of the
    /// configured tick interval.
    ///
    /// # Returns
    ///
    /// The samples of every tick the loop executed, in tick order, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`Self::tick`]'s return values on failure.
    pub async fn run(mut self) -> Result<Vec<TickSample>, CoreError> {
        tracing::info!(
            config = ? self.config,
            init_session_id = self.session_manager.current(),
            "Prototype scheduler core started."
        );
        let tick_interval = Duration::from_millis(self.config.tick_interval_ms.get());
        loop {
            let now = tokio::time::Instant::now();
            let cancellation_token = self.cancellation_token.clone();
            select! {
                () = cancellation_token.cancelled() => {
                    tracing::info!(
                        num_ticks = self.tick_samples.len(),
                        "Prototype scheduler core cancelled. Shutting down."
                    );
                    return Ok(self.tick_samples);
                }
                result = self.tick() => {
                    result.inspect_err(|e| tracing::error!(
                        err = % e,
                        "Prototype scheduler core exits on error."
                    ))?;
                }
            }
            let sleep_time = tick_interval.saturating_sub(now.elapsed());
            if sleep_time.is_zero() {
                tokio::task::yield_now().await;
            } else {
                tokio::time::sleep(sleep_time).await;
            }
        }
    }

    /// Executes one tick of the scheduling loop, appending its [`TickSample`] to the sample log.
    ///
    /// Processing the polling results is skipped while a storage poll is still in flight, but the
    /// dispatch buffer is refilled from already-buffered tasks on every tick regardless. A skipped
    /// step is reported as zero.
    ///
    /// The measurements come with two caveats a reader needs:
    ///
    /// * The collect step is a non-blocking [`AsyncInboundQueueReader::try_collect_result`], so its
    ///   cost is the bookkeeping of taking a finished poll's results rather than time spent waiting
    ///   on storage.
    /// * Starting the next poll is billed to the apply step rather than to the collect step it
    ///   belongs to conceptually, because the poll's per-lane fetch counts are derived from the
    ///   buffer occupancy the earlier steps produce and it must therefore run after them.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`AsyncInboundQueueReader::try_collect_result`]'s return values on failure.
    /// * Forwards [`Self::start_inbound_poll`]'s return values on failure.
    /// * Forwards [`Self::fill_dispatch_queues`]'s return values on failure.
    pub(crate) async fn tick(&mut self) -> Result<(), CoreError> {
        let mut timer = TickTimer::start();
        let poll_state = self
            .inbound_queue_reader
            .try_collect_result(self.session_manager.current())
            .await?;
        match poll_state {
            InboundPollState::Ready {
                session_id,
                ready_entries,
                commit_ready_entries,
                cleanup_ready_entries,
            } => {
                if session_id != self.session_manager.current() {
                    self.apply_session_bump(session_id);
                }
                let rescheduled_entries = self.drain_reschedule_queue(session_id);
                timer.finish_step(TickStep::Collect);

                let rg_updates = self.process_polling_results(
                    commit_ready_entries,
                    cleanup_ready_entries,
                    ready_entries,
                    rescheduled_entries,
                );
                timer.finish_step(TickStep::Process);

                self.apply_rg_updates(rg_updates);
                self.start_inbound_poll()?;
                timer.finish_step(TickStep::Apply);
            }
            InboundPollState::NotStarted => {
                self.start_inbound_poll()?;
                timer.finish_step(TickStep::Collect);
            }
            InboundPollState::Pending => timer.finish_step(TickStep::Collect),
        }

        let (jobs_to_retire, assignments_published) = self.fill_dispatch_queues()?;
        timer.finish_step(TickStep::Fill);

        self.retire_jobs(jobs_to_retire);
        timer.finish_step(TickStep::Retire);

        let active_resource_groups = u64::try_from(self.active_rg_list.len()).unwrap_or(u64::MAX);
        self.tick_samples
            .push(timer.finish(assignments_published, active_resource_groups));
        Ok(())
    }

    /// Discards every piece of state published in a session older than `new_session_id`.
    ///
    /// Clearing the dedup set is load-bearing rather than tidy: storage replays its ready tasks
    /// after a bump, and a stale dedup entry would drop a replayed task while the registry no
    /// longer holds anything to schedule it from.
    ///
    /// [`Self::rg_units`], [`Self::rg_index`], and [`Self::active_rg_list`] must be cleared as one
    /// operation, and this is the only place any of them is cleared. Positions in `rg_units` carry
    /// no generation, so an index that outlives the flush does not fail: it resolves against the
    /// new session's units, either out of bounds or -- once the new session has re-created a few
    /// groups -- silently against the wrong group. Nothing in the type system checks this.
    ///
    /// [`Self::rg_table`] must be cleared together with them, and that too is a correctness
    /// requirement rather than tidiness. A group's queue closes only once every sender has been
    /// dropped, and a scheduling unit's write side is one of them; a unit that survived the table's
    /// flush would therefore hold a write side onto a queue whose readers are gone, and publishing
    /// into a closed queue is fatal to the core.
    fn apply_session_bump(&mut self, new_session_id: SessionId) {
        tracing::info!(
            new_session_id,
            num_resource_groups = self.rg_table.len(),
            num_jobs = self.job_registry.len(),
            num_outstanding_hints = self.global_queue.len(),
            "Storage session bumped. Flushing the core."
        );

        self.session_manager.bump(new_session_id);
        self.rg_table.clear();
        self.rg_units.clear();
        self.rg_index.clear();
        self.active_rg_list.clear();
        self.last_served_rg = None;
        self.job_registry.clear();
        self.global_task_set.clear();
        self.finalized_jobs.clear();
        self.global_queue.drain();
    }

    /// Drains the reschedule queue, dropping assignments published in a session other than
    /// `session_id`.
    ///
    /// # Returns
    ///
    /// The rescheduled assignments, in the same form as the entries drained from the inbound queue.
    fn drain_reschedule_queue(&mut self, session_id: SessionId) -> Vec<InboundEntry> {
        let mut entries = Vec::new();
        while let Ok(assignment) = self.reschedule_queue_reader.try_recv() {
            if assignment.session_id != session_id {
                continue;
            }
            entries.push(InboundEntry {
                resource_group_id: assignment.resource_group_id,
                job_id: assignment.job_id,
                task_id: assignment.task_id,
            });
        }
        entries
    }

    /// Folds the tick's ready tasks into the finalized job table, the global task set, and the job
    /// registry.
    ///
    /// Finalizations are processed before regular tasks, so a regular task arriving in the same
    /// batch as its job's finalization is discarded rather than scheduled.
    ///
    /// # Returns
    ///
    /// The per-resource-group updates the tick produced.
    fn process_polling_results(
        &mut self,
        commit_ready_entries: Vec<InboundEntry>,
        cleanup_ready_entries: Vec<InboundEntry>,
        ready_entries: Vec<InboundEntry>,
        rescheduled_entries: Vec<InboundEntry>,
    ) -> HashMap<ResourceGroupId, RgUpdate> {
        let mut commit_ready = commit_ready_entries;
        let mut cleanup_ready = cleanup_ready_entries;
        let mut regular = ready_entries;
        for entry in rescheduled_entries {
            match entry.task_id {
                TaskId::Commit => commit_ready.push(entry),
                TaskId::Cleanup => cleanup_ready.push(entry),
                TaskId::Index(_) => regular.push(entry),
            }
        }

        let mut rg_updates: HashMap<ResourceGroupId, RgUpdate> = HashMap::new();
        for (entries, kind) in [
            (commit_ready, FinalizeKind::Commit),
            (cleanup_ready, FinalizeKind::Cleanup),
        ] {
            for entry in entries {
                if !self.finalized_jobs.insert(entry.job_id) {
                    continue;
                }
                // The job's still-buffered regular tasks will never be published, so they must
                // leave the dedup set with it or nothing would ever remove them.
                if let Some(mut job_entry) = self.job_registry.remove_by_job_id(entry.job_id) {
                    for task_index in job_entry.take_ready_tasks() {
                        self.global_task_set
                            .remove(&(entry.job_id, TaskId::Index(task_index)));
                    }
                }
                rg_updates
                    .entry(entry.resource_group_id)
                    .or_default()
                    .finalized
                    .push((entry.job_id, kind));
            }
        }

        let mut batches: HashMap<JobId, (ResourceGroupId, Vec<TaskIndex>)> = HashMap::new();
        for entry in regular {
            let TaskId::Index(task_index) = entry.task_id else {
                continue;
            };
            batches
                .entry(entry.job_id)
                .or_insert_with(|| (entry.resource_group_id, Vec::new()))
                .1
                .push(task_index);
        }
        for (job_id, (rg_id, mut task_indices)) in batches {
            if self.finalized_jobs.contains(&job_id) {
                continue;
            }
            task_indices.sort_unstable();
            task_indices.dedup();
            let global_task_set = &mut self.global_task_set;
            task_indices
                .retain(|task_index| global_task_set.insert((job_id, TaskId::Index(*task_index))));
            if task_indices.is_empty() {
                continue;
            }
            if let UpsertOutcome::New(job_key) = self.job_registry.upsert(job_id, task_indices) {
                rg_updates.entry(rg_id).or_default().new_jobs.push(job_key);
            }
        }

        rg_updates
    }

    /// Applies the tick's per-resource-group updates to the scheduling units, activating every
    /// group the updates touch.
    fn apply_rg_updates(&mut self, rg_updates: HashMap<ResourceGroupId, RgUpdate>) {
        for (rg_id, update) in rg_updates {
            let unit_index = self.get_or_create_unit(rg_id);
            let unit = &mut self.rg_units[unit_index];
            for (job_id, kind) in update.finalized {
                unit.push_finalization(job_id, kind);
            }
            for job_key in update.new_jobs {
                unit.place_new_job(job_key);
            }
            let activated = !unit.is_active;
            unit.is_active = true;
            if activated {
                self.active_rg_list.push(unit_index);
            }
        }
    }

    /// # Returns
    ///
    /// The position of `rg_id`'s scheduling unit in [`Self::rg_units`], appending a unit built
    /// against the group's dispatch queue endpoints if the core has none.
    fn get_or_create_unit(&mut self, rg_id: ResourceGroupId) -> usize {
        if let Some(unit_index) = self.rg_index.get(&rg_id) {
            return *unit_index;
        }

        let endpoints = self
            .rg_table
            .get_or_create(rg_id, self.session_manager.current());
        let unit_index = self.rg_units.len();
        self.rg_units.push(RgSchedulingUnit::new(
            rg_id,
            &endpoints,
            self.config.active_job_list_capacity.get(),
        ));
        self.rg_index.insert(rg_id, unit_index);
        unit_index
    }

    /// Publishes assignments into the per-resource-group dispatch queues under the admission
    /// policy, then deactivates the groups that ran out of work.
    ///
    /// # Returns
    ///
    /// A tuple on success, containing:
    ///
    /// * The jobs that exhausted their downgrade budget and must be retired.
    /// * The number of assignments published.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`CoreError::FatalPublication`] if a group's dispatch queue or the global dispatch queue
    ///   is closed, in which case the assignments the core makes can no longer reach an execution
    ///   manager.
    fn fill_dispatch_queues(&mut self) -> Result<(Vec<JobKey>, u64), CoreError> {
        let mut jobs_to_retire = Vec::new();
        let mut assignments_published = 0;
        if self.active_rg_list.is_empty() {
            return Ok((jobs_to_retire, assignments_published));
        }

        // The decision loop needs the scheduling units and the job arena borrowed mutably at the
        // same time, which the borrow checker accepts only for bindings that name distinct fields.
        let Self {
            global_task_set,
            job_registry,
            rg_units,
            active_rg_list,
            last_served_rg,
            config,
            global_queue,
            session_manager,
            id_issuer,
            ..
        } = self;

        let mut rg_rr_list = Vec::with_capacity(active_rg_list.len());
        let mut occupancy = 0;
        let mut last_served_index = None;
        for (index, unit_index) in active_rg_list.iter().enumerate() {
            let unit = &rg_units[*unit_index];
            occupancy += unit.dispatch_queue_size();
            if Some(unit.rg_id) == *last_served_rg {
                last_served_index = Some(index);
            }
            rg_rr_list.push(*unit_index);
        }
        // Bounding by the free space measured here is what makes the loop terminate: the queues
        // drain concurrently, so the true free space only ever grows.
        let mut free = config
            .dispatch_queue_capacity
            .get()
            .saturating_sub(occupancy);
        // Rotating the arm rather than the list keeps the same group from always being visited
        // first, which matters because `free` shrinks as the tick proceeds.
        let mut arm = last_served_index.map_or(0, |index| (index + 1) % rg_rr_list.len());

        for unit_index in &rg_rr_list {
            rg_units[*unit_index].promote_pending_jobs(job_registry, &mut jobs_to_retire);
        }

        let session_id = session_manager.current();
        let mut exhausted_units = Vec::new();
        while 0 != free && !rg_rr_list.is_empty() {
            let unit_index = rg_rr_list[arm];
            let unit = &mut rg_units[unit_index];
            let result = unit.try_make_assignment(
                free,
                session_id,
                id_issuer,
                global_queue,
                job_registry,
                &mut jobs_to_retire,
            );
            match result {
                Ok((job_id, task_id)) => {
                    global_task_set.remove(&(job_id, task_id));
                    free -= 1;
                    assignments_published += 1;
                    *last_served_rg = Some(unit.rg_id);
                    arm = (arm + 1) % rg_rr_list.len();
                }
                Err(err) => {
                    match err {
                        MakeAssignmentError::NoTask => exhausted_units.push(unit_index),
                        MakeAssignmentError::DispatchQueueFull => (),
                        MakeAssignmentError::DispatchQueueClosed
                        | MakeAssignmentError::BroadcastQueueClosed => {
                            return Err(CoreError::FatalPublication(err));
                        }
                    }
                    rg_rr_list.swap_remove(arm);
                    // `swap_remove` moved the tail element into this slot, so advancing the arm
                    // here would skip it.
                    if arm == rg_rr_list.len() {
                        arm = 0;
                    }
                }
            }
        }

        for unit_index in active_rg_list.iter() {
            rg_units[*unit_index].apply_downgrades(job_registry);
        }
        self.deactivate_exhausted_units(exhausted_units);

        Ok((jobs_to_retire, assignments_published))
    }

    /// Takes every exhausted group that also holds no assignment off the active resource group
    /// list.
    ///
    /// The empty-queue condition is required for correctness: free space is summed over the active
    /// list alone, so a deactivated group still holding assignments would hide its occupancy and
    /// let the core over-admit.
    fn deactivate_exhausted_units(&mut self, exhausted_units: Vec<usize>) {
        for unit_index in exhausted_units {
            let unit = &mut self.rg_units[unit_index];
            if unit.has_schedulable_task() || 0 != unit.dispatch_queue_size() {
                continue;
            }
            unit.is_active = false;
            if let Some(position) = self
                .active_rg_list
                .iter()
                .position(|active_index| *active_index == unit_index)
            {
                self.active_rg_list.swap_remove(position);
            }
        }
    }

    /// Drops the registry's entry for every job that ran out of downgrade lives.
    fn retire_jobs(&mut self, jobs_to_retire: Vec<JobKey>) {
        for job_key in jobs_to_retire {
            self.job_registry.remove(job_key);
        }
    }

    /// Starts the next inbound poll, sizing each lane's fetch count by the buffer capacity that
    /// lane still has left.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`AsyncInboundQueueReader::start`]'s return values on failure.
    fn start_inbound_poll(&mut self) -> Result<(), CoreError> {
        let (num_commit_ready, num_cleanup_ready) = self.count_buffered_finalizations();
        let max_ready_entries = self
            .config
            .ready_task_capacity
            .get()
            .saturating_sub(self.global_task_set.len());
        let max_commit_ready_entries = self
            .config
            .commit_ready_task_capacity
            .get()
            .saturating_sub(num_commit_ready);
        let max_cleanup_ready_entries = self
            .config
            .cleanup_ready_task_capacity
            .get()
            .saturating_sub(num_cleanup_ready);

        self.inbound_queue_reader.start(
            Duration::from_millis(self.config.storage_poll_timeout_ms),
            max_ready_entries,
            max_commit_ready_entries,
            max_cleanup_ready_entries,
        )
    }

    /// # Returns
    ///
    /// A tuple containing:
    ///
    /// * The number of buffered commit tasks.
    /// * The number of buffered cleanup tasks.
    fn count_buffered_finalizations(&self) -> (usize, usize) {
        let mut num_commit_ready = 0;
        let mut num_cleanup_ready = 0;
        for unit in &self.rg_units {
            let (num_commits, num_cleanups) = unit.num_buffered_finalizations();
            num_commit_ready += num_commits;
            num_cleanup_ready += num_cleanups;
        }
        (num_commit_ready, num_cleanup_ready)
    }
}

/// Runs `core` on a dedicated OS thread under a current-thread runtime, discarding its tick
/// samples.
///
/// The thread is a measurement choice rather than a requirement: it keeps the loop's cadence and
/// per-step timings clear of whatever else shares the caller's runtime. A caller that does not care
/// about either may equally spawn [`Core::run`] as an ordinary task.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client the core polls the inbound queue through.
///
/// # Returns
///
/// The join handle of the spawned thread, resolving to the core's exit status.
pub fn run_core_on_dedicated_thread<StorageClientType: SchedulerStorageClient>(
    core: Core<StorageClientType>,
) -> std::thread::JoinHandle<Result<(), CoreError>> {
    std::thread::spawn(move || run_core_to_completion(core).map(|_| ()))
}

/// Runs `core` exactly as [`run_core_on_dedicated_thread`] does, keeping its tick samples.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client the core polls the inbound queue through.
///
/// # Returns
///
/// The join handle of the spawned thread, resolving to the core's tick samples or to its exit
/// status on failure.
pub fn run_core_on_dedicated_thread_with_tick_samples<StorageClientType: SchedulerStorageClient>(
    core: Core<StorageClientType>,
) -> std::thread::JoinHandle<Result<Vec<TickSample>, CoreError>> {
    std::thread::spawn(move || run_core_to_completion(core))
}

/// Runs `core` to completion on the calling thread under a current-thread runtime of its own.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client the core polls the inbound queue through.
///
/// # Returns
///
/// Forwards [`Core::run`]'s return value on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`CoreError::Internal`] if the runtime could not be built.
/// * Forwards [`Core::run`]'s return values on failure.
fn run_core_to_completion<StorageClientType: SchedulerStorageClient>(
    core: Core<StorageClientType>,
) -> Result<Vec<TickSample>, CoreError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| CoreError::Internal(e.to_string()))?;

    runtime.block_on(core.run())
}

/// The updates one tick produced for a single resource group.
#[derive(Default)]
struct RgUpdate {
    finalized: Vec<(JobId, FinalizeKind)>,
    new_jobs: Vec<JobKey>,
}

/// The state of an asynchronous inbound-queue poll.
enum InboundPollState {
    /// The poll has completed, carrying the polled session and the entries drained from each
    /// inbound-queue lane.
    Ready {
        session_id: SessionId,
        ready_entries: Vec<InboundEntry>,
        commit_ready_entries: Vec<InboundEntry>,
        cleanup_ready_entries: Vec<InboundEntry>,
    },

    /// The poll is still in flight.
    Pending,

    /// No poll has been started.
    NotStarted,
}

/// A reader that runs inbound-queue polls as background tasks, with at most one polling request
/// (from all three lanes) in flight at a time.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client used to poll the inbound queue.
struct AsyncInboundQueueReader<StorageClientType: SchedulerStorageClient> {
    storage_client: StorageClientType,
    handles: Option<InboundPollHandles>,
}

impl<StorageClientType: SchedulerStorageClient> AsyncInboundQueueReader<StorageClientType> {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created reader with no poll in flight.
    const fn new(storage_client: StorageClientType) -> Self {
        Self {
            storage_client,
            handles: None,
        }
    }

    /// Tries to collect the result of the in-flight poll without blocking, releasing the poll
    /// handles once a result is produced.
    ///
    /// # Returns
    ///
    /// On success:
    ///
    /// * [`InboundPollState::NotStarted`] if no poll is in flight.
    /// * Forwards [`InboundPollHandles::try_collect_result`]'s return values otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`InboundPollHandles::try_collect_result`]'s return values on failure.
    async fn try_collect_result(
        &mut self,
        curr_session_id: SessionId,
    ) -> Result<InboundPollState, CoreError> {
        let Some(handles) = &mut self.handles else {
            return Ok(InboundPollState::NotStarted);
        };
        let poll_state = handles.try_collect_result(curr_session_id).await?;
        if !matches!(poll_state, InboundPollState::Pending) {
            self.handles = None;
        }
        Ok(poll_state)
    }

    /// Starts a new inbound poll, polling each inbound-queue lane as a background task.
    ///
    /// Lanes whose entry limit is 0 are not polled; if all limits are 0, no poll is started.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`CoreError::Internal`] if a poll is already in flight.
    fn start(
        &mut self,
        storage_poll_timeout: Duration,
        max_ready_entries: usize,
        max_commit_ready_entries: usize,
        max_cleanup_ready_entries: usize,
    ) -> Result<(), CoreError> {
        if self.handles.is_some() {
            return Err(CoreError::Internal(
                "inbound poll handle already exists".to_owned(),
            ));
        }
        if 0 == max_ready_entries && 0 == max_commit_ready_entries && 0 == max_cleanup_ready_entries
        {
            return Ok(());
        }

        let ready_storage_client = self.storage_client.clone();
        let ready_handle = tokio::task::spawn(async move {
            if 0 == max_ready_entries {
                return Ok((0, Vec::new()));
            }
            ready_storage_client
                .poll_ready(max_ready_entries, storage_poll_timeout)
                .await
        });

        let commit_ready_storage_client = self.storage_client.clone();
        let commit_ready_handle = tokio::task::spawn(async move {
            if 0 == max_commit_ready_entries {
                return Ok((0, Vec::new()));
            }
            commit_ready_storage_client
                .poll_commit_ready(max_commit_ready_entries, storage_poll_timeout)
                .await
        });

        let cleanup_ready_storage_client = self.storage_client.clone();
        let cleanup_ready_handle = tokio::task::spawn(async move {
            if 0 == max_cleanup_ready_entries {
                return Ok((0, Vec::new()));
            }
            cleanup_ready_storage_client
                .poll_cleanup_ready(max_cleanup_ready_entries, storage_poll_timeout)
                .await
        });

        self.handles = Some(InboundPollHandles {
            regular: ready_handle,
            commit: commit_ready_handle,
            cleanup: cleanup_ready_handle,
        });
        Ok(())
    }
}

/// The join handles of one in-flight inbound poll, one per inbound-queue lane.
struct InboundPollHandles {
    regular: tokio::task::JoinHandle<Result<(SessionId, Vec<InboundEntry>), StorageClientError>>,
    commit: tokio::task::JoinHandle<Result<(SessionId, Vec<InboundEntry>), StorageClientError>>,
    cleanup: tokio::task::JoinHandle<Result<(SessionId, Vec<InboundEntry>), StorageClientError>>,
}

impl InboundPollHandles {
    /// Tries to collect the results of all lane polls without blocking.
    ///
    /// Entries from lanes that report an older session than the latest observed session are
    /// dropped.
    ///
    /// # Returns
    ///
    /// On success:
    ///
    /// * [`InboundPollState::Pending`] if any lane poll is still in flight.
    /// * [`InboundPollState::Ready`] with the latest observed session and its entries otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`CoreError::Internal`] if any lane's polling task fails to join.
    /// * Forwards [`SchedulerStorageClient::poll_ready`]'s return values on failure.
    /// * Forwards [`SchedulerStorageClient::poll_commit_ready`]'s return values on failure.
    /// * Forwards [`SchedulerStorageClient::poll_cleanup_ready`]'s return values on failure.
    async fn try_collect_result(
        &mut self,
        curr_session_id: SessionId,
    ) -> Result<InboundPollState, CoreError> {
        if !self.regular.is_finished() || !self.commit.is_finished() || !self.cleanup.is_finished()
        {
            return Ok(InboundPollState::Pending);
        }

        let (ready_session_id, ready_entries) = (&mut self.regular)
            .await
            .map_err(|e| CoreError::Internal(e.to_string()))??;
        let (commit_session_id, commit_ready_entries) = (&mut self.commit)
            .await
            .map_err(|e| CoreError::Internal(e.to_string()))??;
        let (cleanup_session_id, cleanup_ready_entries) = (&mut self.cleanup)
            .await
            .map_err(|e| CoreError::Internal(e.to_string()))??;

        let latest_session_id = curr_session_id
            .max(ready_session_id)
            .max(commit_session_id)
            .max(cleanup_session_id);

        Ok(InboundPollState::Ready {
            session_id: latest_session_id,
            ready_entries: Self::drop_if_stale(ready_session_id, latest_session_id, ready_entries),
            commit_ready_entries: Self::drop_if_stale(
                commit_session_id,
                latest_session_id,
                commit_ready_entries,
            ),
            cleanup_ready_entries: Self::drop_if_stale(
                cleanup_session_id,
                latest_session_id,
                cleanup_ready_entries,
            ),
        })
    }

    /// # Returns
    ///
    /// `entries` if `session_id` matches `latest_session_id`, or an empty vector otherwise.
    fn drop_if_stale(
        session_id: SessionId,
        latest_session_id: SessionId,
        entries: Vec<InboundEntry>,
    ) -> Vec<InboundEntry> {
        if session_id == latest_session_id {
            entries
        } else {
            Vec::new()
        }
    }
}
