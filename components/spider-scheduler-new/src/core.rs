//! The prototype scheduler core: the single-threaded, tick-based decision loop.
//!
//! Each tick collects the inbound queue's polling results, folds them into the core's job registry
//! and resource group scheduling units, refills the dispatch buffer under the admission policy, and
//! retires the jobs that ran dry. The loop holds `Rc`-shared scheduling state across await points
//! and is therefore `!Send`: it runs on a dedicated thread under a `LocalSet`, while the inbound
//! polls it issues are ordinary `Send` background tasks.
//!
//! Every tick is timed step by step into a [`TickSample`]. Being single-threaded, the core owns the
//! resulting log outright and hands it back when the loop stops.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;
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
use crate::job_registry::JobRegistry;
use crate::job_registry::SharedJobEntry;
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
    pub(crate) rg_units: HashMap<ResourceGroupId, Rc<RefCell<RgSchedulingUnit>>>,
    pub(crate) active_rg_list: Vec<Rc<RefCell<RgSchedulingUnit>>>,
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
            rg_units: HashMap::new(),
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
    // The core co-owns its scheduling state through `Rc`, so its future is `!Send` by
    // construction. It runs on a dedicated thread under a `LocalSet`, never on a shared executor.
    #[allow(clippy::future_not_send)]
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
    // `!Send` for the same reason as [`Self::run`]: the core co-owns `Rc` scheduling state across
    // the await points below, and runs under a `LocalSet` rather than a shared executor.
    #[allow(clippy::future_not_send)]
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

        let (jobs_to_retire, assignments_published) = self.fill_dispatch_queues();
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
                if let Some(job_entry) = self.job_registry.finalize_and_remove(entry.job_id) {
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
            if let UpsertOutcome::New(entry) = self.job_registry.upsert(job_id, rg_id, task_indices)
            {
                rg_updates
                    .entry(entry.rg_id())
                    .or_default()
                    .new_jobs
                    .push(entry);
            }
        }

        rg_updates
    }

    /// Applies the tick's per-resource-group updates to the scheduling units, activating every
    /// group the updates touch.
    fn apply_rg_updates(&mut self, rg_updates: HashMap<ResourceGroupId, RgUpdate>) {
        for (rg_id, update) in rg_updates {
            let unit = self.get_or_create_unit(rg_id);
            let activated = {
                let mut unit_ref = unit.borrow_mut();
                for (job_id, kind) in update.finalized {
                    unit_ref.push_finalization(job_id, kind);
                }
                for entry in update.new_jobs {
                    unit_ref.place_new_job(entry);
                }
                let activated = !unit_ref.is_active;
                unit_ref.is_active = true;
                activated
            };
            if activated {
                self.active_rg_list.push(unit);
            }
        }
    }

    /// # Returns
    ///
    /// The scheduling unit of `rg_id`, created against the group's dispatch queue endpoints if the
    /// core has none.
    fn get_or_create_unit(&mut self, rg_id: ResourceGroupId) -> Rc<RefCell<RgSchedulingUnit>> {
        if let Some(unit) = self.rg_units.get(&rg_id) {
            return Rc::clone(unit);
        }

        let endpoints = self
            .rg_table
            .get_or_create(rg_id, self.session_manager.current());
        let unit = Rc::new(RefCell::new(RgSchedulingUnit::new(
            rg_id,
            endpoints,
            self.config.active_job_list_capacity.get(),
        )));
        self.rg_units.insert(rg_id, Rc::clone(&unit));
        unit
    }

    /// Publishes assignments into the per-resource-group dispatch queues under the admission
    /// policy, then deactivates the groups that ran out of work.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///
    /// * The jobs that exhausted their downgrade budget and must be retired.
    /// * The number of assignments published.
    fn fill_dispatch_queues(&mut self) -> (Vec<JobId>, u64) {
        let mut jobs_to_retire = Vec::new();
        let mut assignments_published = 0;
        if self.active_rg_list.is_empty() {
            return (jobs_to_retire, assignments_published);
        }

        let mut rg_rr_list = Vec::with_capacity(self.active_rg_list.len());
        let mut occupancy = 0;
        let mut last_served_index = None;
        for (index, unit) in self.active_rg_list.iter().enumerate() {
            {
                let unit_ref = unit.borrow();
                occupancy += unit_ref.dispatch_queue_size();
                if Some(unit_ref.rg_id) == self.last_served_rg {
                    last_served_index = Some(index);
                }
            }
            rg_rr_list.push(Rc::clone(unit));
        }
        // Bounding by the free space measured here is what makes the loop terminate: the queues
        // drain concurrently, so the true free space only ever grows.
        let mut free = self
            .config
            .dispatch_queue_capacity
            .get()
            .saturating_sub(occupancy);
        // Rotating the arm rather than the list keeps the same group from always being visited
        // first, which matters because `free` shrinks as the tick proceeds.
        let mut arm = last_served_index.map_or(0, |index| (index + 1) % rg_rr_list.len());

        for unit in &rg_rr_list {
            unit.borrow_mut().promote_pending_jobs(&mut jobs_to_retire);
        }

        let session_id = self.session_manager.current();
        let mut exhausted_units = Vec::new();
        while 0 != free && !rg_rr_list.is_empty() {
            let unit = Rc::clone(&rg_rr_list[arm]);
            let result = unit.borrow_mut().try_make_assignment(
                free,
                session_id,
                &self.id_issuer,
                &self.global_queue,
                &mut jobs_to_retire,
            );
            match result {
                Ok((job_id, task_id)) => {
                    self.global_task_set.remove(&(job_id, task_id));
                    free -= 1;
                    assignments_published += 1;
                    self.last_served_rg = Some(unit.borrow().rg_id);
                    arm = (arm + 1) % rg_rr_list.len();
                }
                Err(err) => {
                    if MakeAssignmentError::NoTask == err {
                        exhausted_units.push(unit);
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

        for unit in &self.active_rg_list {
            unit.borrow_mut().apply_downgrades();
        }
        self.deactivate_exhausted_units(exhausted_units);

        (jobs_to_retire, assignments_published)
    }

    /// Takes every exhausted group that also holds no assignment off the active resource group
    /// list.
    ///
    /// The empty-queue condition is required for correctness: free space is summed over the active
    /// list alone, so a deactivated group still holding assignments would hide its occupancy and
    /// let the core over-admit.
    fn deactivate_exhausted_units(&mut self, exhausted_units: Vec<Rc<RefCell<RgSchedulingUnit>>>) {
        for unit in exhausted_units {
            {
                let mut unit_ref = unit.borrow_mut();
                if unit_ref.has_schedulable_task() || 0 != unit_ref.dispatch_queue_size() {
                    continue;
                }
                unit_ref.is_active = false;
            }
            if let Some(index) = self
                .active_rg_list
                .iter()
                .position(|active_unit| Rc::ptr_eq(active_unit, &unit))
            {
                self.active_rg_list.swap_remove(index);
            }
        }
    }

    /// Drops the core's co-ownership of every job that ran out of downgrade lives.
    fn retire_jobs(&mut self, jobs_to_retire: Vec<JobId>) {
        for job_id in jobs_to_retire {
            self.job_registry.remove(job_id);
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
        for unit in self.rg_units.values() {
            let (num_commits, num_cleanups) = unit.borrow().num_buffered_finalizations();
            num_commit_ready += num_commits;
            num_cleanup_ready += num_cleanups;
        }
        (num_commit_ready, num_cleanup_ready)
    }
}

/// Runs the core built by `core_factory` on a dedicated OS thread, under a current-thread runtime
/// and a `LocalSet`, discarding the core's tick samples.
///
/// The core is built on the spawned thread rather than handed to it: it co-owns its scheduling
/// state through `Rc` and is therefore `!Send`.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client the core polls the inbound queue through.
/// * `CoreFactoryType` - The factory that builds the core on the spawned thread.
///
/// # Returns
///
/// The join handle of the spawned thread, resolving to the core's exit status.
pub fn run_core_on_dedicated_thread<
    StorageClientType: SchedulerStorageClient,
    CoreFactoryType: FnOnce() -> Core<StorageClientType> + Send + 'static,
>(
    core_factory: CoreFactoryType,
) -> std::thread::JoinHandle<Result<(), CoreError>> {
    std::thread::spawn(move || run_core_locally(core_factory).map(|_| ()))
}

/// Runs the core built by `core_factory` exactly as [`run_core_on_dedicated_thread`] does, keeping
/// its tick samples.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client the core polls the inbound queue through.
/// * `CoreFactoryType` - The factory that builds the core on the spawned thread.
///
/// # Returns
///
/// The join handle of the spawned thread, resolving to the core's tick samples or to its exit
/// status on failure.
pub fn run_core_on_dedicated_thread_with_tick_samples<
    StorageClientType: SchedulerStorageClient,
    CoreFactoryType: FnOnce() -> Core<StorageClientType> + Send + 'static,
>(
    core_factory: CoreFactoryType,
) -> std::thread::JoinHandle<Result<Vec<TickSample>, CoreError>> {
    std::thread::spawn(move || run_core_locally(core_factory))
}

/// Builds a core with `core_factory` on the calling thread and runs it to completion under a
/// current-thread runtime and a `LocalSet`.
///
/// # Type Parameters
///
/// * `StorageClientType` - The storage client the core polls the inbound queue through.
/// * `CoreFactoryType` - The factory that builds the core.
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
fn run_core_locally<
    StorageClientType: SchedulerStorageClient,
    CoreFactoryType: FnOnce() -> Core<StorageClientType>,
>(
    core_factory: CoreFactoryType,
) -> Result<Vec<TickSample>, CoreError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| CoreError::Internal(e.to_string()))?;
    let local_set = tokio::task::LocalSet::new();

    local_set.block_on(&runtime, core_factory().run())
}

/// The number of chances a job that produced no task gets before it is retired.
pub(crate) const DOWNGRADE_LIVES: u32 = 1;

/// The updates one tick produced for a single resource group.
#[derive(Default)]
struct RgUpdate {
    finalized: Vec<(JobId, FinalizeKind)>,
    new_jobs: Vec<SharedJobEntry>,
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
