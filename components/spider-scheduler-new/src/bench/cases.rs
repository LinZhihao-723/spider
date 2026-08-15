//! The benchmark case table.
//!
//! The table is the single source of truth for every dimension of a benchmark run: both bench
//! binaries select a case from it by name, and every result file embeds the case it was produced
//! from, so no dimension is ever restated in a script. The values are those of
//! `claude/scheduler-redesign/benchmark-contract.md` §2, which is normative for them.
//!
//! The dispatch queue capacity `B` of a case is derived rather than chosen. Under the admission
//! policy of `claude/scheduler-redesign/design.md` §6.2 with α = 1 and `N` backlogged groups, the
//! buffer's free space settles at `F = B / (N + 1)`. Requiring that the reserve
//!
//! ```text
//! R = (shared workers / #RG) + (dedicated workers per group)
//! ```
//!
//! always fits in that free space, so that every worker able to ask a single group for work finds
//! a slot, gives `F ≥ R` and therefore `B = R × (N + 1)`.

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::num::NonZeroUsize;

use serde::Deserialize;
use serde::Serialize;

use crate::bench::u64_to_f64;
use crate::bench::usize_to_f64;
use crate::config::CoreConfig;

/// The number of jobs each resource group contributes over a run.
pub const JOBS_PER_RESOURCE_GROUP: usize = 128;

/// The number of jobs released together, batch `n + 1` being released once every job of batch `n`
/// has had all of its tasks reported complete.
pub const JOBS_PER_BATCH: usize = 32;

/// The number of batches each resource group releases.
pub const NUM_BATCHES: usize = 4;

/// The number of tasks every job carries.
pub const TASKS_PER_JOB: usize = 1024;

/// The time a worker pretends to spend executing one task, in milliseconds.
pub const TASK_DURATION_MS: u64 = 5;

/// The number of ready tasks the fake inbound queue emits in a single poll response.
pub const INBOUND_WAVE_SIZE: usize = 256;

/// The number of active jobs each resource group may hold.
pub const ACTIVE_JOB_LIST_CAPACITY: usize = 16;

/// The cadence of the core's tick loop, in milliseconds.
pub const TICK_INTERVAL_MS: u64 = 1;

/// The time an inbound queue poll may block on the storage side, in milliseconds.
pub const STORAGE_POLL_TIMEOUT_MS: u64 = 5;

/// Every case a benchmark run may select, the smoke case last.
pub static ALL_CASES: [BenchCase; 6] = [
    BenchCase {
        name: Cow::Borrowed("A"),
        num_resource_groups: 1,
        jobs_per_resource_group: JOBS_PER_RESOURCE_GROUP,
        jobs_per_batch: JOBS_PER_BATCH,
        num_batches: NUM_BATCHES,
        tasks_per_job: TASKS_PER_JOB,
        task_duration_ms: TASK_DURATION_MS,
        inbound_wave_size: INBOUND_WAVE_SIZE,
        active_job_list_capacity: ACTIVE_JOB_LIST_CAPACITY,
        tick_interval_ms: TICK_INTERVAL_MS,
        storage_poll_timeout_ms: STORAGE_POLL_TIMEOUT_MS,
        dispatch_queue_capacity: 80,
        shared_workers: 8,
        dedicated_workers_per_group: 32,
    },
    BenchCase {
        name: Cow::Borrowed("B"),
        num_resource_groups: 2,
        jobs_per_resource_group: JOBS_PER_RESOURCE_GROUP,
        jobs_per_batch: JOBS_PER_BATCH,
        num_batches: NUM_BATCHES,
        tasks_per_job: TASKS_PER_JOB,
        task_duration_ms: TASK_DURATION_MS,
        inbound_wave_size: INBOUND_WAVE_SIZE,
        active_job_list_capacity: ACTIVE_JOB_LIST_CAPACITY,
        tick_interval_ms: TICK_INTERVAL_MS,
        storage_poll_timeout_ms: STORAGE_POLL_TIMEOUT_MS,
        dispatch_queue_capacity: 120,
        shared_workers: 16,
        dedicated_workers_per_group: 32,
    },
    BenchCase {
        name: Cow::Borrowed("C"),
        num_resource_groups: 4,
        jobs_per_resource_group: JOBS_PER_RESOURCE_GROUP,
        jobs_per_batch: JOBS_PER_BATCH,
        num_batches: NUM_BATCHES,
        tasks_per_job: TASKS_PER_JOB,
        task_duration_ms: TASK_DURATION_MS,
        inbound_wave_size: INBOUND_WAVE_SIZE,
        active_job_list_capacity: ACTIVE_JOB_LIST_CAPACITY,
        tick_interval_ms: TICK_INTERVAL_MS,
        storage_poll_timeout_ms: STORAGE_POLL_TIMEOUT_MS,
        dispatch_queue_capacity: 200,
        shared_workers: 32,
        dedicated_workers_per_group: 32,
    },
    BenchCase {
        name: Cow::Borrowed("D"),
        num_resource_groups: 8,
        jobs_per_resource_group: JOBS_PER_RESOURCE_GROUP,
        jobs_per_batch: JOBS_PER_BATCH,
        num_batches: NUM_BATCHES,
        tasks_per_job: TASKS_PER_JOB,
        task_duration_ms: TASK_DURATION_MS,
        inbound_wave_size: INBOUND_WAVE_SIZE,
        active_job_list_capacity: ACTIVE_JOB_LIST_CAPACITY,
        tick_interval_ms: TICK_INTERVAL_MS,
        storage_poll_timeout_ms: STORAGE_POLL_TIMEOUT_MS,
        dispatch_queue_capacity: 360,
        shared_workers: 64,
        dedicated_workers_per_group: 32,
    },
    BenchCase {
        name: Cow::Borrowed("E"),
        num_resource_groups: 8,
        jobs_per_resource_group: JOBS_PER_RESOURCE_GROUP,
        jobs_per_batch: JOBS_PER_BATCH,
        num_batches: NUM_BATCHES,
        tasks_per_job: TASKS_PER_JOB,
        task_duration_ms: TASK_DURATION_MS,
        inbound_wave_size: INBOUND_WAVE_SIZE,
        active_job_list_capacity: ACTIVE_JOB_LIST_CAPACITY,
        tick_interval_ms: TICK_INTERVAL_MS,
        storage_poll_timeout_ms: STORAGE_POLL_TIMEOUT_MS,
        dispatch_queue_capacity: 144,
        shared_workers: 128,
        dedicated_workers_per_group: 0,
    },
    BenchCase {
        name: Cow::Borrowed("SMOKE"),
        num_resource_groups: 2,
        jobs_per_resource_group: 4,
        jobs_per_batch: 2,
        num_batches: 2,
        tasks_per_job: 256,
        task_duration_ms: TASK_DURATION_MS,
        inbound_wave_size: INBOUND_WAVE_SIZE,
        active_job_list_capacity: ACTIVE_JOB_LIST_CAPACITY,
        tick_interval_ms: TICK_INTERVAL_MS,
        storage_poll_timeout_ms: STORAGE_POLL_TIMEOUT_MS,
        dispatch_queue_capacity: 24,
        shared_workers: 8,
        dedicated_workers_per_group: 4,
    },
];

/// One benchmark configuration, complete enough that a result file carrying it needs no other
/// context to be interpreted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchCase {
    /// The case's name, which selects it and names its result files.
    pub name: Cow<'static, str>,

    /// The number of resource groups the workload spans.
    pub num_resource_groups: usize,

    /// The number of jobs each resource group contributes over the run.
    pub jobs_per_resource_group: usize,

    /// The number of jobs released together.
    pub jobs_per_batch: usize,

    /// The number of batches each resource group releases.
    pub num_batches: usize,

    /// The number of tasks every job carries.
    pub tasks_per_job: usize,

    /// The time a worker pretends to spend executing one task, in milliseconds.
    pub task_duration_ms: u64,

    /// The number of ready tasks the fake inbound queue emits in a single poll response.
    pub inbound_wave_size: usize,

    /// The number of active jobs each resource group may hold.
    pub active_job_list_capacity: usize,

    /// The cadence of the core's tick loop, in milliseconds.
    pub tick_interval_ms: u64,

    /// The time an inbound queue poll may block on the storage side, in milliseconds.
    pub storage_poll_timeout_ms: u64,

    /// The total dispatch buffer size `B`, derived as `R × (N + 1)` from the module's reserve rule
    /// rather than chosen.
    pub dispatch_queue_capacity: usize,

    /// The number of general execution managers, which serve every resource group.
    pub shared_workers: usize,

    /// The number of execution managers pinned to each resource group.
    pub dedicated_workers_per_group: usize,
}

impl BenchCase {
    /// Looks the case named `name` up in [`ALL_CASES`], ignoring ASCII case.
    ///
    /// # Returns
    ///
    /// * The named case.
    /// * `None` if no case carries that name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        ALL_CASES
            .iter()
            .find(|case| case.name.eq_ignore_ascii_case(name))
            .cloned()
    }

    /// # Returns
    ///
    /// The number of jobs the whole workload carries.
    #[must_use]
    pub const fn total_jobs(&self) -> usize {
        self.num_resource_groups * self.jobs_per_resource_group
    }

    /// # Returns
    ///
    /// The number of tasks the whole workload carries.
    #[must_use]
    pub const fn total_tasks(&self) -> usize {
        self.total_jobs() * self.tasks_per_job
    }

    /// # Returns
    ///
    /// The number of execution managers the run drives, pinned and general together.
    #[must_use]
    pub const fn total_workers(&self) -> usize {
        self.shared_workers + self.num_resource_groups * self.dedicated_workers_per_group
    }

    /// # Returns
    ///
    /// The inbound queue's regular-lane fetch count, sized so that the core can buffer the entire
    /// workload and no task ever waits in the inbound queue.
    #[must_use]
    pub const fn ready_task_capacity(&self) -> usize {
        self.total_tasks()
    }

    /// # Returns
    ///
    /// The reserve `R`: the number of workers that may ask a single resource group for an
    /// assignment, which is the free space the dispatch buffer must keep.
    #[must_use]
    pub const fn reserve(&self) -> usize {
        match self.shared_workers.checked_div(self.num_resource_groups) {
            Some(shared_per_group) => shared_per_group + self.dedicated_workers_per_group,
            None => self.dedicated_workers_per_group,
        }
    }

    /// # Returns
    ///
    /// The dispatch queue capacity `R × (N + 1)` the module's rule prescribes, which
    /// [`Self::dispatch_queue_capacity`] must equal.
    #[must_use]
    pub const fn derived_dispatch_queue_capacity(&self) -> usize {
        self.reserve() * (self.num_resource_groups + 1)
    }

    /// # Returns
    ///
    /// The steady-state duration the case is expected to take, in seconds: the workload's total
    /// execution time spread over every worker. A run that greatly exceeds it is worker-starved
    /// rather than scheduler-bound.
    #[must_use]
    pub fn expected_duration_secs(&self) -> f64 {
        let num_workers = usize_to_f64(self.total_workers());
        if 0.0 == num_workers {
            return f64::INFINITY;
        }
        let total_task_ms = usize_to_f64(self.total_tasks()) * u64_to_f64(self.task_duration_ms);

        total_task_ms / (num_workers * 1000.0)
    }

    /// # Returns
    ///
    /// The core configuration the case runs the scheduler under.
    ///
    /// # Panics
    ///
    /// Panics if the case's dispatch queue capacity, active job list capacity, tick interval, or
    /// derived fetch counts are zero, which no case in [`ALL_CASES`] is.
    #[must_use]
    pub const fn core_config(&self) -> CoreConfig {
        let num_jobs = NonZeroUsize::new(self.total_jobs()).expect("the case has at least one job");

        CoreConfig {
            dispatch_queue_capacity: NonZeroUsize::new(self.dispatch_queue_capacity)
                .expect("the case's dispatch queue capacity is non-zero"),
            active_job_list_capacity: NonZeroUsize::new(self.active_job_list_capacity)
                .expect("the case's active job list capacity is non-zero"),
            ready_task_capacity: NonZeroUsize::new(self.ready_task_capacity())
                .expect("the case has at least one task"),
            commit_ready_task_capacity: num_jobs,
            cleanup_ready_task_capacity: num_jobs,
            storage_poll_timeout_ms: self.storage_poll_timeout_ms,
            tick_interval_ms: NonZeroU64::new(self.tick_interval_ms)
                .expect("the case's tick interval is non-zero"),
        }
    }

    /// # Returns
    ///
    /// The file name the scheduler side writes its results to.
    #[must_use]
    pub fn scheduler_result_file_name(&self) -> String {
        format!("{}-scheduler.json", self.name.to_lowercase())
    }

    /// # Returns
    ///
    /// The file name the worker process at `worker_index` writes its results to.
    #[must_use]
    pub fn worker_result_file_name(&self, worker_index: usize) -> String {
        format!("{}-workers-{worker_index}.json", self.name.to_lowercase())
    }
}
