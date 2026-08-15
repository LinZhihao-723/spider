//! The benchmark's result types and the hot-path accumulators that feed them.
//!
//! Every quantity the benchmark contract asks for is stored raw where its volume allows -- one
//! record per tick and one per job -- and as a histogram where it does not, which is the four
//! dispatch latency series. Each result file embeds the case it was produced from, the build
//! profile, and the host's core count, so a file is interpretable on its own.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::thread::available_parallelism;
use std::time::Duration;
use std::time::Instant;

use dashmap::DashMap;
use serde::Deserialize;
use serde::Serialize;

use crate::bench::cases::BenchCase;
use crate::bench::histogram::LatencyHistogram;
use crate::bench::histogram::percentile_of;
use crate::types::JobId;
use crate::types::ResourceGroupId;

/// The per-job progress the dispatch service updates as completions arrive, sharded so that no
/// global lock appears on the request path.
pub type JobProgressMap = DashMap<JobId, JobProgress>;

/// One tick of the core, timed per step.
///
/// Steps 2 and 3 are skipped on a tick whose storage poll is still in flight, and read as zero
/// then. The total is measured over the whole tick and so may exceed the sum of the steps by the
/// tick's own bookkeeping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickSample {
    /// Step 1: draining the inbound poll result and the reschedule queue, and starting the next
    /// poll.
    pub collect_nanos: u64,

    /// Step 2: deduplicating and grouping the polled entries.
    pub process_nanos: u64,

    /// Step 3: creating scheduling units and placing new jobs.
    pub apply_nanos: u64,

    /// Step 4: the round-robin admission loop.
    pub fill_nanos: u64,

    /// Step 5: removing retired jobs from the job registry.
    pub retire_nanos: u64,

    /// The whole tick.
    pub total_nanos: u64,

    /// The number of assignments the tick published.
    pub assignments_published: u64,

    /// The number of resource groups that were active at the end of the tick.
    pub active_resource_groups: u64,
}

/// One step of the core's tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TickStep {
    /// Step 1.
    Collect,

    /// Step 2.
    Process,

    /// Step 3.
    Apply,

    /// Step 4.
    Fill,

    /// Step 5.
    Retire,
}

/// Accumulates a [`TickSample`] over one tick.
///
/// The timer reads the clock once per step boundary, so a step's cost is the interval between the
/// boundary that closed the previous step and the one that closes it. A step the tick skips is
/// simply never closed and stays zero, and the interval it did not spend is attributed to whichever
/// step is closed next.
#[derive(Clone, Copy, Debug)]
pub struct TickTimer {
    tick_start: Instant,
    step_start: Instant,
    sample: TickSample,
}

impl TickTimer {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created timer whose first step is already running.
    #[must_use]
    pub fn start() -> Self {
        let now = Instant::now();

        Self {
            tick_start: now,
            step_start: now,
            sample: TickSample::default(),
        }
    }

    /// Closes `step`, recording the time since the previous step boundary against it.
    pub fn finish_step(&mut self, step: TickStep) {
        let now = Instant::now();
        let nanos = duration_to_nanos(now.duration_since(self.step_start));
        match step {
            TickStep::Collect => self.sample.collect_nanos = nanos,
            TickStep::Process => self.sample.process_nanos = nanos,
            TickStep::Apply => self.sample.apply_nanos = nanos,
            TickStep::Fill => self.sample.fill_nanos = nanos,
            TickStep::Retire => self.sample.retire_nanos = nanos,
        }
        self.step_start = now;
    }

    /// Closes the tick.
    ///
    /// # Returns
    ///
    /// The tick's sample, carrying whatever steps were closed.
    #[must_use]
    pub fn finish(mut self, assignments_published: u64, active_resource_groups: u64) -> TickSample {
        self.sample.total_nanos = duration_to_nanos(self.tick_start.elapsed());
        self.sample.assignments_published = assignments_published;
        self.sample.active_resource_groups = active_resource_groups;

        self.sample
    }
}

/// One job's end-to-end timing, measured from the run's start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobE2eRecord {
    /// The job.
    pub job_id: u64,

    /// The resource group that owns the job.
    pub resource_group_id: u64,

    /// When the fake inbound queue first emitted a task of the job.
    pub first_seen_nanos: u64,

    /// When the dispatch service last received a completion report for a task of the job.
    pub last_completed_nanos: u64,

    /// The number of the job's tasks reported complete, which must equal the case's tasks per job
    /// for the run to be valid.
    pub tasks_completed: u64,
}

impl JobE2eRecord {
    /// # Returns
    ///
    /// The job's end-to-end duration, or [`Duration::ZERO`] if no completion was ever reported for
    /// it.
    #[must_use]
    pub const fn e2e(&self) -> Duration {
        Duration::from_nanos(
            self.last_completed_nanos
                .saturating_sub(self.first_seen_nanos),
        )
    }
}

/// One job's progress, updated concurrently on the dispatch service's request path.
#[derive(Debug)]
pub struct JobProgress {
    resource_group_id: ResourceGroupId,
    first_seen_nanos: AtomicU64,
    last_completed_nanos: AtomicU64,
    tasks_completed: AtomicU64,
}

impl JobProgress {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created progress entry first seen at `first_seen_nanos`, with no completion
    /// recorded.
    #[must_use]
    pub const fn new(resource_group_id: ResourceGroupId, first_seen_nanos: u64) -> Self {
        Self {
            resource_group_id,
            first_seen_nanos: AtomicU64::new(first_seen_nanos),
            last_completed_nanos: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
        }
    }

    /// Records that the job was seen at `nanos`, keeping the earliest such observation.
    pub fn note_first_seen(&self, nanos: u64) {
        self.first_seen_nanos.fetch_min(nanos, Ordering::Relaxed);
    }

    /// Records one completed task of the job, keeping the latest completion time.
    pub fn record_completion(&self, completed_nanos: u64) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
        self.last_completed_nanos
            .fetch_max(completed_nanos, Ordering::Relaxed);
    }

    /// # Returns
    ///
    /// The number of the job's tasks reported complete so far.
    #[must_use]
    pub fn tasks_completed(&self) -> u64 {
        self.tasks_completed.load(Ordering::Relaxed)
    }

    /// # Returns
    ///
    /// The job's progress as a storable record.
    #[must_use]
    pub fn to_record(&self, job_id: JobId) -> JobE2eRecord {
        JobE2eRecord {
            job_id: job_id.get(),
            resource_group_id: self.resource_group_id.get(),
            first_seen_nanos: self.first_seen_nanos.load(Ordering::Relaxed),
            last_completed_nanos: self.last_completed_nanos.load(Ordering::Relaxed),
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed),
        }
    }
}

/// A dispatch latency series, stored as its raw buckets so that any percentile can be recomputed
/// from the file, together with the summary the report quotes directly.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencySeries {
    /// The raw bucket counts, laid out as [`crate::bench::histogram`] describes.
    pub buckets: Vec<u64>,

    /// The number of samples.
    pub count: u64,

    /// The arithmetic mean.
    pub mean_nanos: u64,

    /// The smallest sample.
    pub min_nanos: u64,

    /// The largest sample.
    pub max_nanos: u64,

    /// The median.
    pub p50_nanos: u64,

    /// The 90th percentile.
    pub p90_nanos: u64,

    /// The 99th percentile.
    pub p99_nanos: u64,

    /// The 99.9th percentile.
    pub p999_nanos: u64,
}

impl LatencySeries {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created series summarizing `histogram`.
    #[must_use]
    pub fn from_histogram(histogram: &LatencyHistogram) -> Self {
        let buckets = histogram.buckets();

        Self {
            count: histogram.count(),
            mean_nanos: duration_to_nanos(histogram.mean()),
            min_nanos: duration_to_nanos(histogram.min()),
            max_nanos: duration_to_nanos(histogram.max()),
            p50_nanos: duration_to_nanos(percentile_of(&buckets, 50.0)),
            p90_nanos: duration_to_nanos(percentile_of(&buckets, 90.0)),
            p99_nanos: duration_to_nanos(percentile_of(&buckets, 99.0)),
            p999_nanos: duration_to_nanos(percentile_of(&buckets, 99.9)),
            buckets,
        }
    }

    /// Factory function.
    ///
    /// Folds owned samples -- the form the workers collect client-side latency in -- into a
    /// histogram and summarizes it.
    ///
    /// # Returns
    ///
    /// A newly created series over `samples`.
    #[must_use]
    pub fn from_samples(samples: &[Duration]) -> Self {
        let histogram = LatencyHistogram::new();
        for sample in samples {
            histogram.record(*sample);
        }

        Self::from_histogram(&histogram)
    }

    /// Recomputes a percentile from the stored buckets.
    ///
    /// # Returns
    ///
    /// Forwards [`percentile_of`]'s return value over this series' buckets.
    #[must_use]
    pub fn percentile(&self, percentile: f64) -> Duration {
        percentile_of(&self.buckets, percentile)
    }
}

/// Everything the scheduler side of one benchmark case produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerResults {
    /// The case the run was produced from.
    pub case: BenchCase,

    /// The number of cores the scheduler host reported.
    pub host_num_cores: usize,

    /// The build profile the scheduler binary was compiled with.
    pub build_profile: String,

    /// The wall-clock duration of the run, from the first inbound emission to the drain.
    pub wall_clock_nanos: u64,

    /// One sample per tick, in tick order.
    pub tick_samples: Vec<TickSample>,

    /// One record per job, in job order.
    pub job_e2e_records: Vec<JobE2eRecord>,

    /// Server-side latency of the pinned requests served without awaiting on an empty queue.
    pub pinned_immediate_dispatch_latency: LatencySeries,

    /// Server-side latency of the pinned requests that awaited an assignment's arrival.
    pub pinned_waited_dispatch_latency: LatencySeries,

    /// Server-side latency of the general requests served without awaiting on an empty queue.
    pub general_immediate_dispatch_latency: LatencySeries,

    /// Server-side latency of the general requests that awaited an assignment's arrival.
    pub general_waited_dispatch_latency: LatencySeries,

    /// The number of pinned requests that were served no assignment, which are never timed.
    pub num_pinned_empty_responses: u64,

    /// The number of general requests that were served no assignment, which are never timed.
    pub num_general_empty_responses: u64,
}

/// Everything one worker process of a benchmark case produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerResults {
    /// The case the run was produced from.
    pub case: BenchCase,

    /// The index of this worker process among the case's worker processes.
    pub worker_index: usize,

    /// The number of gRPC channels the process' worker coroutines spread over, which is one apiece
    /// unless the run deliberately narrowed it.
    pub channel_pool_size: usize,

    /// The number of cores the worker host reported.
    pub host_num_cores: usize,

    /// The build profile the worker binary was compiled with.
    pub build_profile: String,

    /// Client-side latency of this process' pinned requests the scheduler served immediately.
    pub pinned_immediate_dispatch_latency: LatencySeries,

    /// Client-side latency of this process' pinned requests the scheduler held until an assignment
    /// arrived.
    pub pinned_waited_dispatch_latency: LatencySeries,

    /// Client-side latency of this process' general requests the scheduler served immediately.
    pub general_immediate_dispatch_latency: LatencySeries,

    /// Client-side latency of this process' general requests the scheduler held until an
    /// assignment arrived.
    pub general_waited_dispatch_latency: LatencySeries,

    /// The number of assignments this process' pinned workers received.
    pub num_pinned_assignments: u64,

    /// The number of assignments this process' general workers received.
    pub num_general_assignments: u64,

    /// The number of responses that carried no assignment, which are excluded from the latency
    /// series.
    pub num_empty_responses: u64,
}

/// Snapshots every job's progress.
///
/// # Returns
///
/// One record per job, ordered by job ID.
#[must_use]
pub fn collect_job_records(progress: &JobProgressMap) -> Vec<JobE2eRecord> {
    let mut records: Vec<JobE2eRecord> = progress
        .iter()
        .map(|entry| entry.value().to_record(*entry.key()))
        .collect();
    records.sort_unstable_by_key(|record| record.job_id);

    records
}

/// # Returns
///
/// The build profile the crate was compiled with, which a result file records so that a debug run
/// is never mistaken for a measurement.
#[must_use]
pub const fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// # Returns
///
/// The number of cores the host reports, or `0` if it reports none.
#[must_use]
pub fn host_num_cores() -> usize {
    available_parallelism().map_or(0, std::num::NonZeroUsize::get)
}

/// # Returns
///
/// `duration` in nanoseconds, saturated at [`u64::MAX`].
#[must_use]
pub fn duration_to_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
