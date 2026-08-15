//! Dispatch and latency measurements collected by the harness workers.

use std::time::Duration;

use crate::bench::cases::BenchCase;
use crate::bench::histogram::LatencyHistogram;
use crate::bench::results::LatencySeries;
use crate::bench::results::WorkerResults;
use crate::bench::results::build_profile;
use crate::bench::results::host_num_cores;
use crate::harness::fake_worker::WorkerReport;
use crate::proto::DispatchClass as WireDispatchClass;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskAssignmentId;
use crate::types::TaskId;

#[cfg(test)]
mod tests;

/// One assignment a worker received, together with the latency of the request that carried it and
/// the time the worker finished executing it.
#[derive(Clone, Debug)]
pub struct DispatchRecord {
    /// The identifier the scheduler stamped onto the assignment when it published it.
    pub assignment_id: TaskAssignmentId,

    /// The resource group the assignment belongs to.
    pub resource_group_id: ResourceGroupId,

    /// The job the assigned task belongs to.
    pub job_id: JobId,

    /// The assigned task.
    pub task_id: TaskId,

    /// Client-side latency: request send to response receipt.
    pub latency: Duration,

    /// Elapsed time from the start of the run to when the worker finished executing this
    /// assignment.
    pub completed_at: Duration,
}

/// The merged, sorted client-side latencies of every request a run's workers completed.
///
/// The samples are sorted once, when the instance is built, so repeated queries are index lookups.
#[derive(Clone, Debug, Default)]
pub struct LatencySamples(Vec<Duration>);

impl LatencySamples {
    /// Merges the latency samples of every worker in `reports`.
    ///
    /// # Returns
    ///
    /// The merged samples, sorted in ascending order.
    #[must_use]
    pub fn from_reports(reports: &[WorkerReport]) -> Self {
        let num_samples: usize = reports.iter().map(|report| report.latencies.len()).sum();
        let mut samples = Vec::with_capacity(num_samples);
        for report in reports {
            samples.extend_from_slice(&report.latencies);
        }
        samples.sort_unstable();

        Self(samples)
    }

    /// # Returns
    ///
    /// The number of samples.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.0.len()
    }

    /// # Returns
    ///
    /// The arithmetic mean of the samples, or [`Duration::ZERO`] if there are none.
    #[must_use]
    pub fn mean(&self) -> Duration {
        let num_samples = u32::try_from(self.0.len()).unwrap_or(u32::MAX);
        if 0 == num_samples {
            return Duration::ZERO;
        }
        let total = self.0.iter().fold(Duration::ZERO, |accumulated, sample| {
            accumulated.saturating_add(*sample)
        });

        total / num_samples
    }

    /// Queries the samples at the given percentile.
    ///
    /// The convention is linear interpolation between adjacent order statistics, the default of
    /// `numpy.percentile`: over `n` sorted samples, `percentile` selects the fractional rank
    /// `r = percentile / 100 * (n - 1)`, and the result is the linear blend of the samples at
    /// `floor(r)` and `ceil(r)` weighted by `r`'s fractional part. A percentile outside `[0, 100]`
    /// is clamped into that range, and a NaN percentile is treated as `0`.
    ///
    /// # Returns
    ///
    /// The latency at `percentile`, or [`Duration::ZERO`] if there are no samples.
    #[must_use]
    pub fn percentile(&self, percentile: f64) -> Duration {
        let Some(last_index) = self.0.len().checked_sub(1) else {
            return Duration::ZERO;
        };
        let ratio = if percentile.is_nan() {
            0.0
        } else {
            percentile.clamp(0.0, 100.0) / 100.0
        };
        let rank = ratio * index_to_f64(last_index);
        let lower_index = floor_rank_index(rank, last_index);
        let lower = self.0[lower_index];
        let upper = self.0[(lower_index + 1).min(last_index)];
        let fraction = rank - index_to_f64(lower_index);

        lower.saturating_add(upper.saturating_sub(lower).mul_f64(fraction))
    }
}

/// The dispatch path a worker exercises.
///
/// The benchmark reports the two separately because they are structurally different: a pinned
/// request touches one resource group's queue and nothing else, while a general request pops the
/// global hint channel and may traverse several stale hints before it finds work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerKind {
    /// An execution manager pinned to a single resource group.
    Pinned,

    /// An execution manager that accepts assignments from any resource group.
    General,
}

impl WorkerKind {
    /// # Returns
    ///
    /// The path an execution manager identifying itself with `resource_group_id` takes.
    #[must_use]
    pub const fn from_resource_group_id(resource_group_id: Option<ResourceGroupId>) -> Self {
        match resource_group_id {
            Some(_) => Self::Pinned,
            None => Self::General,
        }
    }
}

/// The class a request that returned an assignment was served in.
///
/// The scheduler reports the class it served a request in, and the client files its own sample by
/// that same class, so that `client - server` is a difference over one population rather than over
/// two. The class an assignment-less response carries is not represented: such a request measures
/// the long-poll wait rather than a dispatch and is only ever counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchClass {
    /// An assignment was available when the request reached the scheduler.
    Immediate,

    /// The scheduler held the request on an empty queue until an assignment arrived.
    Waited,
}

impl DispatchClass {
    /// # Returns
    ///
    /// The class `wire` names, or `None` if it names one that is never timed.
    #[must_use]
    pub const fn from_wire(wire: WireDispatchClass) -> Option<Self> {
        match wire {
            WireDispatchClass::Immediate => Some(Self::Immediate),
            WireDispatchClass::Waited => Some(Self::Waited),
            WireDispatchClass::Unspecified | WireDispatchClass::Empty => None,
        }
    }
}

/// The client-side latencies one benchmark worker owns for the length of a run.
///
/// Nothing here is shared with another worker, so recording a sample costs a single push into one
/// of two buffers reserved before the first request; the merge into the run's histograms happens
/// only once every worker has stopped. The two buffers are the classes the scheduler serves a
/// request in, kept apart because a waited request measures the supply of work rather than the cost
/// of a dispatch. Responses that carried no assignment are counted but never timed at all, their
/// latency being the long-poll wait.
#[derive(Clone, Debug)]
pub struct WorkerSamples {
    /// The dispatch path the worker exercised.
    pub kind: WorkerKind,

    /// The latency of every request the scheduler served immediately.
    pub immediate_latencies: Vec<Duration>,

    /// The latency of every request the scheduler held until an assignment arrived.
    pub waited_latencies: Vec<Duration>,

    /// The number of responses that carried no assignment.
    pub num_empty_responses: u64,
}

impl WorkerSamples {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// Newly created samples for a worker on `kind`'s path, with room for `capacity` latencies of
    /// each class.
    #[must_use]
    pub fn with_capacity(kind: WorkerKind, capacity: usize) -> Self {
        Self {
            kind,
            immediate_latencies: Vec::with_capacity(capacity),
            waited_latencies: Vec::with_capacity(capacity),
            num_empty_responses: 0,
        }
    }

    /// Files `latency` under the class the scheduler served the request in.
    pub fn record(&mut self, dispatch_class: DispatchClass, latency: Duration) {
        match dispatch_class {
            DispatchClass::Immediate => self.immediate_latencies.push(latency),
            DispatchClass::Waited => self.waited_latencies.push(latency),
        }
    }

    /// # Returns
    ///
    /// The number of assignments the worker received, which is the number of timed requests.
    #[must_use]
    pub fn num_assignments(&self) -> u64 {
        let num_samples = self
            .immediate_latencies
            .len()
            .saturating_add(self.waited_latencies.len());

        u64::try_from(num_samples).unwrap_or(u64::MAX)
    }
}

/// The merged client-side measurements of every worker one benchmark process hosted.
///
/// The four latency series are the dispatch path crossed with the class the scheduler served the
/// request in, which is the shape the server-side series take as well: the two sides are only
/// comparable series by series, because a waited request measures the supply of work and an
/// immediate one measures the cost of handing it over.
#[derive(Clone, Debug)]
pub struct WorkerPoolReport {
    /// The number of gRPC channels the process' workers spread over.
    pub channel_pool_size: usize,

    /// Whether the scheduler reported the run drained before the workers stopped. A report taken
    /// from a run that did not drain measures a truncated workload.
    pub drained: bool,

    /// Client-side latency of the requests the pinned workers had served immediately.
    pub pinned_immediate_dispatch_latency: LatencySeries,

    /// Client-side latency of the requests the pinned workers waited on.
    pub pinned_waited_dispatch_latency: LatencySeries,

    /// Client-side latency of the requests the general workers had served immediately.
    pub general_immediate_dispatch_latency: LatencySeries,

    /// Client-side latency of the requests the general workers waited on.
    pub general_waited_dispatch_latency: LatencySeries,

    /// The number of responses that carried no assignment.
    pub num_empty_responses: u64,
}

impl WorkerPoolReport {
    /// Merges the owned samples of every worker of a run into one series per dispatch path and
    /// class.
    ///
    /// # Returns
    ///
    /// A newly created report over `samples`.
    #[must_use]
    pub fn from_samples(
        channel_pool_size: usize,
        drained: bool,
        samples: &[WorkerSamples],
    ) -> Self {
        let pinned_immediate = LatencyHistogram::new();
        let pinned_waited = LatencyHistogram::new();
        let general_immediate = LatencyHistogram::new();
        let general_waited = LatencyHistogram::new();
        let mut num_empty_responses: u64 = 0;
        for worker_samples in samples {
            let (immediate, waited) = match worker_samples.kind {
                WorkerKind::Pinned => (&pinned_immediate, &pinned_waited),
                WorkerKind::General => (&general_immediate, &general_waited),
            };
            for latency in &worker_samples.immediate_latencies {
                immediate.record(*latency);
            }
            for latency in &worker_samples.waited_latencies {
                waited.record(*latency);
            }
            num_empty_responses =
                num_empty_responses.saturating_add(worker_samples.num_empty_responses);
        }

        Self {
            channel_pool_size,
            drained,
            pinned_immediate_dispatch_latency: LatencySeries::from_histogram(&pinned_immediate),
            pinned_waited_dispatch_latency: LatencySeries::from_histogram(&pinned_waited),
            general_immediate_dispatch_latency: LatencySeries::from_histogram(&general_immediate),
            general_waited_dispatch_latency: LatencySeries::from_histogram(&general_waited),
            num_empty_responses,
        }
    }

    /// # Returns
    ///
    /// The number of assignments the pinned workers received, of either class.
    #[must_use]
    pub const fn num_pinned_assignments(&self) -> u64 {
        self.pinned_immediate_dispatch_latency
            .count
            .saturating_add(self.pinned_waited_dispatch_latency.count)
    }

    /// # Returns
    ///
    /// The number of assignments the general workers received, of either class.
    #[must_use]
    pub const fn num_general_assignments(&self) -> u64 {
        self.general_immediate_dispatch_latency
            .count
            .saturating_add(self.general_waited_dispatch_latency.count)
    }

    /// Stamps the report with the case and the host it was produced on.
    ///
    /// # Returns
    ///
    /// The self-describing result file the worker process at `worker_index` writes.
    #[must_use]
    pub fn into_results(self, case: BenchCase, worker_index: usize) -> WorkerResults {
        let num_pinned_assignments = self.num_pinned_assignments();
        let num_general_assignments = self.num_general_assignments();

        WorkerResults {
            case,
            worker_index,
            channel_pool_size: self.channel_pool_size,
            host_num_cores: host_num_cores(),
            build_profile: build_profile().to_owned(),
            pinned_immediate_dispatch_latency: self.pinned_immediate_dispatch_latency,
            pinned_waited_dispatch_latency: self.pinned_waited_dispatch_latency,
            general_immediate_dispatch_latency: self.general_immediate_dispatch_latency,
            general_waited_dispatch_latency: self.general_waited_dispatch_latency,
            num_pinned_assignments,
            num_general_assignments,
            num_empty_responses: self.num_empty_responses,
        }
    }
}

/// Locates the integer part of `rank` among the sample indices.
///
/// The search compares indices against `rank` rather than converting `rank` to an integer, because
/// the crate's lint configuration rejects floating-point-to-integer casts.
///
/// # Returns
///
/// The greatest index in `0..=last_index` that does not exceed `rank`.
fn floor_rank_index(rank: f64, last_index: usize) -> usize {
    let mut low = 0;
    let mut high = last_index;
    while low < high {
        let mid = low.midpoint(high) + 1;
        if index_to_f64(mid) <= rank {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    low
}

/// Converts a sample index into the floating-point domain the rank arithmetic uses.
///
/// # Returns
///
/// `index` as an `f64`, saturated at [`u32::MAX`] for sample counts beyond a `u32`.
fn index_to_f64(index: usize) -> f64 {
    f64::from(u32::try_from(index).unwrap_or(u32::MAX))
}
