//! Dispatch and latency measurements collected by the harness workers.

use std::time::Duration;

use crate::harness::fake_worker::WorkerReport;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskAssignmentId;
use crate::types::TaskId;

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
