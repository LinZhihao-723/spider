//! A lock-free latency histogram for the benchmark's hot paths.
//!
//! Dispatch latency is sampled on every request that returns an assignment, which at the largest
//! case is tens of thousands of samples per second across hundreds of concurrent tasks. Recording
//! into a lock or a growing vector there would measure the instrumentation rather than the
//! scheduler, so a sample is a handful of relaxed atomic increments into a fixed bucket array
//! instead.
//!
//! The buckets are log-linear: the first 64 nanoseconds are resolved to the nanosecond and each
//! octave above that is split into 64 buckets, giving sub-nanosecond-relative error at the low end
//! and a bounded 1.6% relative error everywhere. Because the layout is fixed, the raw bucket array
//! is the histogram's serialized form, and any percentile can be recomputed from a stored array
//! long after the run.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::bench::u64_to_f64;

/// The number of buckets every octave above the linear region is divided into.
pub const SUB_BUCKET_COUNT: u64 = 64;

/// The number of octaves the histogram tracks, the first of which is the linear region.
pub const NUM_OCTAVES: usize = 31;

/// The total number of buckets a histogram holds, `NUM_OCTAVES × SUB_BUCKET_COUNT`.
pub const BUCKET_COUNT: usize = NUM_OCTAVES * 64;

/// The largest value the histogram resolves, in nanoseconds. Larger samples are recorded in the
/// last bucket.
pub const MAX_TRACKED_NANOS: u64 = (1_u64 << (NUM_OCTAVES + 5)) - 1;

/// A fixed-bucket latency histogram that any number of tasks may record into concurrently.
///
/// Queries are meant to run once recording has quiesced: they read the buckets and the running
/// totals separately, so a query racing a recording may observe the two out of step.
#[derive(Debug)]
pub struct LatencyHistogram {
    buckets: Vec<AtomicU64>,
    total_nanos: AtomicU64,
    min_nanos: AtomicU64,
    max_nanos: AtomicU64,
}

impl LatencyHistogram {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created histogram holding no samples.
    #[must_use]
    pub fn new() -> Self {
        let mut buckets = Vec::with_capacity(BUCKET_COUNT);
        buckets.resize_with(BUCKET_COUNT, AtomicU64::default);

        Self {
            buckets,
            total_nanos: AtomicU64::new(0),
            min_nanos: AtomicU64::new(u64::MAX),
            max_nanos: AtomicU64::new(0),
        }
    }

    /// Records one sample.
    pub fn record(&self, sample: Duration) {
        self.record_nanos(u64::try_from(sample.as_nanos()).unwrap_or(u64::MAX));
    }

    /// Records one sample already expressed in nanoseconds.
    pub fn record_nanos(&self, nanos: u64) {
        self.buckets[bucket_index(nanos)].fetch_add(1, Ordering::Relaxed);
        self.total_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.min_nanos.fetch_min(nanos, Ordering::Relaxed);
        self.max_nanos.fetch_max(nanos, Ordering::Relaxed);
    }

    /// # Returns
    ///
    /// The number of samples recorded, summed over the buckets.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.buckets
            .iter()
            .map(|bucket| bucket.load(Ordering::Relaxed))
            .sum()
    }

    /// # Returns
    ///
    /// The arithmetic mean of the samples, computed from their exact sum rather than from the
    /// buckets, or [`Duration::ZERO`] if there are none.
    #[must_use]
    pub fn mean(&self) -> Duration {
        let count = self.count();
        if 0 == count {
            return Duration::ZERO;
        }

        Duration::from_nanos(self.total_nanos.load(Ordering::Relaxed) / count)
    }

    /// # Returns
    ///
    /// The smallest sample recorded, or [`Duration::ZERO`] if there are none.
    #[must_use]
    pub fn min(&self) -> Duration {
        let min_nanos = self.min_nanos.load(Ordering::Relaxed);
        if u64::MAX == min_nanos {
            return Duration::ZERO;
        }

        Duration::from_nanos(min_nanos)
    }

    /// # Returns
    ///
    /// The largest sample recorded, or [`Duration::ZERO`] if there are none.
    #[must_use]
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.max_nanos.load(Ordering::Relaxed))
    }

    /// Queries the samples at the given percentile.
    ///
    /// # Returns
    ///
    /// Forwards [`percentile_of`]'s return value over this histogram's buckets.
    #[must_use]
    pub fn percentile(&self, percentile: f64) -> Duration {
        percentile_of(&self.buckets(), percentile)
    }

    /// # Returns
    ///
    /// The raw bucket counts, which are the histogram's serialized form.
    #[must_use]
    pub fn buckets(&self) -> Vec<u64> {
        self.buckets
            .iter()
            .map(|bucket| bucket.load(Ordering::Relaxed))
            .collect()
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Queries a stored bucket array at the given percentile.
///
/// The convention is the nearest-rank method: the result is the bucket holding the sample of rank
/// `ceil(percentile / 100 × count)`, reported at that bucket's upper bound so that the stated
/// percentile is never understated. A percentile outside `[0, 100]` is clamped into that range and
/// a NaN percentile is treated as `0`.
///
/// # Returns
///
/// The latency at `percentile`, or [`Duration::ZERO`] if `buckets` holds no sample.
#[must_use]
pub fn percentile_of(buckets: &[u64], percentile: f64) -> Duration {
    let count: u64 = buckets.iter().sum();
    if 0 == count {
        return Duration::ZERO;
    }
    let ratio = if percentile.is_nan() {
        0.0
    } else {
        percentile.clamp(0.0, 100.0) / 100.0
    };
    let rank = ratio * u64_to_f64(count);

    let mut cumulative = 0_u64;
    for (index, bucket_count) in buckets.iter().enumerate() {
        cumulative += *bucket_count;
        if 0 == cumulative {
            continue;
        }
        if u64_to_f64(cumulative) >= rank {
            return Duration::from_nanos(bucket_upper_bound(index));
        }
    }

    Duration::from_nanos(bucket_upper_bound(buckets.len().saturating_sub(1)))
}

/// # Returns
///
/// The smallest value the bucket at `index` holds, in nanoseconds.
#[must_use]
pub fn bucket_lower_bound(index: usize) -> u64 {
    let octave = octave_of(index);
    let offset = u64::try_from(index.min(LAST_BUCKET_INDEX)).unwrap_or(0) % SUB_BUCKET_COUNT;
    if 0 == octave {
        return offset;
    }

    (1_u64 << (octave + SUB_BUCKET_BITS - 1)) + offset * bucket_width(octave)
}

/// # Returns
///
/// The largest value the bucket at `index` holds, in nanoseconds.
#[must_use]
pub fn bucket_upper_bound(index: usize) -> u64 {
    bucket_lower_bound(index) + bucket_width(octave_of(index)) - 1
}

/// The base-two logarithm of [`SUB_BUCKET_COUNT`].
const SUB_BUCKET_BITS: u64 = 6;

/// The index of the bucket every sample beyond [`MAX_TRACKED_NANOS`] falls into.
const LAST_BUCKET_INDEX: usize = BUCKET_COUNT - 1;

/// # Returns
///
/// The index of the bucket holding `nanos`, the last bucket if `nanos` exceeds
/// [`MAX_TRACKED_NANOS`].
fn bucket_index(nanos: u64) -> usize {
    let clamped = nanos.min(MAX_TRACKED_NANOS);
    if clamped < SUB_BUCKET_COUNT {
        return usize::try_from(clamped).unwrap_or(LAST_BUCKET_INDEX);
    }
    let most_significant_bit = u64::from(u64::BITS - 1 - clamped.leading_zeros());
    let octave = most_significant_bit + 1 - SUB_BUCKET_BITS;
    let offset =
        (clamped - (1_u64 << most_significant_bit)) >> (most_significant_bit - SUB_BUCKET_BITS);

    usize::try_from(octave * SUB_BUCKET_COUNT + offset).unwrap_or(LAST_BUCKET_INDEX)
}

/// # Returns
///
/// The octave the bucket at `index` belongs to, saturating at the last bucket's octave.
fn octave_of(index: usize) -> u64 {
    u64::try_from(index.min(LAST_BUCKET_INDEX)).unwrap_or(0) / SUB_BUCKET_COUNT
}

/// # Returns
///
/// The number of nanoseconds every bucket of `octave` spans.
const fn bucket_width(octave: u64) -> u64 {
    if 0 == octave {
        return 1;
    }

    1 << (octave - 1)
}
