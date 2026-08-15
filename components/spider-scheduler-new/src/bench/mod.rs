//! The scaffolding of the prototype scheduler's performance evaluation.
//!
//! The benchmark is specified by `claude/scheduler-redesign/benchmark-contract.md`: which
//! configurations are run, what is measured on each, and what the result files hold. This module
//! carries the pieces both bench binaries share -- the case table they select from, the lock-free
//! histogram the request path records into, and the serializable result types they write -- so that
//! a scheduler-side file and a worker-side file of the same run describe the same configuration by
//! construction.

pub mod cases;
pub mod histogram;
pub mod results;

#[cfg(test)]
mod tests;

pub use crate::bench::cases::ALL_CASES;
pub use crate::bench::cases::BenchCase;
pub use crate::bench::histogram::LatencyHistogram;
pub use crate::bench::results::JobE2eRecord;
pub use crate::bench::results::JobProgress;
pub use crate::bench::results::JobProgressMap;
pub use crate::bench::results::LatencySeries;
pub use crate::bench::results::SchedulerResults;
pub use crate::bench::results::TickSample;
pub use crate::bench::results::TickStep;
pub use crate::bench::results::TickTimer;
pub use crate::bench::results::WorkerResults;

/// The number of values a `u32` spans, the multiplier that recombines a `u64`'s two halves.
const U32_SPAN: f64 = 4_294_967_296.0;

/// Converts an integer into the floating-point domain the percentile and duration arithmetic uses.
///
/// The conversion goes through the value's two 32-bit halves because the crate's lint configuration
/// rejects the numeric casts a direct conversion would need.
///
/// # Returns
///
/// `value` as an `f64`, rounded to the nearest representable value beyond 2^53.
pub(crate) fn u64_to_f64(value: u64) -> f64 {
    let high = f64::from(u32::try_from(value >> 32).unwrap_or(u32::MAX));
    let low = f64::from(u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX));

    high.mul_add(U32_SPAN, low)
}

/// # Returns
///
/// Forwards [`u64_to_f64`]'s return value for `value`.
pub(crate) fn usize_to_f64(value: usize) -> f64 {
    u64_to_f64(u64::try_from(value).unwrap_or(u64::MAX))
}
