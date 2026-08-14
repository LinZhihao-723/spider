//! Unit tests for the latency statistics the harness reports a run with.
//!
//! The percentile query is hand-rolled float arithmetic over an integer-indexed sample vector, so
//! the cases pinned here are the ones that arithmetic gets wrong: the ends of the range, a sample
//! set too small to interpolate over, and a percentile that is not a number in `[0, 100]`.

use std::time::Duration;

use crate::harness::LatencySamples;
use crate::harness::WorkerReport;

/// The samples every interpolation test queries, in the order a worker recorded them.
const UNSORTED_SAMPLE_MS: [u64; 4] = [30, 10, 40, 20];

#[test]
fn from_reports_merges_and_sorts_every_worker_s_samples() {
    let samples = make_samples(&[&[30, 10], &[40, 20]]);

    assert_eq!(samples.count(), UNSORTED_SAMPLE_MS.len());
    assert_eq!(samples.percentile(0.0), Duration::from_millis(10));
    assert_eq!(samples.percentile(100.0), Duration::from_millis(40));
}

#[test]
fn percentile_interpolates_between_adjacent_samples() {
    let samples = make_samples(&[&UNSORTED_SAMPLE_MS]);

    assert_eq!(samples.percentile(0.0), Duration::from_millis(10));
    assert_eq!(samples.percentile(25.0), Duration::from_micros(17_500));
    assert_eq!(samples.percentile(50.0), Duration::from_millis(25));
    assert_eq!(samples.percentile(100.0), Duration::from_millis(40));
}

#[test]
fn percentile_clamps_a_value_outside_the_range_and_treats_nan_as_zero() {
    let samples = make_samples(&[&UNSORTED_SAMPLE_MS]);

    assert_eq!(samples.percentile(-10.0), Duration::from_millis(10));
    assert_eq!(
        samples.percentile(f64::NEG_INFINITY),
        samples.percentile(0.0)
    );
    assert_eq!(samples.percentile(150.0), Duration::from_millis(40));
    assert_eq!(samples.percentile(f64::INFINITY), samples.percentile(100.0));
    assert_eq!(samples.percentile(f64::NAN), Duration::from_millis(10));
}

#[test]
fn a_single_sample_answers_every_percentile() {
    let samples = make_samples(&[&[7]]);

    assert_eq!(samples.count(), 1);
    assert_eq!(samples.percentile(0.0), Duration::from_millis(7));
    assert_eq!(samples.percentile(50.0), Duration::from_millis(7));
    assert_eq!(samples.percentile(100.0), Duration::from_millis(7));
    assert_eq!(samples.mean(), Duration::from_millis(7));
}

#[test]
fn an_empty_sample_set_answers_zero() {
    let samples = LatencySamples::from_reports(&[]);

    assert_eq!(samples.count(), 0);
    assert_eq!(samples.mean(), Duration::ZERO);
    assert_eq!(samples.percentile(0.0), Duration::ZERO);
    assert_eq!(samples.percentile(99.9), Duration::ZERO);
    assert_eq!(samples.percentile(f64::NAN), Duration::ZERO);
}

#[test]
fn mean_averages_every_sample() {
    assert_eq!(
        make_samples(&[&UNSORTED_SAMPLE_MS]).mean(),
        Duration::from_millis(25)
    );
    assert_eq!(
        make_samples(&[&[1, 2], &[], &[3]]).mean(),
        Duration::from_millis(2)
    );
}

/// # Returns
///
/// The merged samples of one worker report per entry of `latencies_ms`, each carrying that entry's
/// latencies in milliseconds.
fn make_samples(latencies_ms: &[&[u64]]) -> LatencySamples {
    let reports: Vec<WorkerReport> = latencies_ms
        .iter()
        .map(|worker_latencies_ms| WorkerReport {
            latencies: worker_latencies_ms
                .iter()
                .map(|latency_ms| Duration::from_millis(*latency_ms))
                .collect(),
            ..WorkerReport::default()
        })
        .collect();

    LatencySamples::from_reports(&reports)
}
