//! Unit tests for the benchmark scaffolding.
//!
//! The case table is checked against the benchmark contract's table row by row, and the histogram
//! against hand-computable sample sets, so that a change to either is caught here rather than in a
//! result file nobody can reproduce.

use std::time::Duration;

use crate::bench::cases::ALL_CASES;
use crate::bench::cases::BenchCase;
use crate::bench::histogram::BUCKET_COUNT;
use crate::bench::histogram::LatencyHistogram;
use crate::bench::histogram::MAX_TRACKED_NANOS;
use crate::bench::histogram::NUM_OCTAVES;
use crate::bench::histogram::SUB_BUCKET_COUNT;
use crate::bench::histogram::bucket_lower_bound;
use crate::bench::histogram::bucket_upper_bound;
use crate::bench::histogram::percentile_of;
use crate::bench::results::JobProgress;
use crate::bench::results::JobProgressMap;
use crate::bench::results::LatencySeries;
use crate::bench::results::TickStep;
use crate::bench::results::TickTimer;
use crate::bench::results::collect_job_records;
use crate::types::JobId;
use crate::types::ResourceGroupId;

/// One row of the benchmark contract's case table.
struct ExpectedCase {
    name: &'static str,
    num_resource_groups: usize,
    total_tasks: usize,
    shared_workers: usize,
    dedicated_workers_per_group: usize,
    total_workers: usize,
    reserve: usize,
    dispatch_queue_capacity: usize,
}

/// The contract's table, transcribed independently of the case table it checks.
const EXPECTED_CASES: [ExpectedCase; 6] = [
    ExpectedCase {
        name: "A",
        num_resource_groups: 1,
        total_tasks: 131_072,
        shared_workers: 8,
        dedicated_workers_per_group: 32,
        total_workers: 40,
        reserve: 40,
        dispatch_queue_capacity: 80,
    },
    ExpectedCase {
        name: "B",
        num_resource_groups: 2,
        total_tasks: 262_144,
        shared_workers: 16,
        dedicated_workers_per_group: 32,
        total_workers: 80,
        reserve: 40,
        dispatch_queue_capacity: 120,
    },
    ExpectedCase {
        name: "C",
        num_resource_groups: 4,
        total_tasks: 524_288,
        shared_workers: 32,
        dedicated_workers_per_group: 32,
        total_workers: 160,
        reserve: 40,
        dispatch_queue_capacity: 200,
    },
    ExpectedCase {
        name: "D",
        num_resource_groups: 8,
        total_tasks: 1_048_576,
        shared_workers: 64,
        dedicated_workers_per_group: 32,
        total_workers: 320,
        reserve: 40,
        dispatch_queue_capacity: 360,
    },
    ExpectedCase {
        name: "E",
        num_resource_groups: 8,
        total_tasks: 1_048_576,
        shared_workers: 128,
        dedicated_workers_per_group: 0,
        total_workers: 128,
        reserve: 16,
        dispatch_queue_capacity: 144,
    },
    ExpectedCase {
        name: "SMOKE",
        num_resource_groups: 2,
        total_tasks: 2048,
        shared_workers: 8,
        dedicated_workers_per_group: 4,
        total_workers: 16,
        reserve: 8,
        dispatch_queue_capacity: 24,
    },
];

#[test]
fn case_table_matches_the_contract() {
    assert_eq!(ALL_CASES.len(), EXPECTED_CASES.len());
    for (case, expected) in ALL_CASES.iter().zip(EXPECTED_CASES.iter()) {
        assert_eq!(case.name, expected.name);
        assert_eq!(case.num_resource_groups, expected.num_resource_groups);
        assert_eq!(case.total_tasks(), expected.total_tasks);
        assert_eq!(case.shared_workers, expected.shared_workers);
        assert_eq!(
            case.dedicated_workers_per_group,
            expected.dedicated_workers_per_group
        );
        assert_eq!(case.total_workers(), expected.total_workers);
        assert_eq!(case.reserve(), expected.reserve);
        assert_eq!(
            case.dispatch_queue_capacity,
            expected.dispatch_queue_capacity
        );
    }
}

#[test]
fn dispatch_queue_capacity_follows_the_reserve_rule() {
    for case in &ALL_CASES {
        assert_eq!(
            case.dispatch_queue_capacity,
            case.derived_dispatch_queue_capacity(),
            "case {} does not satisfy B = R * (N + 1)",
            case.name
        );
    }
}

#[test]
fn every_case_yields_a_usable_core_config() {
    for case in &ALL_CASES {
        let config = case.core_config();
        assert_eq!(
            config.dispatch_queue_capacity.get(),
            case.dispatch_queue_capacity
        );
        assert_eq!(config.ready_task_capacity.get(), case.total_tasks());
        assert_eq!(config.active_job_list_capacity.get(), 16);
        assert_eq!(config.tick_interval_ms.get(), 1);
        assert_eq!(config.storage_poll_timeout_ms, 5);
        assert_eq!(
            case.jobs_per_batch * case.num_batches,
            case.jobs_per_resource_group
        );
    }
}

#[test]
fn cases_are_selectable_by_name() {
    for case in &ALL_CASES {
        let selected = BenchCase::from_name(&case.name.to_lowercase())
            .expect("every table entry is selectable by its own name");
        assert_eq!(&selected, case);
    }
    assert_eq!(BenchCase::from_name("F"), None);
}

#[test]
fn expected_duration_matches_the_contract() {
    let case_a = BenchCase::from_name("A").expect("case A is in the table");
    assert!((case_a.expected_duration_secs() - 16.384).abs() < 1e-9);
    let case_e = BenchCase::from_name("E").expect("case E is in the table");
    assert!((case_e.expected_duration_secs() - 40.96).abs() < 1e-9);
}

#[test]
fn result_file_names_follow_the_contract() {
    let case = BenchCase::from_name("smoke").expect("the smoke case is in the table");
    assert_eq!(case.scheduler_result_file_name(), "smoke-scheduler.json");
    assert_eq!(case.worker_result_file_name(2), "smoke-workers-2.json");
}

#[test]
fn buckets_tile_the_tracked_range_without_gaps() {
    assert_eq!(
        BUCKET_COUNT,
        NUM_OCTAVES * usize::try_from(SUB_BUCKET_COUNT).expect("64 fits in a usize")
    );
    assert_eq!(bucket_lower_bound(0), 0);
    assert_eq!(bucket_upper_bound(BUCKET_COUNT - 1), MAX_TRACKED_NANOS);
    for index in 0..BUCKET_COUNT - 1 {
        assert_eq!(bucket_upper_bound(index) + 1, bucket_lower_bound(index + 1));
    }
}

#[test]
fn every_sample_lands_in_a_bucket_that_contains_it() {
    let samples = [
        0,
        1,
        63,
        64,
        127,
        128,
        1023,
        1024,
        1_000_000,
        1_000_000_000,
        MAX_TRACKED_NANOS,
    ];
    for sample in samples {
        let histogram = LatencyHistogram::new();
        histogram.record_nanos(sample);
        let buckets = histogram.buckets();
        let index = buckets
            .iter()
            .position(|count| 1 == *count)
            .expect("the sample was recorded into some bucket");
        assert!(bucket_lower_bound(index) <= sample);
        assert!(sample <= bucket_upper_bound(index));
    }
}

#[test]
fn low_end_resolution_is_sub_microsecond() {
    for sample in [1_u64, 63, 512, 1024] {
        let histogram = LatencyHistogram::new();
        histogram.record_nanos(sample);
        let buckets = histogram.buckets();
        let index = buckets
            .iter()
            .position(|count| 1 == *count)
            .expect("the sample was recorded into some bucket");
        let width = bucket_upper_bound(index) - bucket_lower_bound(index) + 1;
        assert!(width <= 16, "sample {sample} landed in a {width} ns bucket");
    }
}

#[test]
fn percentiles_of_a_dense_sample_set_are_exact() {
    let histogram = LatencyHistogram::new();
    for nanos in 1..=100_u64 {
        histogram.record_nanos(nanos);
    }

    assert_eq!(histogram.count(), 100);
    assert_eq!(histogram.min(), Duration::from_nanos(1));
    assert_eq!(histogram.max(), Duration::from_nanos(100));
    assert_eq!(histogram.mean(), Duration::from_nanos(50));
    assert_eq!(histogram.percentile(0.0), Duration::from_nanos(1));
    assert_eq!(histogram.percentile(50.0), Duration::from_nanos(50));
    assert_eq!(histogram.percentile(90.0), Duration::from_nanos(90));
    assert_eq!(histogram.percentile(99.0), Duration::from_nanos(99));
    assert_eq!(histogram.percentile(99.9), Duration::from_nanos(100));
    assert_eq!(histogram.percentile(100.0), Duration::from_nanos(100));
}

#[test]
fn percentiles_ignore_the_tail_a_few_outliers_form() {
    let histogram = LatencyHistogram::new();
    for _ in 0..999 {
        histogram.record_nanos(10);
    }
    histogram.record_nanos(1_000_000);

    assert_eq!(histogram.percentile(99.0), Duration::from_nanos(10));
    assert!(histogram.percentile(100.0) >= Duration::from_millis(1));
    assert_eq!(histogram.max(), Duration::from_millis(1));
}

#[test]
fn out_of_range_percentiles_are_clamped() {
    let histogram = LatencyHistogram::new();
    histogram.record_nanos(7);
    histogram.record_nanos(9);

    assert_eq!(histogram.percentile(-10.0), Duration::from_nanos(7));
    assert_eq!(histogram.percentile(f64::NAN), Duration::from_nanos(7));
    assert_eq!(histogram.percentile(1000.0), Duration::from_nanos(9));
}

#[test]
fn an_empty_histogram_reports_zeros() {
    let histogram = LatencyHistogram::new();

    assert_eq!(histogram.count(), 0);
    assert_eq!(histogram.min(), Duration::ZERO);
    assert_eq!(histogram.max(), Duration::ZERO);
    assert_eq!(histogram.mean(), Duration::ZERO);
    assert_eq!(histogram.percentile(50.0), Duration::ZERO);
    assert_eq!(percentile_of(&histogram.buckets(), 50.0), Duration::ZERO);
}

#[test]
fn samples_beyond_the_tracked_range_land_in_the_last_bucket() {
    let histogram = LatencyHistogram::new();
    histogram.record(Duration::from_hours(1));

    let buckets = histogram.buckets();
    assert_eq!(buckets[BUCKET_COUNT - 1], 1);
    assert_eq!(histogram.max(), Duration::from_hours(1));
    assert_eq!(
        histogram.percentile(50.0),
        Duration::from_nanos(MAX_TRACKED_NANOS)
    );
}

#[test]
fn a_series_recomputes_percentiles_from_its_stored_buckets() {
    let samples: Vec<Duration> = (1..=100_u64).map(Duration::from_nanos).collect();
    let series = LatencySeries::from_samples(&samples);

    assert_eq!(series.count, 100);
    assert_eq!(series.min_nanos, 1);
    assert_eq!(series.max_nanos, 100);
    assert_eq!(series.mean_nanos, 50);
    assert_eq!(series.p50_nanos, 50);
    assert_eq!(series.p90_nanos, 90);
    assert_eq!(series.p99_nanos, 99);
    assert_eq!(series.p999_nanos, 100);
    assert_eq!(series.buckets.len(), BUCKET_COUNT);
    assert_eq!(series.percentile(75.0), Duration::from_nanos(75));
}

#[test]
fn a_tick_timer_leaves_skipped_steps_at_zero() {
    let mut timer = TickTimer::start();
    timer.finish_step(TickStep::Collect);
    timer.finish_step(TickStep::Fill);
    let sample = timer.finish(7, 3);

    assert_eq!(sample.process_nanos, 0);
    assert_eq!(sample.apply_nanos, 0);
    assert_eq!(sample.retire_nanos, 0);
    assert_eq!(sample.assignments_published, 7);
    assert_eq!(sample.active_resource_groups, 3);
    assert!(sample.total_nanos >= sample.collect_nanos + sample.fill_nanos);
}

#[test]
fn job_progress_keeps_the_earliest_arrival_and_the_latest_completion() {
    let progress = JobProgress::new(ResourceGroupId::from(4), 100);
    progress.note_first_seen(150);
    progress.note_first_seen(50);
    progress.record_completion(300);
    progress.record_completion(200);

    let record = progress.to_record(JobId::from(9));
    assert_eq!(record.job_id, 9);
    assert_eq!(record.resource_group_id, 4);
    assert_eq!(record.first_seen_nanos, 50);
    assert_eq!(record.last_completed_nanos, 300);
    assert_eq!(record.tasks_completed, 2);
    assert_eq!(progress.tasks_completed(), 2);
    assert_eq!(record.e2e(), Duration::from_nanos(250));
}

#[test]
fn job_records_are_collected_in_job_order() {
    let progress = JobProgressMap::new();
    progress.insert(
        JobId::from(3),
        JobProgress::new(ResourceGroupId::from(1), 30),
    );
    progress.insert(
        JobId::from(1),
        JobProgress::new(ResourceGroupId::from(1), 10),
    );
    progress.insert(
        JobId::from(2),
        JobProgress::new(ResourceGroupId::from(2), 20),
    );

    let records = collect_job_records(&progress);
    let job_ids: Vec<u64> = records.iter().map(|record| record.job_id).collect();
    assert_eq!(job_ids, vec![1, 2, 3]);
}
