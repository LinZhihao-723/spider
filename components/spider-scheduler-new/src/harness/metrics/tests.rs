//! Unit tests for the benchmark workers' client-side measurements.

use std::time::Duration;

use crate::harness::metrics::DispatchClass;
use crate::harness::metrics::WorkerKind;
use crate::harness::metrics::WorkerPoolReport;
use crate::harness::metrics::WorkerSamples;
use crate::proto::DispatchClass as WireDispatchClass;
use crate::types::ResourceGroupId;

/// # Returns
///
/// Samples for a worker on `kind`'s path that timed `immediate_ms` immediately served requests and
/// `waited_ms` waited ones, and saw `num_empty_responses` responses without an assignment.
fn worker_samples(
    kind: WorkerKind,
    immediate_ms: &[u64],
    waited_ms: &[u64],
    num_empty_responses: u64,
) -> WorkerSamples {
    let mut samples = WorkerSamples::with_capacity(kind, immediate_ms.len() + waited_ms.len());
    for latency_ms in immediate_ms {
        samples.record(DispatchClass::Immediate, Duration::from_millis(*latency_ms));
    }
    for latency_ms in waited_ms {
        samples.record(DispatchClass::Waited, Duration::from_millis(*latency_ms));
    }
    samples.num_empty_responses = num_empty_responses;

    samples
}

#[test]
fn worker_kind_follows_the_presence_of_a_resource_group() {
    assert_eq!(
        WorkerKind::from_resource_group_id(Some(ResourceGroupId::from(3))),
        WorkerKind::Pinned
    );
    assert_eq!(
        WorkerKind::from_resource_group_id(None),
        WorkerKind::General
    );
}

#[test]
fn only_the_wire_classes_that_carry_an_assignment_are_timed() {
    assert_eq!(
        DispatchClass::from_wire(WireDispatchClass::Immediate),
        Some(DispatchClass::Immediate)
    );
    assert_eq!(
        DispatchClass::from_wire(WireDispatchClass::Waited),
        Some(DispatchClass::Waited)
    );
    assert_eq!(DispatchClass::from_wire(WireDispatchClass::Empty), None);
    assert_eq!(
        DispatchClass::from_wire(WireDispatchClass::Unspecified),
        None
    );
}

#[test]
fn a_pool_report_keeps_the_two_dispatch_paths_apart() {
    let samples = [
        worker_samples(WorkerKind::Pinned, &[1, 3], &[], 2),
        worker_samples(WorkerKind::General, &[10, 20, 30], &[], 5),
        worker_samples(WorkerKind::Pinned, &[5], &[], 1),
    ];
    let report = WorkerPoolReport::from_samples(4, true, &samples);

    assert_eq!(report.channel_pool_size, 4);
    assert!(report.drained);
    assert_eq!(report.num_pinned_assignments(), 3);
    assert_eq!(report.num_general_assignments(), 3);
    assert_eq!(report.num_empty_responses, 8);
    assert_eq!(report.pinned_immediate_dispatch_latency.count, 3);
    assert_eq!(report.general_immediate_dispatch_latency.count, 3);
    assert!(
        report.pinned_immediate_dispatch_latency.max_nanos
            < report.general_immediate_dispatch_latency.min_nanos
    );
}

#[test]
fn a_pool_report_keeps_the_two_dispatch_classes_apart() {
    let samples = [
        worker_samples(WorkerKind::Pinned, &[1, 2], &[40], 0),
        worker_samples(WorkerKind::General, &[3], &[50, 60], 0),
    ];
    let report = WorkerPoolReport::from_samples(3, true, &samples);

    assert_eq!(report.pinned_immediate_dispatch_latency.count, 2);
    assert_eq!(report.pinned_waited_dispatch_latency.count, 1);
    assert_eq!(report.general_immediate_dispatch_latency.count, 1);
    assert_eq!(report.general_waited_dispatch_latency.count, 2);
    assert_eq!(report.num_pinned_assignments(), 3);
    assert_eq!(report.num_general_assignments(), 3);
    assert!(
        report.pinned_immediate_dispatch_latency.max_nanos
            < report.pinned_waited_dispatch_latency.min_nanos
    );
}

#[test]
fn a_pool_report_over_no_samples_is_empty() {
    let report = WorkerPoolReport::from_samples(1, false, &[]);

    assert_eq!(report.num_pinned_assignments(), 0);
    assert_eq!(report.num_general_assignments(), 0);
    assert_eq!(report.num_empty_responses, 0);
    assert_eq!(report.pinned_immediate_dispatch_latency.count, 0);
    assert_eq!(report.pinned_waited_dispatch_latency.count, 0);
    assert_eq!(report.general_immediate_dispatch_latency.count, 0);
    assert_eq!(report.general_waited_dispatch_latency.count, 0);
}

#[test]
fn samples_count_only_the_requests_that_returned_an_assignment() {
    let samples = worker_samples(WorkerKind::General, &[2, 4], &[6, 8], 11);

    assert_eq!(samples.num_assignments(), 4);
    assert_eq!(samples.num_empty_responses, 11);
}
