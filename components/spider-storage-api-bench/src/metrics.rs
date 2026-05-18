use std::{
    collections::HashMap,
    sync::{
        Arc,
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use serde::{Deserialize, Serialize};
use tabled::{Table, Tabled};
use uuid::Uuid;

/// Whether a storage request can wait inside the storage service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestCategory {
    NonBlocking,
    Blocking,
    Phase,
}

impl RequestCategory {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonBlocking => "non_blocking",
            Self::Blocking => "blocking",
            Self::Phase => "phase",
        }
    }
}

/// Completed latency observation for one storage request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestLatencySample {
    pub operation: String,
    pub category: String,
    pub latency_micros: u128,
    pub succeeded: bool,
}

impl RequestLatencySample {
    #[must_use]
    pub fn success(
        operation: &'static str,
        category: RequestCategory,
        latency: std::time::Duration,
    ) -> Self {
        Self {
            operation: operation.to_owned(),
            category: category.as_str().to_owned(),
            latency_micros: latency.as_micros(),
            succeeded: true,
        }
    }

    #[must_use]
    pub fn failure(
        operation: &'static str,
        category: RequestCategory,
        latency: std::time::Duration,
    ) -> Self {
        Self {
            operation: operation.to_owned(),
            category: category.as_str().to_owned(),
            latency_micros: latency.as_micros(),
            succeeded: false,
        }
    }
}

/// Completed end-to-end latency observation for one benchmark job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobLatencySample {
    pub latency_micros: u128,
    pub succeeded: bool,
}

impl JobLatencySample {
    #[must_use]
    pub const fn success(latency: std::time::Duration) -> Self {
        Self {
            latency_micros: latency.as_micros(),
            succeeded: true,
        }
    }

    #[must_use]
    pub const fn failure(latency: std::time::Duration) -> Self {
        Self {
            latency_micros: latency.as_micros(),
            succeeded: false,
        }
    }
}

/// End-to-end latency summary printed to the console.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Tabled)]
pub struct JobLatencySummary {
    pub count: usize,
    pub failed_jobs: usize,
    pub p50_us: u128,
    pub p90_us: u128,
    pub p99_us: u128,
    pub max_us: u128,
}

/// Storage request latency summary printed to the console.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Tabled)]
pub struct RequestLatencySummary {
    pub category: String,
    pub operation: String,
    pub count: usize,
    pub errors: usize,
    pub avg_us: u128,
    pub p50_us: u128,
    pub p90_us: u128,
    pub p99_us: u128,
    pub max_us: u128,
}

/// Single byte-size observation recorded by the storage service.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestSizeSample {
    pub operation: String,
    pub bytes: u64,
}

/// Aggregated byte-size observations for one operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Tabled)]
pub struct RequestSizeSummary {
    pub operation: String,
    pub count: usize,
    pub avg_bytes: u64,
    pub p50_bytes: u64,
    pub p99_bytes: u64,
    pub max_bytes: u64,
    pub total_bytes: u64,
}

/// Server-side metrics summary for one benchmark session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerMetricsSessionReport {
    pub metrics_session_id: String,
    pub label: Option<String>,
    pub elapsed_micros: u128,
    pub request_latency: Vec<RequestLatencySummary>,
    pub low_count_request_latency: Vec<RequestLatencySample>,
    #[serde(default)]
    pub request_sizes: Vec<RequestSizeSummary>,
}

/// In-memory server-side metrics sessions for benchmark runs.
#[derive(Clone, Default)]
pub struct ServerMetricsRegistry {
    inner: Arc<ServerMetricsRegistryInner>,
}

#[derive(Default)]
struct ServerMetricsRegistryInner {
    active_count: AtomicUsize,
    sessions: Mutex<HashMap<String, ServerMetricsSession>>,
}

struct ServerMetricsSession {
    label: Option<String>,
    started_at: Instant,
    samples: Vec<RequestLatencySample>,
    size_samples: Vec<RequestSizeSample>,
}

impl ServerMetricsRegistry {
    /// Starts collecting request metrics for a benchmark session.
    ///
    /// # Returns
    ///
    /// A newly generated metrics session ID.
    ///
    /// # Panics
    ///
    /// Panics if the metrics session lock is poisoned.
    #[must_use]
    pub fn start_session(&self, label: Option<String>) -> String {
        let session_id = Uuid::new_v4().to_string();
        let session = ServerMetricsSession {
            label,
            started_at: Instant::now(),
            samples: Vec::new(),
            size_samples: Vec::new(),
        };
        self.inner
            .sessions
            .lock()
            .expect("server metrics sessions lock should not be poisoned")
            .insert(session_id.clone(), session);
        self.inner.active_count.fetch_add(1, Ordering::Relaxed);
        session_id
    }

    /// Stops collecting request metrics for a benchmark session.
    ///
    /// # Returns
    ///
    /// The finalized session report if the session exists.
    ///
    /// # Panics
    ///
    /// Panics if the metrics session lock is poisoned.
    #[must_use]
    pub fn end_session(&self, session_id: &str) -> Option<ServerMetricsSessionReport> {
        let session = self
            .inner
            .sessions
            .lock()
            .expect("server metrics sessions lock should not be poisoned")
            .remove(session_id)?;
        self.inner.active_count.fetch_sub(1, Ordering::Relaxed);
        Some(ServerMetricsSessionReport {
            metrics_session_id: session_id.to_owned(),
            label: session.label,
            elapsed_micros: session.started_at.elapsed().as_micros(),
            request_latency: summarize_requests(&session.samples),
            low_count_request_latency: low_count_samples(&session.samples),
            request_sizes: summarize_sizes(&session.size_samples),
        })
    }

    /// Records one storage request observation into all active sessions.
    ///
    /// # Panics
    ///
    /// Panics if the metrics session lock is poisoned.
    pub fn record_request(
        &self,
        operation: &'static str,
        category: RequestCategory,
        latency: std::time::Duration,
        succeeded: bool,
    ) {
        if self.inner.active_count.load(Ordering::Relaxed) == 0 {
            return;
        }
        let sample = if succeeded {
            RequestLatencySample::success(operation, category, latency)
        } else {
            RequestLatencySample::failure(operation, category, latency)
        };
        for session in self
            .inner
            .sessions
            .lock()
            .expect("server metrics sessions lock should not be poisoned")
            .values_mut()
        {
            session.samples.push(sample.clone());
        }
    }

    /// Records a byte-size observation into all active sessions.
    ///
    /// # Panics
    ///
    /// Panics if the metrics session lock is poisoned.
    pub fn record_size(&self, operation: &'static str, bytes: u64) {
        if self.inner.active_count.load(Ordering::Relaxed) == 0 {
            return;
        }
        let sample = RequestSizeSample {
            operation: operation.to_owned(),
            bytes,
        };
        for session in self
            .inner
            .sessions
            .lock()
            .expect("server metrics sessions lock should not be poisoned")
            .values_mut()
        {
            session.size_samples.push(sample.clone());
        }
    }
}

fn low_count_samples(samples: &[RequestLatencySample]) -> Vec<RequestLatencySample> {
    let mut counts: HashMap<(&str, &str), usize> = HashMap::new();
    for sample in samples {
        *counts
            .entry((sample.category.as_str(), sample.operation.as_str()))
            .or_default() += 1;
    }
    samples
        .iter()
        .filter(|sample| {
            counts
                .get(&(sample.category.as_str(), sample.operation.as_str()))
                .is_some_and(|count| *count <= LOW_COUNT_SAMPLE_LIMIT)
        })
        .cloned()
        .collect()
}

const LOW_COUNT_SAMPLE_LIMIT: usize = 100;

/// Aggregates job latency samples into a summary row.
#[must_use]
pub fn summarize(samples: &[JobLatencySample]) -> JobLatencySummary {
    let mut latencies: Vec<u128> = samples.iter().map(|sample| sample.latency_micros).collect();
    latencies.sort_unstable();
    JobLatencySummary {
        count: latencies.len(),
        failed_jobs: samples.iter().filter(|sample| !sample.succeeded).count(),
        p50_us: percentile(&latencies, 50),
        p90_us: percentile(&latencies, 90),
        p99_us: percentile(&latencies, 99),
        max_us: latencies.last().copied().unwrap_or(0),
    }
}

/// Aggregates storage request latency samples into table rows.
#[must_use]
pub fn summarize_requests(samples: &[RequestLatencySample]) -> Vec<RequestLatencySummary> {
    let mut groups: Vec<(&str, &str)> = samples
        .iter()
        .map(|sample| (sample.category.as_str(), sample.operation.as_str()))
        .collect();
    groups.sort_unstable();
    groups.dedup();

    groups
        .into_iter()
        .map(|(category, operation)| {
            let mut latencies: Vec<u128> = samples
                .iter()
                .filter(|sample| sample.category == category && sample.operation == operation)
                .map(|sample| sample.latency_micros)
                .collect();
            latencies.sort_unstable();
            RequestLatencySummary {
                category: category.to_owned(),
                operation: operation.to_owned(),
                count: latencies.len(),
                errors: samples
                    .iter()
                    .filter(|sample| {
                        sample.category == category
                            && sample.operation == operation
                            && !sample.succeeded
                    })
                    .count(),
                avg_us: average(&latencies),
                p50_us: percentile(&latencies, 50),
                p90_us: percentile(&latencies, 90),
                p99_us: percentile(&latencies, 99),
                max_us: latencies.last().copied().unwrap_or(0),
            }
        })
        .collect()
}

/// Aggregates byte-size samples into per-operation rows.
#[must_use]
pub fn summarize_sizes(samples: &[RequestSizeSample]) -> Vec<RequestSizeSummary> {
    let mut operations: Vec<&str> = samples
        .iter()
        .map(|sample| sample.operation.as_str())
        .collect();
    operations.sort_unstable();
    operations.dedup();

    operations
        .into_iter()
        .map(|operation| {
            let mut values: Vec<u64> = samples
                .iter()
                .filter(|sample| sample.operation == operation)
                .map(|sample| sample.bytes)
                .collect();
            values.sort_unstable();
            let total: u128 = values.iter().map(|&v| u128::from(v)).sum();
            let count = values.len();
            let avg = if count == 0 {
                0
            } else {
                u64::try_from(total / count as u128).unwrap_or(u64::MAX)
            };
            RequestSizeSummary {
                operation: operation.to_owned(),
                count,
                avg_bytes: avg,
                p50_bytes: size_percentile(&values, 50),
                p99_bytes: size_percentile(&values, 99),
                max_bytes: values.last().copied().unwrap_or(0),
                total_bytes: u64::try_from(total).unwrap_or(u64::MAX),
            }
        })
        .collect()
}

fn size_percentile(sorted_values: &[u64], percentile_value: usize) -> u64 {
    if sorted_values.is_empty() {
        return 0;
    }
    let index = ((sorted_values.len() - 1) * percentile_value).div_ceil(100);
    sorted_values[index]
}

/// Renders the job latency summary as a console table.
#[must_use]
pub fn render_summary(summary: &JobLatencySummary) -> String {
    Table::new([summary]).to_string()
}

/// Renders request latency summary rows as a console table.
#[must_use]
pub fn render_request_summary(rows: &[RequestLatencySummary]) -> String {
    Table::new(rows).to_string()
}

fn percentile(sorted_values: &[u128], percentile_value: usize) -> u128 {
    if sorted_values.is_empty() {
        return 0;
    }
    let index = ((sorted_values.len() - 1) * percentile_value).div_ceil(100);
    sorted_values[index]
}

fn average(values: &[u128]) -> u128 {
    if values.is_empty() {
        return 0;
    }
    values.iter().sum::<u128>() / values.len() as u128
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{JobLatencySample, RequestCategory, ServerMetricsRegistry, summarize};

    #[test]
    fn summarize_computes_job_latency_percentiles_and_failures() {
        let samples = vec![
            JobLatencySample::success(Duration::from_micros(10)),
            JobLatencySample::success(Duration::from_micros(20)),
            JobLatencySample::failure(Duration::from_micros(100)),
        ];
        let summary = summarize(&samples);
        assert_eq!(3, summary.count);
        assert_eq!(1, summary.failed_jobs);
        assert_eq!(20, summary.p50_us);
        assert_eq!(100, summary.p90_us);
        assert_eq!(100, summary.p99_us);
    }

    #[test]
    fn summarize_requests_groups_by_operation_and_category() {
        let samples = vec![
            super::RequestLatencySample::success(
                "register_job",
                super::RequestCategory::NonBlocking,
                Duration::from_micros(10),
            ),
            super::RequestLatencySample::failure(
                "register_job",
                super::RequestCategory::NonBlocking,
                Duration::from_micros(20),
            ),
            super::RequestLatencySample::success(
                "poll_ready_tasks",
                super::RequestCategory::Blocking,
                Duration::from_micros(100),
            ),
        ];
        let rows = super::summarize_requests(&samples);
        assert_eq!(2, rows.len());
        assert_eq!("blocking", rows[0].category);
        assert_eq!("poll_ready_tasks", rows[0].operation);
        assert_eq!(1, rows[0].count);
        assert_eq!(0, rows[0].errors);
        assert_eq!("non_blocking", rows[1].category);
        assert_eq!("register_job", rows[1].operation);
        assert_eq!(2, rows[1].count);
        assert_eq!(1, rows[1].errors);
        assert_eq!(15, rows[1].avg_us);
    }

    #[test]
    fn server_metrics_session_reports_only_samples_inside_session() {
        let registry = ServerMetricsRegistry::default();
        registry.record_request(
            "register_job",
            RequestCategory::NonBlocking,
            Duration::from_micros(5),
            true,
        );

        let session_id = registry.start_session(Some("flat".to_owned()));
        registry.record_request(
            "register_job",
            RequestCategory::NonBlocking,
            Duration::from_micros(10),
            true,
        );
        registry.record_request(
            "register_job",
            RequestCategory::NonBlocking,
            Duration::from_micros(30),
            false,
        );
        registry.record_request(
            "poll_ready_tasks",
            RequestCategory::Blocking,
            Duration::from_micros(100),
            true,
        );

        let report = registry
            .end_session(&session_id)
            .expect("started session should end");
        assert_eq!(session_id, report.metrics_session_id);
        assert_eq!(Some("flat".to_owned()), report.label);
        assert_eq!(2, report.request_latency.len());
        assert_eq!(3, report.low_count_request_latency.len());
        assert_eq!("blocking", report.request_latency[0].category);
        assert_eq!("poll_ready_tasks", report.request_latency[0].operation);
        assert_eq!(1, report.request_latency[0].count);
        assert_eq!("non_blocking", report.request_latency[1].category);
        assert_eq!("register_job", report.request_latency[1].operation);
        assert_eq!(2, report.request_latency[1].count);
        assert_eq!(1, report.request_latency[1].errors);
    }

    #[test]
    fn server_metrics_session_can_only_end_once() {
        let registry = ServerMetricsRegistry::default();
        let session_id = registry.start_session(None);
        assert!(registry.end_session(&session_id).is_some());
        assert!(registry.end_session(&session_id).is_none());
    }
}
