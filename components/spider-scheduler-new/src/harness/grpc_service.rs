//! The tonic service exposing the prototype scheduler to harness workers.
//!
//! The harness drives the prototype through the same gRPC surface the production scheduler
//! presents, so a measurement taken here includes every cost a real execution manager pays. The
//! service is a thin adapter: it neither buffers nor retries, and the only decision it makes is
//! whether a request is pinned to a resource group or general.
//!
//! A benchmark run additionally attaches the instrumentation the benchmark contract asks for --
//! server-side dispatch latency, per-job completion progress, and the run-status the workers poll
//! -- all of which record through atomics so that the request path stays lock-free at the rates the
//! largest case reaches.

use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;

use crate::bench::histogram::LatencyHistogram;
use crate::bench::results::JobProgress;
use crate::bench::results::JobProgressMap;
use crate::bench::results::duration_to_nanos;
use crate::dispatch_queue::DispatchOutcome;
use crate::dispatch_queue::DispatchService;
use crate::error::HarnessError;
use crate::harness::fake_storage::FakeStorage;
use crate::proto::Assignment;
use crate::proto::DispatchClass;
use crate::proto::NextTaskRequest;
use crate::proto::NextTaskResponse;
use crate::proto::PrototypeSchedulerService;
use crate::proto::PrototypeSchedulerServiceServer;
use crate::proto::RunStatusRequest;
use crate::proto::RunStatusResponse;
use crate::proto::Void;
use crate::proto::next_task_response;
use crate::proto::prototype_bench_service_server::PrototypeBenchService;
use crate::proto::prototype_bench_service_server::PrototypeBenchServiceServer;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskId;

/// The resource group a completion report is attributed to when it names a job the run has never
/// seen emitted and the reporting execution manager is not pinned.
const UNKNOWN_RESOURCE_GROUP_ID: u64 = 0;

/// A tonic server serving one [`DispatchService`] on an ephemeral local port.
#[derive(Debug)]
pub struct HarnessServer {
    endpoint: String,
    cancellation_token: CancellationToken,
    serve_task: JoinHandle<Result<(), tonic::transport::Error>>,
}

impl HarnessServer {
    /// Factory function.
    ///
    /// Binds an ephemeral port on the loopback interface and starts serving in the background, so
    /// that concurrently running integration tests never collide on a port.
    ///
    /// # Returns
    ///
    /// A newly created server already accepting connections on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`Self::start_on`]'s return values on failure.
    pub async fn start(
        dispatch_service: DispatchService,
        storage: FakeStorage,
    ) -> Result<Self, HarnessError> {
        Self::start_on(
            dispatch_service,
            storage,
            SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            None,
        )
        .await
    }

    /// Factory function.
    ///
    /// Serves the same dispatch surface as [`Self::start`] on a caller-chosen address, with
    /// `instrumentation` recording every request and answering the run-status the benchmark's
    /// workers poll.
    ///
    /// # Returns
    ///
    /// A newly created server already accepting connections on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`Self::start_on`]'s return values on failure.
    pub async fn start_bench(
        dispatch_service: DispatchService,
        storage: FakeStorage,
        bind_address: SocketAddr,
        instrumentation: Arc<BenchInstrumentation>,
    ) -> Result<Self, HarnessError> {
        Self::start_on(
            dispatch_service,
            storage,
            bind_address,
            Some(instrumentation),
        )
        .await
    }

    /// # Returns
    ///
    /// The URL a client connects to in order to reach this server.
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.endpoint.clone()
    }

    /// Asks the server to stop accepting connections and waits for it to finish serving.
    pub async fn shutdown(self) {
        self.cancellation_token.cancel();
        match self.serve_task.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    error = % error,
                    "Harness server stopped with a transport error."
                );
            }
            Err(error) => {
                tracing::warn!(
                    error = % error,
                    "Harness server task could not be joined."
                );
            }
        }
    }

    /// Binds `bind_address` and starts serving the dispatch and bench services in the background.
    ///
    /// # Returns
    ///
    /// A newly created server already accepting connections on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`HarnessError::Bind`] if `bind_address` could not be bound or the assigned address could
    ///   not be read back.
    async fn start_on(
        dispatch_service: DispatchService,
        storage: FakeStorage,
        bind_address: SocketAddr,
        instrumentation: Option<Arc<BenchInstrumentation>>,
    ) -> Result<Self, HarnessError> {
        let listener = TcpListener::bind(bind_address)
            .await
            .map_err(|error| HarnessError::Bind(error.to_string()))?;
        let local_addr = listener
            .local_addr()
            .map_err(|error| HarnessError::Bind(error.to_string()))?;

        let cancellation_token = CancellationToken::new();
        let shutdown_token = cancellation_token.clone();
        let bench_service = BenchAdapter {
            instrumentation: instrumentation.clone(),
        };
        let serve_task = tokio::spawn(
            Server::builder()
                .add_service(PrototypeSchedulerServiceServer::new(Adapter {
                    dispatch_service,
                    storage,
                    instrumentation,
                }))
                .add_service(PrototypeBenchServiceServer::new(bench_service))
                .serve_with_incoming_shutdown(TcpIncoming::from(listener), async move {
                    shutdown_token.cancelled().await;
                }),
        );

        Ok(Self {
            endpoint: format!("http://{local_addr}"),
            cancellation_token,
            serve_task,
        })
    }
}

/// Everything the dispatch service records on behalf of a benchmark run.
///
/// Every accumulator is an atomic or a sharded map of atomics, so a request never waits on another
/// request to record. All timestamps are offsets from the run origin the constructor is given,
/// which is what lets a completion timestamp taken here be compared against a first-seen timestamp
/// taken by whichever component emits the workload.
///
/// Latency is kept in one histogram per path and class, because pooling an immediate request with
/// one that blocked waiting for work to exist would report the supply of work as dispatch cost.
#[derive(Debug)]
pub struct BenchInstrumentation {
    run_origin: Instant,
    total_tasks: u64,
    pinned_immediate_dispatch_latency: LatencyHistogram,
    pinned_waited_dispatch_latency: LatencyHistogram,
    general_immediate_dispatch_latency: LatencyHistogram,
    general_waited_dispatch_latency: LatencyHistogram,
    job_progress: Arc<JobProgressMap>,
    tasks_completed: AtomicU64,
    num_pinned_empty_responses: AtomicU64,
    num_general_empty_responses: AtomicU64,
}

impl BenchInstrumentation {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created instrumentation holding no samples, over a run of `total_tasks` tasks whose
    /// timestamps are measured from `run_origin`.
    #[must_use]
    pub fn new(run_origin: Instant, total_tasks: u64) -> Self {
        Self::with_progress(run_origin, total_tasks, Arc::new(JobProgressMap::new()))
    }

    /// Factory function.
    ///
    /// Records completions into a progress map the caller already holds, which is what lets the
    /// component emitting the workload record the other end of a job's end-to-end time into the
    /// same entries. Both sides must be built from the same `run_origin`, or the two ends would be
    /// measured against different clocks.
    ///
    /// # Returns
    ///
    /// A newly created instrumentation holding no samples of its own, over a run of `total_tasks`
    /// tasks whose timestamps are measured from `run_origin`.
    #[must_use]
    pub fn with_progress(
        run_origin: Instant,
        total_tasks: u64,
        job_progress: Arc<JobProgressMap>,
    ) -> Self {
        Self {
            run_origin,
            total_tasks,
            pinned_immediate_dispatch_latency: LatencyHistogram::new(),
            pinned_waited_dispatch_latency: LatencyHistogram::new(),
            general_immediate_dispatch_latency: LatencyHistogram::new(),
            general_waited_dispatch_latency: LatencyHistogram::new(),
            job_progress,
            tasks_completed: AtomicU64::new(0),
            num_pinned_empty_responses: AtomicU64::new(0),
            num_general_empty_responses: AtomicU64::new(0),
        }
    }

    /// Records that a task of `job_id` was emitted into the inbound queue, keeping the earliest
    /// such observation as the job's first-seen timestamp.
    pub fn note_job_first_seen(&self, job_id: JobId, resource_group_id: ResourceGroupId) {
        let nanos = self.elapsed_nanos();
        self.job_progress
            .entry(job_id)
            .and_modify(|progress| progress.note_first_seen(nanos))
            .or_insert_with(|| JobProgress::new(resource_group_id, nanos));
    }

    /// # Returns
    ///
    /// The server-side latency of the requests pinned execution managers made that were served
    /// without ever awaiting on an empty queue, which is what dispatch costs.
    #[must_use]
    pub const fn pinned_immediate_dispatch_latency(&self) -> &LatencyHistogram {
        &self.pinned_immediate_dispatch_latency
    }

    /// # Returns
    ///
    /// The server-side latency of the requests pinned execution managers made that awaited an
    /// assignment's arrival, which is dominated by supply rather than by the scheduler.
    #[must_use]
    pub const fn pinned_waited_dispatch_latency(&self) -> &LatencyHistogram {
        &self.pinned_waited_dispatch_latency
    }

    /// # Returns
    ///
    /// The server-side latency of the requests general execution managers made that were served
    /// without ever awaiting on an empty hint channel, which is what dispatch costs.
    #[must_use]
    pub const fn general_immediate_dispatch_latency(&self) -> &LatencyHistogram {
        &self.general_immediate_dispatch_latency
    }

    /// # Returns
    ///
    /// The server-side latency of the requests general execution managers made that awaited a
    /// hint's arrival, which is dominated by supply rather than by the scheduler.
    #[must_use]
    pub const fn general_waited_dispatch_latency(&self) -> &LatencyHistogram {
        &self.general_waited_dispatch_latency
    }

    /// # Returns
    ///
    /// Every job the run has seen, from which the per-job end-to-end records are built.
    #[must_use]
    pub fn job_progress(&self) -> &JobProgressMap {
        &self.job_progress
    }

    /// # Returns
    ///
    /// The number of tasks the run must complete before it is drained.
    #[must_use]
    pub const fn total_tasks(&self) -> u64 {
        self.total_tasks
    }

    /// # Returns
    ///
    /// The number of completion reports received so far.
    #[must_use]
    pub fn tasks_completed(&self) -> u64 {
        self.tasks_completed.load(Ordering::Relaxed)
    }

    /// # Returns
    ///
    /// The number of requests pinned execution managers made that were served no assignment, which
    /// the contract counts but never times.
    #[must_use]
    pub fn num_pinned_empty_responses(&self) -> u64 {
        self.num_pinned_empty_responses.load(Ordering::Relaxed)
    }

    /// # Returns
    ///
    /// The number of requests general execution managers made that were served no assignment,
    /// which the contract counts but never times.
    #[must_use]
    pub fn num_general_empty_responses(&self) -> u64 {
        self.num_general_empty_responses.load(Ordering::Relaxed)
    }

    /// # Returns
    ///
    /// The number of requests over both paths that were served no assignment.
    #[must_use]
    pub fn num_empty_responses(&self) -> u64 {
        self.num_pinned_empty_responses()
            .saturating_add(self.num_general_empty_responses())
    }

    /// # Returns
    ///
    /// Whether every task of the run has been reported complete.
    #[must_use]
    pub fn is_drained(&self) -> bool {
        self.tasks_completed() >= self.total_tasks
    }

    /// # Returns
    ///
    /// The time elapsed since the run origin, in nanoseconds.
    #[must_use]
    pub fn elapsed_nanos(&self) -> u64 {
        duration_to_nanos(self.run_origin.elapsed())
    }

    /// Records one request that was served an assignment without ever awaiting on an empty queue.
    fn record_immediate_dispatch(&self, pinned: bool, latency: Duration) {
        if pinned {
            self.pinned_immediate_dispatch_latency.record(latency);
        } else {
            self.general_immediate_dispatch_latency.record(latency);
        }
    }

    /// Records one request that awaited an assignment's arrival and was then served it.
    fn record_waited_dispatch(&self, pinned: bool, latency: Duration) {
        if pinned {
            self.pinned_waited_dispatch_latency.record(latency);
        } else {
            self.general_waited_dispatch_latency.record(latency);
        }
    }

    /// Records one request that was served no assignment.
    fn record_empty_response(&self, pinned: bool) {
        if pinned {
            self.num_pinned_empty_responses
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.num_general_empty_responses
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records one completed task of `job_id`.
    ///
    /// `resource_group_id` is only consulted when the job has no entry yet, which a well-formed run
    /// never hits: a task is emitted, and so its job first seen, before it can be completed.
    fn record_completion(&self, job_id: JobId, resource_group_id: ResourceGroupId) {
        let nanos = self.elapsed_nanos();
        // The read guard is taken first because it lets completions of jobs in the same shard
        // proceed concurrently, which the entry API's write guard would serialize.
        if let Some(progress) = self.job_progress.get(&job_id) {
            progress.record_completion(nanos);
        } else {
            self.job_progress
                .entry(job_id)
                .or_insert_with(|| JobProgress::new(resource_group_id, nanos))
                .record_completion(nanos);
        }
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
    }
}

/// The adapter translating `NextTask` requests into [`DispatchService`] calls.
struct Adapter {
    dispatch_service: DispatchService,
    storage: FakeStorage,
    instrumentation: Option<Arc<BenchInstrumentation>>,
}

impl Adapter {
    /// Reports the assignment a worker carried back to storage, if it carried one.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`Status::invalid_argument`] if the completed assignment carries no task ID or one that
    ///   has no core representation.
    fn report_completed(&self, request: &NextTaskRequest) -> Result<(), Status> {
        let Some(completed) = request.completed else {
            return Ok(());
        };
        let wire_task_id = completed
            .task_id
            .ok_or_else(|| Status::invalid_argument("completed assignment has no task ID"))?;
        let task_id = TaskId::try_from(wire_task_id)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let job_id = JobId::from(completed.job_id);
        if let Some(instrumentation) = self.instrumentation.as_ref() {
            let resource_group_id = ResourceGroupId::from(
                request
                    .resource_group_id
                    .unwrap_or(UNKNOWN_RESOURCE_GROUP_ID),
            );
            instrumentation.record_completion(job_id, resource_group_id);
        }
        self.storage.complete_task(job_id, task_id);
        Ok(())
    }
}

#[tonic::async_trait]
impl PrototypeSchedulerService for Adapter {
    async fn next_task(
        &self,
        request: Request<NextTaskRequest>,
    ) -> Result<Response<NextTaskResponse>, Status> {
        let request = request.into_inner();
        self.report_completed(&request)?;

        let wait_time = Duration::from_millis(request.wait_time_ms);
        let pinned = request.resource_group_id.is_some();
        let dispatch_start = self.instrumentation.as_ref().map(|_| Instant::now());
        let outcome = if let Some(rg_id) = request.resource_group_id {
            self.dispatch_service
                .next_task_pinned_classified(ResourceGroupId::from(rg_id), wait_time)
                .await
        } else {
            self.dispatch_service
                .next_task_general_classified(wait_time)
                .await
        };

        if let (Some(instrumentation), Some(dispatch_start)) =
            (self.instrumentation.as_ref(), dispatch_start)
        {
            match outcome {
                DispatchOutcome::Immediate(_) => {
                    instrumentation.record_immediate_dispatch(pinned, dispatch_start.elapsed());
                }
                DispatchOutcome::Waited(_) => {
                    instrumentation.record_waited_dispatch(pinned, dispatch_start.elapsed());
                }
                DispatchOutcome::Empty => instrumentation.record_empty_response(pinned),
            }
        }

        let dispatch_class = match outcome {
            DispatchOutcome::Immediate(_) => DispatchClass::Immediate,
            DispatchOutcome::Waited(_) => DispatchClass::Waited,
            DispatchOutcome::Empty => DispatchClass::Empty,
        };
        let result = outcome.into_assignment().map_or_else(
            || next_task_response::Result::NoTask(Void {}),
            |assignment| next_task_response::Result::Assignment(Assignment::from(assignment)),
        );
        Ok(Response::new(NextTaskResponse {
            result: Some(result),
            dispatch_class: dispatch_class.into(),
        }))
    }
}

/// The adapter answering the run-status the benchmark's workers poll.
struct BenchAdapter {
    instrumentation: Option<Arc<BenchInstrumentation>>,
}

#[tonic::async_trait]
impl PrototypeBenchService for BenchAdapter {
    async fn run_status(
        &self,
        _request: Request<RunStatusRequest>,
    ) -> Result<Response<RunStatusResponse>, Status> {
        let instrumentation = self.instrumentation.as_ref().ok_or_else(|| {
            Status::unavailable("the server was not started with benchmark instrumentation")
        })?;

        Ok(Response::new(RunStatusResponse {
            drained: instrumentation.is_drained(),
            tasks_completed: instrumentation.tasks_completed(),
            total_tasks: instrumentation.total_tasks(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use std::time::Instant;

    use crate::bench::results::collect_job_records;
    use crate::harness::grpc_service::BenchInstrumentation;
    use crate::types::JobId;
    use crate::types::ResourceGroupId;

    #[test]
    fn dispatch_latency_is_recorded_per_path_and_class_and_excludes_empty_responses() {
        let instrumentation = BenchInstrumentation::new(Instant::now(), 0);
        instrumentation.record_immediate_dispatch(true, Duration::from_micros(3));
        instrumentation.record_immediate_dispatch(true, Duration::from_micros(5));
        instrumentation.record_waited_dispatch(true, Duration::from_micros(900));
        instrumentation.record_immediate_dispatch(false, Duration::from_micros(7));
        instrumentation.record_waited_dispatch(false, Duration::from_micros(800));
        instrumentation.record_waited_dispatch(false, Duration::from_micros(700));
        instrumentation.record_empty_response(true);
        instrumentation.record_empty_response(false);
        instrumentation.record_empty_response(false);

        assert_eq!(
            instrumentation.pinned_immediate_dispatch_latency().count(),
            2
        );
        assert_eq!(instrumentation.pinned_waited_dispatch_latency().count(), 1);
        assert_eq!(
            instrumentation.general_immediate_dispatch_latency().count(),
            1
        );
        assert_eq!(instrumentation.general_waited_dispatch_latency().count(), 2);
        assert_eq!(instrumentation.num_pinned_empty_responses(), 1);
        assert_eq!(instrumentation.num_general_empty_responses(), 2);
        assert_eq!(instrumentation.num_empty_responses(), 3);
    }

    #[test]
    fn a_completion_report_advances_its_job_and_the_run_status() {
        let instrumentation = BenchInstrumentation::new(Instant::now(), 3);
        let job_id = JobId::from(7);
        let rg_id = ResourceGroupId::from(2);
        instrumentation.note_job_first_seen(job_id, rg_id);
        assert!(!instrumentation.is_drained());

        for _ in 0..3 {
            instrumentation.record_completion(job_id, rg_id);
        }

        assert_eq!(instrumentation.tasks_completed(), 3);
        assert!(instrumentation.is_drained());

        let records = collect_job_records(instrumentation.job_progress());
        assert_eq!(records.len(), 1);
        let record = records[0];
        assert_eq!(record.job_id, 7);
        assert_eq!(record.resource_group_id, 2);
        assert_eq!(record.tasks_completed, 3);
        assert!(record.last_completed_nanos >= record.first_seen_nanos);
    }

    #[test]
    fn a_job_keeps_the_earliest_time_it_was_seen() {
        let instrumentation = BenchInstrumentation::new(Instant::now(), 0);
        let job_id = JobId::from(1);
        let rg_id = ResourceGroupId::from(0);

        instrumentation.note_job_first_seen(job_id, rg_id);
        let first_seen_nanos = collect_job_records(instrumentation.job_progress())
            .first()
            .copied()
            .expect("the job was just seen")
            .first_seen_nanos;
        instrumentation.note_job_first_seen(job_id, rg_id);

        let records = collect_job_records(instrumentation.job_progress());
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].first_seen_nanos, first_seen_nanos);
    }

    #[test]
    fn a_completion_of_an_unseen_job_still_creates_its_record() {
        let instrumentation = BenchInstrumentation::new(Instant::now(), 1);
        instrumentation.record_completion(JobId::from(4), ResourceGroupId::from(5));

        let records = collect_job_records(instrumentation.job_progress());
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].job_id, 4);
        assert_eq!(records[0].resource_group_id, 5);
        assert_eq!(records[0].tasks_completed, 1);
    }
}
