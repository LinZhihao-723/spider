//! A gRPC client that emulates an execution manager pulling and running assignments.

use std::time::Duration;
use std::time::Instant;

use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tonic::transport::Endpoint;

use crate::error::HarnessError;
use crate::harness::metrics::DispatchRecord;
use crate::proto::CompletedAssignment;
use crate::proto::NextTaskRequest;
use crate::proto::PrototypeSchedulerServiceClient;
use crate::proto::TaskId as WireTaskId;
use crate::proto::next_task_response;
use crate::types::ResourceGroupId;
use crate::types::TaskAssignment;

/// The number of samples a worker reserves room for before issuing its first request, so that the
/// measured path never allocates.
const SAMPLE_CAPACITY_HINT: usize = 4096;

/// The configuration of a single emulated execution manager.
#[derive(Clone, Debug)]
pub struct FakeWorkerConfig {
    /// The identity the worker reports to the scheduler.
    pub execution_manager_id: u64,

    /// `None` for a general execution manager.
    pub resource_group_id: Option<ResourceGroupId>,

    /// How long the worker pretends to execute a received assignment.
    pub task_duration_ms: u64,

    /// The long-poll wait the worker asks the scheduler to hold a request for.
    pub next_task_wait_ms: u64,
}

/// Everything one worker observed over a run.
#[derive(Debug, Default)]
pub struct WorkerReport {
    /// The worker's identity.
    pub execution_manager_id: u64,

    /// The resource group the worker was pinned to, if any.
    pub resource_group_id: Option<ResourceGroupId>,

    /// The assignments the worker received, in the order it received them.
    pub dispatches: Vec<DispatchRecord>,

    /// Responses that carried no assignment.
    pub num_empty_responses: usize,

    /// The latency of every completed request, whether or not it carried an assignment.
    pub latencies: Vec<Duration>,
}

/// An emulated execution manager that drains the scheduler over gRPC.
pub struct FakeWorker;

impl FakeWorker {
    /// Pulls assignments from the scheduler until `cancellation_token` fires.
    ///
    /// Each iteration issues one `NextTask` request, reports the previously received assignment as
    /// completed, and sleeps for the configured task duration whenever an assignment came back.
    ///
    /// Every dispatch record is timestamped relative to `run_start`, so records taken from
    /// different workers of the same run share one origin and are directly comparable.
    ///
    /// # Returns
    ///
    /// The worker's report on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`HarnessError::Transport`] if a request fails while the worker is still running.
    /// * Forwards [`connect`]'s return values on failure.
    /// * Forwards [`TaskAssignment::try_from`]'s return values on failure.
    pub async fn run(
        config: FakeWorkerConfig,
        endpoint: String,
        run_start: Instant,
        cancellation_token: CancellationToken,
    ) -> Result<WorkerReport, HarnessError> {
        let mut client = connect(endpoint).await?;
        let mut report = WorkerReport {
            execution_manager_id: config.execution_manager_id,
            resource_group_id: config.resource_group_id,
            dispatches: Vec::with_capacity(SAMPLE_CAPACITY_HINT),
            num_empty_responses: 0,
            latencies: Vec::with_capacity(SAMPLE_CAPACITY_HINT),
        };
        let task_duration = Duration::from_millis(config.task_duration_ms);
        let mut completed: Option<CompletedAssignment> = None;

        while !cancellation_token.is_cancelled() {
            let request = NextTaskRequest {
                execution_manager_id: config.execution_manager_id,
                resource_group_id: config.resource_group_id.map(|rg_id| rg_id.get()),
                wait_time_ms: config.next_task_wait_ms,
                completed: completed.take(),
            };

            let start = Instant::now();
            let response = tokio::select! {
                biased;
                () = cancellation_token.cancelled() => break,
                response = client.next_task(request) => response,
            };
            let latency = start.elapsed();

            let response = match response {
                Ok(response) => response.into_inner(),
                // A cancelled run tears the server down underneath in-flight requests, so a
                // failure there ends the run rather than failing it.
                Err(status) => {
                    if cancellation_token.is_cancelled() {
                        break;
                    }
                    return Err(HarnessError::Transport(status.to_string()));
                }
            };
            report.latencies.push(latency);

            let Some(next_task_response::Result::Assignment(assignment)) = response.result else {
                report.num_empty_responses += 1;
                continue;
            };
            let assignment = TaskAssignment::try_from(assignment)?;
            completed = Some(CompletedAssignment {
                job_id: assignment.job_id.get(),
                task_id: Some(WireTaskId::from(assignment.task_id)),
            });

            tokio::time::sleep(task_duration).await;
            report.dispatches.push(DispatchRecord {
                assignment_id: assignment.id,
                resource_group_id: assignment.resource_group_id,
                job_id: assignment.job_id,
                task_id: assignment.task_id,
                latency,
                completed_at: run_start.elapsed(),
            });
        }

        Ok(report)
    }
}

/// Connects one gRPC client to the harness server.
///
/// # Returns
///
/// A connected client on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Config`] if `endpoint` is not a valid URI.
/// * [`HarnessError::Transport`] if the connection cannot be established.
async fn connect(
    endpoint: String,
) -> Result<PrototypeSchedulerServiceClient<Channel>, HarnessError> {
    // A bare `host:port` is not a URI, so it has to be given a scheme before tonic will parse it.
    let uri = if endpoint.contains("://") {
        endpoint
    } else {
        format!("http://{endpoint}")
    };
    let channel = Endpoint::try_from(uri)
        .map_err(|error| HarnessError::Config(error.to_string()))?
        .connect()
        .await
        .map_err(|error| HarnessError::Transport(error.to_string()))?;

    Ok(PrototypeSchedulerServiceClient::new(channel))
}
