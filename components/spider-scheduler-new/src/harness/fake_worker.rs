//! A gRPC client that emulates an execution manager pulling and running assignments.
//!
//! A single worker is the unit a correctness scenario drives; the benchmark drives a pool of them,
//! hundreds of coroutines in one process each holding a gRPC channel of its own and stopping
//! together once the scheduler reports the run drained.

use std::time::Duration;
use std::time::Instant;

use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tonic::transport::Endpoint;

use crate::bench::cases::BenchCase;
use crate::error::HarnessError;
use crate::harness::metrics::DispatchClass;
use crate::harness::metrics::DispatchRecord;
use crate::harness::metrics::WorkerKind;
use crate::harness::metrics::WorkerPoolReport;
use crate::harness::metrics::WorkerSamples;
use crate::proto::CompletedAssignment;
use crate::proto::NextTaskRequest;
use crate::proto::PrototypeSchedulerServiceClient;
use crate::proto::RunStatusRequest;
use crate::proto::TaskId as WireTaskId;
use crate::proto::next_task_response;
use crate::proto::prototype_bench_service_client::PrototypeBenchServiceClient;
use crate::types::ResourceGroupId;
use crate::types::TaskAssignment;

#[cfg(test)]
mod tests;

/// How often a [`WorkerPool`] asks the scheduler whether the run has drained, in milliseconds.
pub const DEFAULT_RUN_STATUS_POLL_INTERVAL_MS: u64 = 10;

/// The long-poll wait a benchmark worker asks the scheduler to hold a request for, in
/// milliseconds.
pub const DEFAULT_NEXT_TASK_WAIT_MS: u64 = 50;

/// The number of samples a worker reserves room for before issuing its first request, so that the
/// measured path never allocates.
const SAMPLE_CAPACITY_HINT: usize = 4096;

/// The headroom added to a benchmark worker's expected assignment count when its sample buffer is
/// reserved, covering the workers that run ahead of their even share.
const SAMPLE_CAPACITY_SLACK: usize = 1024;

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
        let mut client = connect(&endpoint).await?;
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

/// The description of every worker one benchmark process hosts.
#[derive(Clone, Debug)]
pub struct WorkerPoolConfig {
    /// The scheduler the workers pull from.
    pub endpoint: String,

    /// The number of resource groups the pinned workers are spread over.
    pub num_resource_groups: usize,

    /// The number of workers pinned to each resource group.
    pub dedicated_workers_per_group: usize,

    /// The number of workers that accept assignments from any resource group.
    pub shared_workers: usize,

    /// How long a worker pretends to execute a received assignment.
    pub task_duration_ms: u64,

    /// The long-poll wait a worker asks the scheduler to hold a request for.
    pub next_task_wait_ms: u64,

    /// The number of gRPC channels the workers spread over, which is one apiece unless a run
    /// deliberately narrows it.
    ///
    /// A channel is one HTTP/2 connection, and coroutines sharing one queue behind each other on
    /// it. That queueing is invisible to the server and lands entirely in the client-side series
    /// the contract reads transport cost from: three case-A runs differing only in this figure put
    /// p90 client latency at 487 us with one channel per worker and at 6.75 ms with both 2.5 and
    /// ten workers per channel, while server-side latency stayed at roughly one microsecond
    /// throughout.
    pub channel_pool_size: usize,

    /// The first execution manager ID the pool hands out, so that the workers of a run split over
    /// several processes never collide on an identity.
    pub execution_manager_id_base: u64,

    /// The number of latencies each worker reserves room for before its first request.
    pub sample_capacity_hint: usize,

    /// How often the pool asks the scheduler whether the run has drained.
    pub run_status_poll_interval_ms: u64,

    /// The longest the pool waits for the run to drain, or `0` to wait indefinitely.
    pub run_timeout_ms: u64,
}

impl WorkerPoolConfig {
    /// Factory function.
    ///
    /// Derives the pool from `case`, giving this process the case's entire worker set. A run that
    /// splits the workers over several processes overrides the counts afterwards; `worker_index`
    /// separates the processes' execution manager identities either way. Such a run has to narrow
    /// the channel pool along with the counts to keep a channel per worker.
    ///
    /// # Returns
    ///
    /// A newly created configuration with one gRPC channel per worker, whose poll cadence and
    /// long-poll wait are the module's defaults and which waits indefinitely for the drain.
    #[must_use]
    pub fn from_case(case: &BenchCase, endpoint: String, worker_index: usize) -> Self {
        let num_workers = case.total_workers();
        let assignments_per_worker = case.total_tasks().div_ceil(num_workers.max(1));

        Self {
            endpoint,
            num_resource_groups: case.num_resource_groups,
            dedicated_workers_per_group: case.dedicated_workers_per_group,
            shared_workers: case.shared_workers,
            task_duration_ms: case.task_duration_ms,
            next_task_wait_ms: DEFAULT_NEXT_TASK_WAIT_MS,
            channel_pool_size: num_workers,
            execution_manager_id_base: u64::try_from(worker_index.saturating_mul(num_workers))
                .unwrap_or(u64::MAX),
            sample_capacity_hint: assignments_per_worker.saturating_add(SAMPLE_CAPACITY_SLACK),
            run_status_poll_interval_ms: DEFAULT_RUN_STATUS_POLL_INTERVAL_MS,
            run_timeout_ms: 0,
        }
    }

    /// # Returns
    ///
    /// The number of workers the pool hosts, pinned and general together.
    #[must_use]
    pub const fn num_workers(&self) -> usize {
        self.shared_workers + self.num_resource_groups * self.dedicated_workers_per_group
    }

    /// Lays the pool's workers out, the pinned ones grouped by resource group and the general ones
    /// last, over a contiguous block of execution manager IDs starting at
    /// [`Self::execution_manager_id_base`].
    ///
    /// # Returns
    ///
    /// One configuration per worker.
    #[must_use]
    pub fn worker_configs(&self) -> Vec<FakeWorkerConfig> {
        let mut configs = Vec::with_capacity(self.num_workers());
        for group_index in 0..self.num_resource_groups {
            let resource_group_id =
                ResourceGroupId::from(u64::try_from(group_index).unwrap_or(u64::MAX));
            for _ in 0..self.dedicated_workers_per_group {
                configs.push(self.worker_config(configs.len(), Some(resource_group_id)));
            }
        }
        for _ in 0..self.shared_workers {
            configs.push(self.worker_config(configs.len(), None));
        }

        configs
    }

    /// # Returns
    ///
    /// The configuration of the worker at `worker_offset` in the pool's identity block.
    fn worker_config(
        &self,
        worker_offset: usize,
        resource_group_id: Option<ResourceGroupId>,
    ) -> FakeWorkerConfig {
        FakeWorkerConfig {
            execution_manager_id: self
                .execution_manager_id_base
                .saturating_add(u64::try_from(worker_offset).unwrap_or(u64::MAX)),
            resource_group_id,
            task_duration_ms: self.task_duration_ms,
            next_task_wait_ms: self.next_task_wait_ms,
        }
    }
}

/// A process' worth of emulated execution managers, run as coroutines over a pool of gRPC channels
/// that holds one channel per worker unless the run narrows it.
///
/// The pool is the benchmark's client side. Its workers measure only the requests that returned an
/// assignment, each into a buffer it alone owns and split by the class the scheduler served the
/// request in, and they keep asking for work until the scheduler reports the run drained -- which
/// is what delivers the completions of the last tasks, since a worker reports an assignment
/// complete on the request that follows it.
pub struct WorkerPool;

impl WorkerPool {
    /// Runs every worker the configuration describes until the scheduler reports the run drained,
    /// then merges what they measured.
    ///
    /// # Returns
    ///
    /// The pool's merged report on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`HarnessError::Internal`] if a worker coroutine could not be joined.
    /// * Forwards [`ChannelPool::connect`]'s return values on failure.
    /// * Forwards [`connect_channel`]'s return values on failure.
    /// * Forwards [`run_bench_worker`]'s return values on failure.
    pub async fn run(config: WorkerPoolConfig) -> Result<WorkerPoolReport, HarnessError> {
        let channels = ChannelPool::connect(&config.endpoint, config.channel_pool_size).await?;
        // The status poller gets a connection of its own so that its traffic never shares a
        // connection with a request the benchmark is timing.
        let status_channel = connect_channel(&config.endpoint).await?;
        let cancellation_token = CancellationToken::new();

        let run_timeout = if 0 == config.run_timeout_ms {
            None
        } else {
            Some(Duration::from_millis(config.run_timeout_ms))
        };
        let status_task = tokio::spawn(watch_run_status(
            status_channel,
            Duration::from_millis(config.run_status_poll_interval_ms),
            run_timeout,
            cancellation_token.clone(),
        ));

        let worker_configs = config.worker_configs();
        let num_workers = worker_configs.len();
        let mut worker_tasks = JoinSet::new();
        for (worker_index, worker_config) in worker_configs.into_iter().enumerate() {
            worker_tasks.spawn(run_bench_worker(
                worker_config,
                channels.channel(worker_index),
                config.sample_capacity_hint,
                cancellation_token.clone(),
            ));
        }

        let mut samples = Vec::with_capacity(num_workers);
        let mut first_error = None;
        while let Some(joined) = worker_tasks.join_next().await {
            let error = match joined {
                Ok(Ok(worker_samples)) => {
                    samples.push(worker_samples);
                    continue;
                }
                Ok(Err(error)) => error,
                Err(join_error) => HarnessError::Internal(join_error.to_string()),
            };
            tracing::error!(error = % error, "A benchmark worker failed.");
            // The remaining workers only stop on the shared token, so a failure has to end the run
            // rather than leaving them to drain a workload the report will not describe.
            cancellation_token.cancel();
            first_error.get_or_insert(error);
        }

        let drained = match status_task.await {
            Ok(drained) => drained,
            Err(join_error) => {
                tracing::error!(
                    error = % join_error,
                    "The benchmark run status poller could not be joined."
                );
                false
            }
        };
        if let Some(error) = first_error {
            return Err(error);
        }

        Ok(WorkerPoolReport::from_samples(
            channels.len(),
            drained,
            &samples,
        ))
    }
}

/// The gRPC channels a [`WorkerPool`]'s workers pull over.
struct ChannelPool {
    channels: Vec<Channel>,
}

impl ChannelPool {
    /// Opens `pool_size` connections to `endpoint`, or one if `pool_size` is zero.
    ///
    /// # Returns
    ///
    /// A newly created pool of connected channels on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`connect_channel`]'s return values on failure.
    async fn connect(endpoint: &str, pool_size: usize) -> Result<Self, HarnessError> {
        let pool_size = pool_size.max(1);
        let mut channels = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            channels.push(connect_channel(endpoint).await?);
        }

        Ok(Self { channels })
    }

    /// # Returns
    ///
    /// The number of channels in the pool.
    const fn len(&self) -> usize {
        self.channels.len()
    }

    /// # Returns
    ///
    /// The channel the worker at `worker_index` uses, the workers being spread evenly over the
    /// pool and so holding one apiece whenever the pool is as large as the worker set.
    fn channel(&self, worker_index: usize) -> Channel {
        self.channels[worker_index % self.channels.len()].clone()
    }
}

/// Pulls assignments for one benchmark worker until `cancellation_token` fires.
///
/// The measured window is the gRPC call alone: the request is built and the simulated execution is
/// slept for outside it, and only a response that carried an assignment is timed at all. A timed
/// sample is filed under the class the scheduler reports it served the request in, so that the
/// client-side and server-side series of a run cover the same population.
///
/// # Returns
///
/// What the worker measured on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Transport`] if a request fails while the run is still going.
/// * [`HarnessError::InvalidMessage`] if an assignment arrives without a class to file it under.
async fn run_bench_worker(
    config: FakeWorkerConfig,
    channel: Channel,
    sample_capacity_hint: usize,
    cancellation_token: CancellationToken,
) -> Result<WorkerSamples, HarnessError> {
    let mut client = PrototypeSchedulerServiceClient::new(channel);
    let mut samples = WorkerSamples::with_capacity(
        WorkerKind::from_resource_group_id(config.resource_group_id),
        sample_capacity_hint,
    );
    let resource_group_id = config.resource_group_id.map(|rg_id| rg_id.get());
    let task_duration = Duration::from_millis(config.task_duration_ms);
    let mut completed: Option<CompletedAssignment> = None;

    while !cancellation_token.is_cancelled() {
        let request = NextTaskRequest {
            execution_manager_id: config.execution_manager_id,
            resource_group_id,
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
            // A finished run tears the scheduler down underneath in-flight requests, so a failure
            // there ends the worker rather than failing it.
            Err(status) => {
                if cancellation_token.is_cancelled() {
                    break;
                }
                return Err(HarnessError::Transport(status.to_string()));
            }
        };

        let dispatch_class = DispatchClass::from_wire(response.dispatch_class());
        let Some(next_task_response::Result::Assignment(assignment)) = response.result else {
            samples.num_empty_responses = samples.num_empty_responses.saturating_add(1);
            continue;
        };
        let Some(dispatch_class) = dispatch_class else {
            return Err(HarnessError::InvalidMessage(
                "an assignment arrived without a dispatch class to file its latency under"
                    .to_owned(),
            ));
        };
        samples.record(dispatch_class, latency);
        completed = Some(CompletedAssignment {
            job_id: assignment.job_id,
            task_id: assignment.task_id,
        });

        tokio::time::sleep(task_duration).await;
    }

    Ok(samples)
}

/// Polls the scheduler's run status until it reports the run drained, then cancels
/// `cancellation_token` so that every worker of the pool stops together.
///
/// # Returns
///
/// Whether the scheduler reported the run drained.
async fn watch_run_status(
    channel: Channel,
    poll_interval: Duration,
    run_timeout: Option<Duration>,
    cancellation_token: CancellationToken,
) -> bool {
    let drained = poll_run_status(channel, poll_interval, run_timeout, &cancellation_token).await;
    cancellation_token.cancel();

    drained
}

/// Asks the scheduler for the run's status every `poll_interval` until it reports the run drained,
/// `run_timeout` elapses, `cancellation_token` fires, or the scheduler stops answering.
///
/// # Returns
///
/// Whether the scheduler reported the run drained.
async fn poll_run_status(
    channel: Channel,
    poll_interval: Duration,
    run_timeout: Option<Duration>,
    cancellation_token: &CancellationToken,
) -> bool {
    let mut client = PrototypeBenchServiceClient::new(channel);
    let start = Instant::now();
    loop {
        match client.run_status(RunStatusRequest {}).await {
            Ok(response) => {
                if response.into_inner().drained {
                    return true;
                }
            }
            Err(status) => {
                tracing::error!(
                    error = % status,
                    "The scheduler stopped answering run status; ending the run undrained."
                );
                return false;
            }
        }
        if let Some(timeout) = run_timeout
            && start.elapsed() >= timeout
        {
            tracing::warn!(
                timeout_ms = timeout.as_millis(),
                "The benchmark run did not drain before its timeout."
            );
            return false;
        }

        tokio::select! {
            biased;
            () = cancellation_token.cancelled() => return false,
            () = tokio::time::sleep(poll_interval) => {}
        }
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
/// * Forwards [`connect_channel`]'s return values on failure.
async fn connect(endpoint: &str) -> Result<PrototypeSchedulerServiceClient<Channel>, HarnessError> {
    Ok(PrototypeSchedulerServiceClient::new(
        connect_channel(endpoint).await?,
    ))
}

/// Opens one gRPC channel to the scheduler.
///
/// # Returns
///
/// A connected channel on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Config`] if `endpoint` is not a valid URI.
/// * [`HarnessError::Transport`] if the connection cannot be established.
async fn connect_channel(endpoint: &str) -> Result<Channel, HarnessError> {
    // A bare `host:port` is not a URI, so it has to be given a scheme before tonic will parse it.
    let uri = if endpoint.contains("://") {
        endpoint.to_owned()
    } else {
        format!("http://{endpoint}")
    };

    Endpoint::try_from(uri)
        .map_err(|error| HarnessError::Config(error.to_string()))?
        .connect()
        .await
        .map_err(|error| HarnessError::Transport(error.to_string()))
}
