//! The in-process test harness that drives the prototype scheduler end to end.
//!
//! The harness assembles a complete prototype stack -- the core on its own thread, the dispatch
//! structures the core and the service share, a gRPC server, and a set of gRPC workers -- so that a
//! scenario exercises exactly the path a deployed execution manager takes. Nothing here reaches
//! into the core's state or short-circuits the dispatch service, which is what makes the same
//! harness usable for both correctness scenarios and the performance evaluation.

use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use tokio::sync::mpsc::unbounded_channel;
use tokio_util::sync::CancellationToken;

use crate::config::CoreConfig;
use crate::core::Core;
use crate::core::run_core_on_dedicated_thread;
use crate::dispatch_queue::DispatchService;
use crate::dispatch_queue::GlobalDispatchQueue;
use crate::error::CoreError;
use crate::error::HarnessError;
use crate::resource_group::ResourceGroupTable;
use crate::session::SessionManager;
use crate::storage_client::SchedulerStorageClient;
use crate::types::TaskAssignment;

pub mod fake_storage;
pub mod fake_worker;
pub mod grpc_service;
pub mod metrics;

pub use crate::harness::fake_storage::FakeStorage;
pub use crate::harness::fake_storage::FakeStorageConfig;
pub use crate::harness::fake_storage::FirstSeenRecorder;
pub use crate::harness::fake_storage::ReleaseConfig;
pub use crate::harness::fake_worker::FakeWorker;
pub use crate::harness::fake_worker::FakeWorkerConfig;
pub use crate::harness::fake_worker::WorkerPool;
pub use crate::harness::fake_worker::WorkerPoolConfig;
pub use crate::harness::fake_worker::WorkerReport;
pub use crate::harness::grpc_service::BenchInstrumentation;
pub use crate::harness::grpc_service::HarnessServer;
pub use crate::harness::metrics::DispatchClass;
pub use crate::harness::metrics::DispatchRecord;
pub use crate::harness::metrics::LatencySamples;
pub use crate::harness::metrics::WorkerKind;
pub use crate::harness::metrics::WorkerPoolReport;
pub use crate::harness::metrics::WorkerSamples;

/// Statically asserts that a core and the future its loop produces are both `Send`, so that any
/// runtime can spawn the core and no later change can silently make it thread-bound again.
///
/// The assertion lives here rather than beside [`Core`] because it has to name a concrete storage
/// client, and [`FakeStorage`] is the only one the crate has; the core must not depend on its own
/// harness to state a property of itself.
const _: () = {
    const fn assert_send<SendType: Send>() {}

    const fn assert_returns_send_future<
        StorageClientType: SchedulerStorageClient,
        FutureType: Send,
        RunType: FnOnce(Core<StorageClientType>) -> FutureType + Copy,
    >(
        _run: RunType,
    ) {
    }

    assert_send::<Core<FakeStorage>>();
    assert_returns_send_future(Core::<FakeStorage>::run);
};

/// How often a run re-checks whether the workload has drained.
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// The full description of one harness run.
#[derive(Clone, Debug)]
pub struct HarnessConfig {
    /// The tunables the prototype core runs under.
    pub core: CoreConfig,

    /// The workload the fake storage serves.
    pub storage: FakeStorageConfig,

    /// The execution managers that drain the workload.
    pub workers: Vec<FakeWorkerConfig>,
}

/// A running prototype stack: the core, the shared dispatch structures, and a gRPC server in front
/// of them.
///
/// The workers described by the configuration are not started until [`Self::run_until_drained`] is
/// called, so a scenario may inspect or perturb the stack first.
#[derive(Debug)]
pub struct Harness {
    worker_configs: Vec<FakeWorkerConfig>,
    storage: FakeStorage,
    server: HarnessServer,
    core_cancellation_token: CancellationToken,
    core_thread: JoinHandle<Result<(), CoreError>>,
}

impl Harness {
    /// Factory function.
    ///
    /// Builds the shared dispatch structures, starts the core on its own thread, and brings up a
    /// gRPC server on an ephemeral loopback port.
    ///
    /// # Returns
    ///
    /// A newly created harness with its core already ticking and its server already accepting
    /// connections, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`HarnessServer::start`]'s return values on failure.
    pub async fn start(config: HarnessConfig) -> Result<Self, HarnessError> {
        let rg_table = ResourceGroupTable::new();
        let global_queue = GlobalDispatchQueue::new();
        let session_manager = SessionManager::default();
        let storage = FakeStorage::new(config.storage);
        // No harness scenario replays a lost assignment, so the write side is dropped straight
        // away; the core reads the resulting closed queue exactly as it reads an empty one.
        let (_, reschedule_queue_reader) = unbounded_channel::<TaskAssignment>();
        let core_cancellation_token = CancellationToken::new();

        let core_thread = run_core_on_dedicated_thread(Core::new(
            config.core,
            storage.clone(),
            rg_table.clone(),
            global_queue.clone(),
            session_manager.clone(),
            reschedule_queue_reader,
            core_cancellation_token.clone(),
        ));

        let dispatch_service = DispatchService::new(rg_table, global_queue, session_manager);
        let server = match HarnessServer::start(dispatch_service, storage.clone()).await {
            Ok(server) => server,
            Err(error) => {
                core_cancellation_token.cancel();
                return Err(error);
            }
        };

        Ok(Self {
            worker_configs: config.workers,
            storage,
            server,
            core_cancellation_token,
            core_thread,
        })
    }

    /// # Returns
    ///
    /// The storage the core polls, shared with the running stack.
    #[must_use]
    pub fn storage(&self) -> FakeStorage {
        self.storage.clone()
    }

    /// Runs every configured worker until the workload drains or `timeout` expires, then tears the
    /// whole stack down.
    ///
    /// A run that times out is not an error: it returns everything the workers managed to collect,
    /// so a scenario can report exactly which tasks were never dispatched instead of losing that
    /// evidence to a failure.
    ///
    /// # Returns
    ///
    /// The workers' reports and the storage they drained, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`HarnessError::Internal`] if a worker task could not be joined.
    /// * Forwards [`join_core`]'s return values on failure.
    /// * Forwards [`FakeWorker::run`]'s return values on failure.
    pub async fn run_until_drained(
        self,
        timeout: Duration,
    ) -> Result<HarnessOutcome, HarnessError> {
        let endpoint = self.server.endpoint();
        let worker_cancellation_token = CancellationToken::new();
        let run_start = Instant::now();
        let mut worker_tasks = Vec::with_capacity(self.worker_configs.len());
        for worker_config in self.worker_configs {
            worker_tasks.push(tokio::spawn(FakeWorker::run(
                worker_config,
                endpoint.clone(),
                run_start,
                worker_cancellation_token.clone(),
            )));
        }

        let drained = wait_until_drained(&self.storage, timeout).await;
        worker_cancellation_token.cancel();

        let mut reports = Vec::with_capacity(worker_tasks.len());
        let mut first_error = None;
        for worker_task in worker_tasks {
            match worker_task.await {
                Ok(Ok(report)) => reports.push(report),
                Ok(Err(error)) => {
                    tracing::error!(error = % error, "A harness worker failed.");
                    first_error.get_or_insert(error);
                }
                Err(join_error) => {
                    let error = HarnessError::Internal(join_error.to_string());
                    tracing::error!(error = % error, "A harness worker could not be joined.");
                    first_error.get_or_insert(error);
                }
            }
        }

        self.server.shutdown().await;
        self.core_cancellation_token.cancel();
        let core_result = join_core(self.core_thread).await;

        if !drained {
            tracing::warn!(
                timeout_ms = timeout.as_millis(),
                num_dispatches = reports
                    .iter()
                    .map(|report| report.dispatches.len())
                    .sum::<usize>(),
                "The harness run timed out before the workload drained."
            );
        }
        core_result?;
        if let Some(error) = first_error {
            return Err(error);
        }

        Ok(HarnessOutcome {
            reports,
            storage: self.storage,
        })
    }
}

/// Everything one harness run produced.
#[derive(Debug)]
pub struct HarnessOutcome {
    /// One report per configured worker, in configuration order.
    pub reports: Vec<WorkerReport>,

    /// The storage the run drained, carrying the workload's completion state.
    pub storage: FakeStorage,
}

/// Waits until every task of the workload has been reported complete or `timeout` expires.
///
/// # Returns
///
/// Whether the workload drained before `timeout` expired.
async fn wait_until_drained(storage: &FakeStorage, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if storage.is_drained() {
            return true;
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return false;
        }
        tokio::time::sleep(DRAIN_POLL_INTERVAL.min(remaining)).await;
    }
}

/// Waits for the core's thread to exit.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Internal`] if the blocking join task could not be joined, if the core's thread
///   panicked, or if the core exited on an error.
async fn join_core(core_thread: JoinHandle<Result<(), CoreError>>) -> Result<(), HarnessError> {
    let join_result = tokio::task::spawn_blocking(move || core_thread.join())
        .await
        .map_err(|error| HarnessError::Internal(error.to_string()))?;
    match join_result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(HarnessError::Internal(error.to_string())),
        Err(_) => Err(HarnessError::Internal(
            "the scheduler core thread panicked".to_owned(),
        )),
    }
}
