//! The scheduler side of the prototype's performance evaluation.
//!
//! The binary assembles one benchmark case's stack -- the fake inbound queue, the core on its own
//! thread, and the gRPC dispatch and bench services -- from the case table alone, so that no
//! dimension of a run is restated by whatever launches it. It serves until every task of the
//! workload has been reported complete, then writes the case's scheduler results file.
//!
//! The workers of a run live in separate processes, possibly on separate hosts, and stop when the
//! bench service reports the run drained. The server therefore keeps serving for a short while
//! after the drain, so that every worker learns of it through a status poll rather than through a
//! dropped connection. The core keeps ticking over that interval as well: its samples are cheaper
//! to identify and discard afterwards -- they publish no assignment -- than the burst of empty
//! responses a torn-down core's closed queues would provoke.

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use spider_scheduler_new::BenchCase;
use spider_scheduler_new::CoreError;
use spider_scheduler_new::HarnessError;
use spider_scheduler_new::JobProgressMap;
use spider_scheduler_new::LatencySeries;
use spider_scheduler_new::SchedulerResults;
use spider_scheduler_new::TaskAssignment;
use spider_scheduler_new::TickSample;
use spider_scheduler_new::bench::results::build_profile;
use spider_scheduler_new::bench::results::collect_job_records;
use spider_scheduler_new::bench::results::host_num_cores;
use spider_scheduler_new::core::Core;
use spider_scheduler_new::core::run_core_on_dedicated_thread_with_tick_samples;
use spider_scheduler_new::dispatch_queue::DispatchService;
use spider_scheduler_new::dispatch_queue::GlobalDispatchQueue;
use spider_scheduler_new::harness::BenchInstrumentation;
use spider_scheduler_new::harness::FakeStorage;
use spider_scheduler_new::harness::FakeStorageConfig;
use spider_scheduler_new::harness::FirstSeenRecorder;
use spider_scheduler_new::harness::HarnessServer;
use spider_scheduler_new::harness::ReleaseConfig;
use spider_scheduler_new::resource_group::ResourceGroupTable;
use spider_scheduler_new::session::SessionManager;
use tokio::sync::mpsc::unbounded_channel;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

/// The line a launcher waits for before it starts the workers of a run.
const READY_MARKER: &str = "BENCH_SCHEDULER_READY";

/// The log verbosity a run uses when the environment names none.
const DEFAULT_LOG_FILTER: &str = "info";

/// How often the drain condition is re-checked.
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// How often a run that has not drained yet reports its progress.
const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(5);

/// How long the services keep serving after the run drains, in milliseconds.
///
/// The value only has to cover one worker-side status poll and the round trip that answers it, so
/// that no worker mistakes the shutdown for a failed run. It is kept a small multiple of that poll
/// interval because the core goes on ticking meanwhile, and every one of those ticks is a sample
/// the run's own ticks have to be told apart from.
const DEFAULT_LINGER_MS: u64 = 500;

/// The multiple of a case's expected duration a run may take before it is declared hung.
const TIMEOUT_EXPECTED_DURATION_FACTOR: u64 = 10;

/// The headroom a derived run timeout adds for a case's start-up and drain-out, in milliseconds.
const TIMEOUT_HEADROOM_MS: u64 = 120_000;

/// The multiple of a case's expected tick count the core's sample log is reserved for.
const TICK_SAMPLE_SLACK_FACTOR: u64 = 4;

/// The headroom the core's reserved sample log adds, in ticks.
const TICK_SAMPLE_HEADROOM: usize = 4_096;

/// The command line of the benchmark's scheduler process.
#[derive(Debug, Parser)]
#[command(about = "Runs the scheduler side of one prototype scheduler benchmark case")]
struct Args {
    /// The benchmark case to run, named as the case table names it.
    #[arg(long)]
    case: String,

    /// The address the dispatch and bench services bind.
    #[arg(long, default_value = "127.0.0.1:50151")]
    listen: SocketAddr,

    /// The directory the case's results file is written into, under the name the case dictates.
    #[arg(long)]
    output: PathBuf,

    /// How long the run may take before it is declared hung, in seconds. Defaults to ten times the
    /// case's expected duration plus two minutes.
    #[arg(long)]
    timeout_secs: Option<u64>,

    /// How long the services keep serving after the run drains, in milliseconds.
    #[arg(long, default_value_t = DEFAULT_LINGER_MS)]
    linger_ms: u64,
}

/// # Returns
///
/// [`ExitCode::SUCCESS`] if the case drained and its results were written, and
/// [`ExitCode::FAILURE`] otherwise.
#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER)),
        )
        .with_writer(std::io::stderr)
        .init();

    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = % error, "The benchmark scheduler failed.");
            ExitCode::FAILURE
        }
    }
}

/// Runs one benchmark case to completion and writes its results file.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Config`] if no case carries the requested name.
/// * [`HarnessError::Internal`] if the run did not drain before its timeout, or if a job did not
///   complete exactly the case's tasks per job.
/// * Forwards [`HarnessServer::start_bench`]'s return values on failure.
/// * Forwards [`join_core`]'s return values on failure.
/// * Forwards [`write_results`]'s return values on failure.
async fn run(args: Args) -> Result<(), HarnessError> {
    let case = BenchCase::from_name(&args.case)
        .ok_or_else(|| HarnessError::Config(format!("unknown benchmark case: {}", args.case)))?;
    let run_timeout = Duration::from_millis(args.timeout_secs.map_or_else(
        || default_timeout_ms(&case),
        |secs| secs.saturating_mul(1_000),
    ));

    let run_origin = Instant::now();
    let job_progress = Arc::new(JobProgressMap::new());
    let storage = build_storage(&case, Arc::clone(&job_progress), run_origin);
    let total_tasks = u64::try_from(storage.total_tasks()).unwrap_or(u64::MAX);
    let instrumentation = Arc::new(BenchInstrumentation::with_progress(
        run_origin,
        total_tasks,
        job_progress,
    ));

    let rg_table = ResourceGroupTable::new();
    let global_queue = GlobalDispatchQueue::new();
    let session_manager = SessionManager::default();
    let core_cancellation_token = CancellationToken::new();
    let core_thread = spawn_core(
        &case,
        &storage,
        &rg_table,
        &global_queue,
        &session_manager,
        core_cancellation_token.clone(),
    );

    let dispatch_service = DispatchService::new(rg_table, global_queue, session_manager);
    let server = match HarnessServer::start_bench(
        dispatch_service,
        storage.clone(),
        args.listen,
        Arc::clone(&instrumentation),
    )
    .await
    {
        Ok(server) => server,
        Err(error) => {
            core_cancellation_token.cancel();
            return Err(error);
        }
    };

    println!(
        "{READY_MARKER} case={} endpoint={} total_tasks={total_tasks}",
        case.name,
        server.endpoint()
    );
    tracing::info!(
        case = % case.name,
        endpoint = % server.endpoint(),
        total_tasks,
        timeout_ms = run_timeout.as_millis(),
        "Benchmark scheduler ready."
    );

    let drained = wait_until_drained(&instrumentation, run_timeout).await;
    tokio::time::sleep(Duration::from_millis(args.linger_ms)).await;
    core_cancellation_token.cancel();
    server.shutdown().await;
    let tick_samples = join_core(core_thread).await?;

    if !drained {
        return Err(HarnessError::Internal(format!(
            "the run did not drain within {} ms: {} of {total_tasks} tasks completed",
            run_timeout.as_millis(),
            instrumentation.tasks_completed()
        )));
    }

    let results = build_results(case, &instrumentation, tick_samples);
    let path = args.output.join(results.case.scheduler_result_file_name());
    write_results(&path, &results)?;
    tracing::info!(
        case = % results.case.name,
        path = % path.display(),
        wall_clock_ms = results.wall_clock_nanos / 1_000_000,
        expected_duration_ms = expected_duration_ms(&results.case),
        num_ticks = results.tick_samples.len(),
        linger_ms = args.linger_ms,
        num_jobs = results.job_e2e_records.len(),
        num_pinned_immediate = results.pinned_immediate_dispatch_latency.count,
        num_pinned_waited = results.pinned_waited_dispatch_latency.count,
        num_general_immediate = results.general_immediate_dispatch_latency.count,
        num_general_waited = results.general_waited_dispatch_latency.count,
        num_empty_responses = instrumentation.num_empty_responses(),
        "Benchmark scheduler finished."
    );

    check_completion_counts(&results)
}

/// Builds the fake inbound queue the case's workload is served from, timestamping each job's first
/// emission into `job_progress`.
///
/// # Returns
///
/// A newly created storage serving `case`'s workload.
fn build_storage(
    case: &BenchCase,
    job_progress: Arc<JobProgressMap>,
    run_origin: Instant,
) -> FakeStorage {
    let config = FakeStorageConfig {
        num_resource_groups: case.num_resource_groups,
        num_jobs_per_group: case.jobs_per_resource_group,
        num_tasks_per_job: case.tasks_per_job,
        emit_commit_ready: false,
        emit_cleanup_ready: false,
    };
    let release = ReleaseConfig {
        inbound_wave_size: NonZeroUsize::new(case.inbound_wave_size),
        jobs_per_batch: NonZeroUsize::new(case.jobs_per_batch),
        first_seen_recorder: Some(FirstSeenRecorder::new(job_progress, run_origin)),
    };

    FakeStorage::with_release_config(config, release)
}

/// Starts the case's core on a thread of its own.
///
/// # Returns
///
/// The join handle of the core's thread, resolving to the samples of every tick it executed.
fn spawn_core(
    case: &BenchCase,
    storage: &FakeStorage,
    rg_table: &ResourceGroupTable,
    global_queue: &GlobalDispatchQueue,
    session_manager: &SessionManager,
    cancellation_token: CancellationToken,
) -> JoinHandle<Result<Vec<TickSample>, CoreError>> {
    let core_config = case.core_config();
    let num_ticks = expected_tick_count(case);
    let storage = storage.clone();
    let rg_table = rg_table.clone();
    let global_queue = global_queue.clone();
    let session_manager = session_manager.clone();
    // The benchmark never replays a lost assignment, so the write side is dropped straight away;
    // the core reads the resulting closed queue exactly as it reads an empty one.
    let (_, reschedule_queue_reader) = unbounded_channel::<TaskAssignment>();

    run_core_on_dedicated_thread_with_tick_samples(move || {
        Core::new(
            core_config,
            storage,
            rg_table,
            global_queue,
            session_manager,
            reschedule_queue_reader,
            cancellation_token,
        )
        .with_tick_sample_capacity(num_ticks)
    })
}

/// Waits until every task of the run has been reported complete or `timeout` expires, reporting
/// progress as it goes.
///
/// # Returns
///
/// Whether the run drained before `timeout` expired.
async fn wait_until_drained(instrumentation: &BenchInstrumentation, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut next_log = tokio::time::Instant::now() + PROGRESS_LOG_INTERVAL;
    loop {
        if instrumentation.is_drained() {
            return true;
        }
        let now = tokio::time::Instant::now();
        let remaining = deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            return false;
        }
        if now >= next_log {
            tracing::info!(
                tasks_completed = instrumentation.tasks_completed(),
                total_tasks = instrumentation.total_tasks(),
                "Benchmark run in progress."
            );
            next_log = now + PROGRESS_LOG_INTERVAL;
        }
        tokio::time::sleep(DRAIN_POLL_INTERVAL.min(remaining)).await;
    }
}

/// Waits for the core's thread to exit.
///
/// # Returns
///
/// The samples of every tick the core executed, on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Internal`] if the blocking join task could not be joined, if the core's thread
///   panicked, or if the core exited on an error.
async fn join_core(
    core_thread: JoinHandle<Result<Vec<TickSample>, CoreError>>,
) -> Result<Vec<TickSample>, HarnessError> {
    let join_result = tokio::task::spawn_blocking(move || core_thread.join())
        .await
        .map_err(|error| HarnessError::Internal(error.to_string()))?;
    match join_result {
        Ok(Ok(tick_samples)) => Ok(tick_samples),
        Ok(Err(error)) => Err(HarnessError::Internal(error.to_string())),
        Err(_) => Err(HarnessError::Internal(
            "the scheduler core thread panicked".to_owned(),
        )),
    }
}

/// Assembles everything the run measured into its results file.
///
/// The wall clock runs from the earliest inbound emission to the latest completion report rather
/// than from process start to the poll that observed the drain, so that the figure it reports is
/// the workload's own duration and carries none of the drain poll's cadence.
///
/// # Returns
///
/// The case's self-describing results.
fn build_results(
    case: BenchCase,
    instrumentation: &BenchInstrumentation,
    tick_samples: Vec<TickSample>,
) -> SchedulerResults {
    let job_e2e_records = collect_job_records(instrumentation.job_progress());
    let first_emission_nanos = job_e2e_records
        .iter()
        .map(|record| record.first_seen_nanos)
        .min()
        .unwrap_or(0);
    let last_completion_nanos = job_e2e_records
        .iter()
        .map(|record| record.last_completed_nanos)
        .max()
        .unwrap_or(0);

    SchedulerResults {
        case,
        host_num_cores: host_num_cores(),
        build_profile: build_profile().to_owned(),
        wall_clock_nanos: last_completion_nanos.saturating_sub(first_emission_nanos),
        tick_samples,
        job_e2e_records,
        pinned_immediate_dispatch_latency: LatencySeries::from_histogram(
            instrumentation.pinned_immediate_dispatch_latency(),
        ),
        pinned_waited_dispatch_latency: LatencySeries::from_histogram(
            instrumentation.pinned_waited_dispatch_latency(),
        ),
        general_immediate_dispatch_latency: LatencySeries::from_histogram(
            instrumentation.general_immediate_dispatch_latency(),
        ),
        general_waited_dispatch_latency: LatencySeries::from_histogram(
            instrumentation.general_waited_dispatch_latency(),
        ),
        num_pinned_empty_responses: instrumentation.num_pinned_empty_responses(),
        num_general_empty_responses: instrumentation.num_general_empty_responses(),
    }
}

/// Checks the per-job completion counts the benchmark contract validates a run by.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Internal`] if the run did not see every job of the case, or if a job was not
///   reported complete exactly the case's tasks per job.
fn check_completion_counts(results: &SchedulerResults) -> Result<(), HarnessError> {
    if results.job_e2e_records.len() != results.case.total_jobs() {
        return Err(HarnessError::Internal(format!(
            "the run saw {} of the case's {} jobs",
            results.job_e2e_records.len(),
            results.case.total_jobs()
        )));
    }

    let tasks_per_job = u64::try_from(results.case.tasks_per_job).unwrap_or(u64::MAX);
    let num_incomplete_jobs = results
        .job_e2e_records
        .iter()
        .filter(|record| record.tasks_completed != tasks_per_job)
        .count();
    if 0 != num_incomplete_jobs {
        return Err(HarnessError::Internal(format!(
            "{num_incomplete_jobs} jobs did not complete exactly {tasks_per_job} tasks"
        )));
    }

    Ok(())
}

/// Writes `results` to `path`, creating its directory if it does not exist.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Internal`] if the directory could not be created, or if the file could not be
///   created, serialized into, or flushed.
fn write_results(path: &Path, results: &SchedulerResults) -> Result<(), HarnessError> {
    if let Some(directory) = path.parent() {
        std::fs::create_dir_all(directory).map_err(|error| {
            HarnessError::Internal(format!("failed to create {}: {error}", directory.display()))
        })?;
    }
    let file = File::create(path).map_err(|error| {
        HarnessError::Internal(format!("failed to create {}: {error}", path.display()))
    })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, results).map_err(|error| {
        HarnessError::Internal(format!("failed to serialize {}: {error}", path.display()))
    })?;
    writer.flush().map_err(|error| {
        HarnessError::Internal(format!("failed to write {}: {error}", path.display()))
    })
}

/// # Returns
///
/// The steady-state duration `case` is expected to take, in milliseconds: the workload's total
/// execution time spread over every worker.
fn expected_duration_ms(case: &BenchCase) -> u64 {
    let total_tasks = u64::try_from(case.total_tasks()).unwrap_or(u64::MAX);
    let num_workers = u64::try_from(case.total_workers())
        .unwrap_or(u64::MAX)
        .max(1);

    total_tasks.saturating_mul(case.task_duration_ms) / num_workers
}

/// # Returns
///
/// The timeout a run of `case` is declared hung after, in milliseconds, derived from the case's
/// expected duration so that no case carries a hand-picked limit.
fn default_timeout_ms(case: &BenchCase) -> u64 {
    expected_duration_ms(case)
        .saturating_mul(TIMEOUT_EXPECTED_DURATION_FACTOR)
        .saturating_add(TIMEOUT_HEADROOM_MS)
}

/// # Returns
///
/// The number of tick samples the core reserves room for, generous enough that a run several times
/// slower than expected still never grows the log while it is being measured.
fn expected_tick_count(case: &BenchCase) -> usize {
    let num_ticks = expected_duration_ms(case).saturating_mul(TICK_SAMPLE_SLACK_FACTOR)
        / case.tick_interval_ms.max(1);

    usize::try_from(num_ticks)
        .unwrap_or(usize::MAX)
        .saturating_add(TICK_SAMPLE_HEADROOM)
}
