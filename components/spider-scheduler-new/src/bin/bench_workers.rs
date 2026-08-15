//! The worker side of the prototype's performance evaluation.
//!
//! The binary hosts one shard of a benchmark case's execution managers as coroutines over a pooled
//! set of gRPC channels, and stops once the scheduler's bench service reports the run drained. A
//! single-node run is one shard of one; a multi-node run is the same binary started once per host
//! with a different shard index, so the topology is a command-line argument rather than a code
//! path.
//!
//! Every dimension but the shard split comes from the case table, and the results file the process
//! writes carries the case it ran, so a worker file and the scheduler file of the same run describe
//! one configuration by construction.

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use spider_scheduler_new::BenchCase;
use spider_scheduler_new::HarnessError;
use spider_scheduler_new::WorkerResults;
use spider_scheduler_new::harness::WorkerPool;
use spider_scheduler_new::harness::WorkerPoolConfig;
use tracing_subscriber::EnvFilter;

/// The log verbosity a run uses when the environment names none.
const DEFAULT_LOG_FILTER: &str = "info";

/// The multiple of a case's expected duration a run may take before it is declared hung.
const TIMEOUT_EXPECTED_DURATION_FACTOR: u64 = 10;

/// The headroom a derived run timeout adds for a case's start-up and drain-out, in milliseconds.
const TIMEOUT_HEADROOM_MS: u64 = 120_000;

/// The command line of a benchmark worker process.
#[derive(Debug, Parser)]
#[command(about = "Runs one shard of the workers of a prototype scheduler benchmark case")]
struct Args {
    /// The benchmark case to run, named as the case table names it.
    #[arg(long)]
    case: String,

    /// The scheduler this process pulls assignments from.
    #[arg(long)]
    endpoint: String,

    /// The directory the shard's results file is written into, under the name the case dictates.
    #[arg(long)]
    output: PathBuf,

    /// Which shard of the case's workers this process runs.
    #[arg(long, default_value_t = 0)]
    index: usize,

    /// How many shards the case's workers are split into.
    #[arg(long, default_value_t = 1)]
    num_shards: usize,

    /// How many gRPC channels the process' workers spread over. Defaults to one apiece, which is
    /// what keeps client-side latency a measure of transport rather than of queueing on the
    /// process' own connections; a smaller pool measures that queueing deliberately.
    #[arg(long)]
    channel_pool_size: Option<usize>,

    /// How long the process waits for the run to drain, in seconds. Defaults to ten times the
    /// case's expected duration plus two minutes.
    #[arg(long)]
    timeout_secs: Option<u64>,
}

/// # Returns
///
/// [`ExitCode::SUCCESS`] if the run drained and the shard's results were written, and
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
            tracing::error!(error = % error, "The benchmark workers failed.");
            ExitCode::FAILURE
        }
    }
}

/// Runs this process' shard of a case's workers and writes its results file.
///
/// A run that ends without the scheduler ever reporting it drained measures a truncated workload,
/// so it writes nothing and fails: a results file under the raw results directory is by
/// construction one of a complete run.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Config`] if no case carries the requested name, if the shard split is
///   impossible, or if the shard hosts no worker at all.
/// * [`HarnessError::Internal`] if the run did not drain.
/// * Forwards [`WorkerPool::run`]'s return values on failure.
/// * Forwards [`write_results`]'s return values on failure.
async fn run(args: Args) -> Result<(), HarnessError> {
    let case = BenchCase::from_name(&args.case)
        .ok_or_else(|| HarnessError::Config(format!("unknown benchmark case: {}", args.case)))?;
    let config = build_pool_config(&case, &args)?;
    let num_workers = config.num_workers();
    tracing::info!(
        case = % case.name,
        endpoint = % config.endpoint,
        shard_index = args.index,
        num_shards = args.num_shards,
        num_workers,
        channel_pool_size = config.channel_pool_size,
        timeout_ms = config.run_timeout_ms,
        "Benchmark workers starting."
    );

    let report = WorkerPool::run(config).await?;
    if !report.drained {
        return Err(HarnessError::Internal(format!(
            "the scheduler never reported the run drained; {} assignments were received",
            report
                .num_pinned_assignments()
                .saturating_add(report.num_general_assignments())
        )));
    }

    let results = report.into_results(case, args.index);
    let path = args
        .output
        .join(results.case.worker_result_file_name(args.index));
    write_results(&path, &results)?;
    tracing::info!(
        case = % results.case.name,
        path = % path.display(),
        num_workers,
        num_pinned_immediate = results.pinned_immediate_dispatch_latency.count,
        num_pinned_waited = results.pinned_waited_dispatch_latency.count,
        num_general_immediate = results.general_immediate_dispatch_latency.count,
        num_general_waited = results.general_waited_dispatch_latency.count,
        num_empty_responses = results.num_empty_responses,
        "Benchmark workers finished."
    );

    Ok(())
}

/// Derives this process' worker pool from the case and the shard split.
///
/// The pool starts as the case's entire worker set and is then narrowed to the shard's slice of it,
/// so a single-node run and one node of a multi-node run differ only in the counts. Its channel
/// pool follows the narrowed count unless the command line names a size, so every shard of every
/// run gives each of its workers a channel of its own unless it is told not to.
///
/// # Returns
///
/// The configuration of the shard's workers.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Config`] if there are no shards, if the shard index is not one of them, or if
///   the split leaves this shard with no worker to run.
fn build_pool_config(case: &BenchCase, args: &Args) -> Result<WorkerPoolConfig, HarnessError> {
    if 0 == args.num_shards {
        return Err(HarnessError::Config(
            "a run has at least one shard".to_owned(),
        ));
    }
    if args.index >= args.num_shards {
        return Err(HarnessError::Config(format!(
            "shard index {} is not one of the run's {} shards",
            args.index, args.num_shards
        )));
    }

    let mut config = WorkerPoolConfig::from_case(case, args.endpoint.clone(), args.index);
    config.dedicated_workers_per_group = shard_size(
        case.dedicated_workers_per_group,
        args.num_shards,
        args.index,
    );
    config.shared_workers = shard_size(case.shared_workers, args.num_shards, args.index);
    config.run_timeout_ms = args.timeout_secs.map_or_else(
        || default_timeout_ms(case),
        |secs| secs.saturating_mul(1_000),
    );
    if 0 == config.num_workers() {
        return Err(HarnessError::Config(format!(
            "shard {} of {} hosts none of the case's {} workers",
            args.index,
            args.num_shards,
            case.total_workers()
        )));
    }
    config.channel_pool_size = args
        .channel_pool_size
        .unwrap_or_else(|| config.num_workers());

    Ok(config)
}

/// Splits `total` workers over `num_shards` shards, giving the remainder to the lowest indices.
///
/// # Returns
///
/// The number of workers the shard at `index` runs.
const fn shard_size(total: usize, num_shards: usize, index: usize) -> usize {
    let remainder = total % num_shards;
    let extra = if index < remainder { 1 } else { 0 };

    total / num_shards + extra
}

/// Writes `results` to `path`, creating its directory if it does not exist.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`HarnessError::Internal`] if the directory could not be created, or if the file could not be
///   created, serialized into, or flushed.
fn write_results(path: &Path, results: &WorkerResults) -> Result<(), HarnessError> {
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
/// The timeout a run of `case` is declared hung after, in milliseconds, derived from the case's
/// expected duration -- the workload's total execution time spread over every worker -- so that no
/// case carries a hand-picked limit.
fn default_timeout_ms(case: &BenchCase) -> u64 {
    let total_tasks = u64::try_from(case.total_tasks()).unwrap_or(u64::MAX);
    let num_workers = u64::try_from(case.total_workers())
        .unwrap_or(u64::MAX)
        .max(1);
    let expected_duration_ms = total_tasks.saturating_mul(case.task_duration_ms) / num_workers;

    expected_duration_ms
        .saturating_mul(TIMEOUT_EXPECTED_DURATION_FACTOR)
        .saturating_add(TIMEOUT_HEADROOM_MS)
}
