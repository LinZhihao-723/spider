//! End-to-end latency benchmark for
//! [`ExecutionManagerLivenessManagement::register_execution_manager`].
//!
//! Spawns a configurable number of concurrent workers (default 128), each issuing a stream of
//! `register_execution_manager` calls against a live `MariaDB` instance, and reports per-call
//! latency percentiles plus aggregate throughput.
//!
//! Connection parameters are read from the same environment variables used by the storage tests:
//! `MARIADB_PORT`, `MARIADB_DATABASE`, `MARIADB_USERNAME`, and `MARIADB_PASSWORD`.
//!
//! The following optional environment variables tune the workload:
//!
//! * `BENCH_CONCURRENCY`: Number of concurrent workers (default `128`). Also sets the connection
//!   pool size so every worker can hold its own connection.
//! * `BENCH_ITERS_PER_WORKER`: Timed registration calls per worker (default `100`).
//! * `BENCH_WARMUP_PER_WORKER`: Untimed warmup calls per worker, used to fill the connection pool
//!   before timing begins (default `10`).

use std::{
    net::{IpAddr, Ipv4Addr},
    sync::Arc,
    time::{Duration, Instant},
};

use secrecy::SecretString;
use spider_storage::{
    DatabaseConfig,
    db::{ExecutionManagerLivenessManagement, MariaDbStorageConnector},
};
use tokio::sync::Barrier;

/// Default number of concurrent workers issuing registration calls.
const DEFAULT_CONCURRENCY: usize = 128;
/// Default number of timed registration calls each worker performs.
const DEFAULT_ITERS_PER_WORKER: usize = 100;
/// Default number of untimed warmup calls each worker performs before timing begins.
const DEFAULT_WARMUP_PER_WORKER: usize = 10;
/// IP address registered for every execution manager in the benchmark.
const BENCH_IP: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// Latency samples (in nanoseconds) and error count collected by a single worker.
struct WorkerResult {
    /// Per-call latencies in nanoseconds for the timed region.
    latencies_ns: Vec<u64>,
    /// Number of failed registration calls.
    errors: u64,
}

/// Reads a `usize` from the environment, falling back to `default` if unset or unparseable.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Builds a [`DatabaseConfig`] from the `MARIADB_*` environment variables.
///
/// # Parameters
///
/// * `max_connections`: Maximum number of connections the pool may open.
///
/// # Returns
///
/// A [`DatabaseConfig`] pointing at the `MariaDB` instance described by the environment.
///
/// # Panics
///
/// Panics if any of `MARIADB_PORT`, `MARIADB_DATABASE`, `MARIADB_USERNAME`, or `MARIADB_PASSWORD`
/// is missing, or if `MARIADB_PORT` is not a valid port number.
fn config_from_env(max_connections: u32) -> DatabaseConfig {
    let port: u16 = std::env::var("MARIADB_PORT")
        .expect("MARIADB_PORT must be set")
        .parse()
        .expect("MARIADB_PORT must be a valid port number");
    let name = std::env::var("MARIADB_DATABASE").expect("MARIADB_DATABASE must be set");
    let username = std::env::var("MARIADB_USERNAME").expect("MARIADB_USERNAME must be set");
    let password = std::env::var("MARIADB_PASSWORD").expect("MARIADB_PASSWORD must be set");

    DatabaseConfig {
        host: "localhost".to_string(),
        port,
        name,
        username,
        password: SecretString::from(password),
        max_connections,
    }
}

/// Runs a single worker: warms up the pool, waits for all workers at `barrier`, then issues
/// `iters` timed registration calls.
///
/// # Parameters
///
/// * `connector`: Storage connector shared across workers (cheap to clone; shares the pool).
/// * `barrier`: Barrier all workers wait on so timing starts simultaneously.
/// * `warmup`: Number of untimed calls performed before the barrier.
/// * `iters`: Number of timed calls performed after the barrier.
///
/// # Returns
///
/// The [`WorkerResult`] holding this worker's latency samples and error count.
async fn run_worker(
    connector: MariaDbStorageConnector,
    barrier: Arc<Barrier>,
    warmup: usize,
    iters: usize,
) -> WorkerResult {
    for _ in 0..warmup {
        let _ = connector.register_execution_manager(BENCH_IP).await;
    }

    barrier.wait().await;

    let mut latencies_ns = Vec::with_capacity(iters);
    let mut errors = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let result = connector.register_execution_manager(BENCH_IP).await;
        let elapsed = start.elapsed();
        match result {
            Ok(_) => latencies_ns.push(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)),
            Err(_) => errors += 1,
        }
    }

    WorkerResult {
        latencies_ns,
        errors,
    }
}

/// Returns the latency at percentile `p` (0.0-100.0) from `sorted`, using nearest-rank.
///
/// # Parameters
///
/// * `sorted`: Latency samples in nanoseconds, sorted ascending.
/// * `p`: Percentile in the range `[0.0, 100.0]`.
///
/// # Returns
///
/// The sampled latency in nanoseconds, or `0` if `sorted` is empty.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn percentile_ns(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank]
}

/// Formats a nanosecond latency as milliseconds with three decimal places.
#[allow(clippy::cast_precision_loss)]
fn ms(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

/// Prints the benchmark report to stdout.
///
/// # Parameters
///
/// * `concurrency`: Number of concurrent workers.
/// * `warmup`: Warmup calls per worker.
/// * `iters`: Timed calls per worker.
/// * `latencies_ns`: All timed latency samples in nanoseconds (will be sorted in place).
/// * `errors`: Total number of failed calls across all workers.
/// * `wall`: Wall-clock duration of the timed region.
#[allow(clippy::cast_precision_loss)]
fn report(
    concurrency: usize,
    warmup: usize,
    iters: usize,
    mut latencies_ns: Vec<u64>,
    errors: u64,
    wall: Duration,
) {
    latencies_ns.sort_unstable();
    let count = latencies_ns.len();
    let sum: u128 = latencies_ns.iter().map(|&n| u128::from(n)).sum();
    let mean_ns = if count == 0 {
        0
    } else {
        u64::try_from(sum / count as u128).unwrap_or(u64::MAX)
    };
    let throughput = if wall.as_secs_f64() > 0.0 {
        count as f64 / wall.as_secs_f64()
    } else {
        0.0
    };

    println!("=== register_execution_manager latency benchmark ===");
    println!("concurrency:        {concurrency} workers");
    println!("warmup / worker:    {warmup} calls");
    println!("timed / worker:     {iters} calls");
    println!("timed calls total:  {count}");
    println!("errors:             {errors}");
    println!("wall time:          {:.3} s", wall.as_secs_f64());
    println!("throughput:         {throughput:.1} calls/s");
    println!();
    println!("latency (ms):");
    println!("  min     {:.3}", ms(*latencies_ns.first().unwrap_or(&0)));
    println!("  mean    {:.3}", ms(mean_ns));
    println!("  p50     {:.3}", ms(percentile_ns(&latencies_ns, 50.0)));
    println!("  p90     {:.3}", ms(percentile_ns(&latencies_ns, 90.0)));
    println!("  p95     {:.3}", ms(percentile_ns(&latencies_ns, 95.0)));
    println!("  p99     {:.3}", ms(percentile_ns(&latencies_ns, 99.0)));
    println!("  p99.9   {:.3}", ms(percentile_ns(&latencies_ns, 99.9)));
    println!("  max     {:.3}", ms(*latencies_ns.last().unwrap_or(&0)));
}

#[tokio::main]
async fn main() {
    let concurrency = env_usize("BENCH_CONCURRENCY", DEFAULT_CONCURRENCY).max(1);
    let iters = env_usize("BENCH_ITERS_PER_WORKER", DEFAULT_ITERS_PER_WORKER).max(1);
    let warmup = env_usize("BENCH_WARMUP_PER_WORKER", DEFAULT_WARMUP_PER_WORKER);

    let max_connections = u32::try_from(concurrency).unwrap_or(u32::MAX);
    let config = config_from_env(max_connections);
    let connector = MariaDbStorageConnector::connect(&config)
        .await
        .expect("failed to connect to MariaDB");

    // The barrier has `concurrency + 1` parties: one per worker plus `main`. Workers arrive after
    // warmup; `main` arrives immediately. The barrier therefore releases exactly when the timed
    // region begins, letting `main` start the wall-clock timer at that instant.
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let connector = connector.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(tokio::spawn(run_worker(connector, barrier, warmup, iters)));
    }

    barrier.wait().await;
    let wall_start = Instant::now();
    let mut latencies_ns = Vec::with_capacity(concurrency * iters);
    let mut errors = 0;
    for handle in handles {
        let result = handle.await.expect("worker task panicked");
        latencies_ns.extend(result.latencies_ns);
        errors += result.errors;
    }
    let wall = wall_start.elapsed();

    report(concurrency, warmup, iters, latencies_ns, errors, wall);
}
