use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use async_trait::async_trait;
use clap::Parser;
use spider_core::{
    task::TaskIndex,
    types::{
        id::{JobId, ResourceGroupId},
        io::TaskOutput,
    },
};
use spider_storage::cache::{
    ReadyQueueConnector, TaskId,
    build_job,
    error::InternalError,
};
use tokio::sync::Mutex;
use tonic::{Request, Response, Status, transport::Server};

use spider_bench::{
    MockDb, MockInstancePool,
    build_flat_graph, build_neural_net_graph,
    job_state_to_proto,
    proto::{
        self,
        bench_scheduler_server::{BenchScheduler, BenchSchedulerServer},
    },
};

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser)]
#[command(name = "cache-server", about = "Cache-layer benchmark server")]
struct Cli {
    #[arg(long, default_value = "flat")]
    benchmark: String,

    #[arg(long, default_value_t = 10_000)]
    num_tasks: usize,

    #[arg(long, default_value_t = 50051)]
    port: u16,

    #[arg(long, default_value_t = 10)]
    neural_net_layers: usize,

    #[arg(long, default_value_t = 1000)]
    neural_net_width: usize,

    #[arg(long, default_value_t = 10)]
    neural_net_fan_in: usize,

    /// Number of workers expected. Must match the client's --num-workers.
    /// Used to create per-worker dispatch channels.
    #[arg(long, default_value_t = 128)]
    num_workers: usize,
}

// =============================================================================
// Server ready queue — round-robin dispatches to per-worker channels
// =============================================================================

type Jcb = spider_storage::cache::JobControlBlock<ServerReadyQueue, MockDb, MockInstancePool>;

struct ServerReadyQueue {
    worker_txs: Vec<tokio::sync::mpsc::UnboundedSender<TaskIndex>>,
    round_robin: AtomicU64,
}

#[async_trait]
impl ReadyQueueConnector for ServerReadyQueue {
    async fn send_task_ready(
        &self,
        _job_id: JobId,
        task_ids: Vec<TaskIndex>,
    ) -> Result<(), InternalError> {
        let num_workers = self.worker_txs.len();
        for idx in task_ids {
            let worker =
                self.round_robin.fetch_add(1, Ordering::Relaxed) as usize % num_workers;
            let _ = self.worker_txs[worker].send(idx);
        }
        Ok(())
    }

    async fn send_commit_ready(&self, _job_id: JobId) -> Result<(), InternalError> {
        Ok(())
    }

    async fn send_cleanup_ready(&self, _job_id: JobId) -> Result<(), InternalError> {
        Ok(())
    }
}

// =============================================================================
// gRPC service implementation
// =============================================================================

/// Server-side latency collector for RPC handlers.
struct ServerStats {
    register_durations: Mutex<Vec<std::time::Duration>>,
    submit_durations: Mutex<Vec<std::time::Duration>>,
}

impl ServerStats {
    fn new() -> Self {
        Self {
            register_durations: Mutex::new(Vec::new()),
            submit_durations: Mutex::new(Vec::new()),
        }
    }

    async fn report(&self) {
        let to_ms = |d: &std::time::Duration| d.as_secs_f64() * 1000.0;

        let register = self.register_durations.lock().await;
        let submit = self.submit_durations.lock().await;

        let mut register_ms: Vec<f64> = register.iter().map(to_ms).collect();
        let mut submit_ms: Vec<f64> = submit.iter().map(to_ms).collect();
        register_ms.sort_by(|a, b| a.partial_cmp(b).expect("should be comparable"));
        submit_ms.sort_by(|a, b| a.partial_cmp(b).expect("should be comparable"));

        eprintln!();
        eprintln!("=== Server-side RPC latencies ===");
        eprintln!("  --- RegisterTaskInstance (server) ---");
        eprintln!("    count:                       {:>10}", register_ms.len());
        eprintln!(
            "    avg:                         {:>10.3} ms",
            spider_bench::avg_of(&register_ms)
        );
        eprintln!(
            "    p50:                         {:>10.3} ms",
            spider_bench::percentile(&register_ms, 50.0)
        );
        eprintln!(
            "    p95:                         {:>10.3} ms",
            spider_bench::percentile(&register_ms, 95.0)
        );
        eprintln!(
            "    p99:                         {:>10.3} ms",
            spider_bench::percentile(&register_ms, 99.0)
        );
        eprintln!("  --- SubmitTaskResult (server) ---");
        eprintln!("    count:                       {:>10}", submit_ms.len());
        eprintln!(
            "    avg:                         {:>10.3} ms",
            spider_bench::avg_of(&submit_ms)
        );
        eprintln!(
            "    p50:                         {:>10.3} ms",
            spider_bench::percentile(&submit_ms, 50.0)
        );
        eprintln!(
            "    p95:                         {:>10.3} ms",
            spider_bench::percentile(&submit_ms, 95.0)
        );
        eprintln!(
            "    p99:                         {:>10.3} ms",
            spider_bench::percentile(&submit_ms, 99.0)
        );
        eprintln!();
    }
}

struct BenchService {
    jcb: Arc<Jcb>,
    /// Per-worker receive channels. Indexed by `worker_id` from the request.
    worker_rxs: Vec<Mutex<tokio::sync::mpsc::UnboundedReceiver<TaskIndex>>>,
    done: Arc<AtomicBool>,
    stats: Arc<ServerStats>,
}

#[tonic::async_trait]
impl BenchScheduler for BenchService {
    async fn get_ready_task(
        &self,
        request: Request<proto::GetReadyTaskRequest>,
    ) -> Result<Response<proto::GetReadyTaskResponse>, Status> {
        if self.done.load(Ordering::Relaxed) {
            return Ok(Response::new(proto::GetReadyTaskResponse {
                has_task: false,
                task_index: 0,
            }));
        }

        let worker_id = request.into_inner().worker_id as usize;
        if worker_id >= self.worker_rxs.len() {
            return Err(Status::invalid_argument(format!(
                "worker_id {worker_id} out of range (max {})",
                self.worker_rxs.len() - 1
            )));
        }

        // Each worker has its own channel — no contention with other workers.
        let result = self.worker_rxs[worker_id].lock().await.try_recv();
        match result {
            Ok(task_index) => Ok(Response::new(proto::GetReadyTaskResponse {
                has_task: true,
                task_index: task_index as u64,
            })),
            Err(_) => Ok(Response::new(proto::GetReadyTaskResponse {
                has_task: false,
                task_index: 0,
            })),
        }
    }

    async fn register_task_instance(
        &self,
        request: Request<proto::RegisterTaskInstanceRequest>,
    ) -> Result<Response<proto::RegisterTaskInstanceResponse>, Status> {
        let start = std::time::Instant::now();

        let req = request.into_inner();
        let task_index = req.task_index as usize;

        let ctx = self
            .jcb
            .create_task_instance(TaskId::TaskIndex(task_index))
            .await
            .map_err(|e| Status::internal(format!("create_task_instance failed: {e:?}")))?;

        let inputs_bytes = rmp_serde::to_vec(&ctx.inputs)
            .map_err(|e| Status::internal(format!("failed to serialize inputs: {e}")))?;

        self.stats.register_durations.lock().await.push(start.elapsed());

        Ok(Response::new(proto::RegisterTaskInstanceResponse {
            task_instance_id: ctx.task_instance_id,
            inputs: inputs_bytes,
        }))
    }

    async fn submit_task_result(
        &self,
        request: Request<proto::SubmitTaskResultRequest>,
    ) -> Result<Response<proto::SubmitTaskResultResponse>, Status> {
        let start = std::time::Instant::now();

        let req = request.into_inner();
        let task_index = req.task_index as usize;

        let state = if req.success {
            let outputs: Vec<TaskOutput> = rmp_serde::from_slice(&req.outputs)
                .map_err(|e| Status::internal(format!("failed to deserialize outputs: {e}")))?;
            self.jcb
                .complete_task_instance(req.task_instance_id, task_index, outputs)
                .await
                .map_err(|e| Status::internal(format!("complete_task_instance failed: {e:?}")))?
        } else {
            self.jcb
                .fail_task_instance(
                    req.task_instance_id,
                    TaskId::TaskIndex(task_index),
                    req.error_message,
                )
                .await
                .map_err(|e| Status::internal(format!("fail_task_instance failed: {e:?}")))?
        };

        self.stats.submit_durations.lock().await.push(start.elapsed());

        if state.is_terminal() {
            self.done.store(true, Ordering::Relaxed);
        }

        Ok(Response::new(proto::SubmitTaskResultResponse {
            job_state: job_state_to_proto(state),
        }))
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Build the graph.
    let (graph, job_inputs) = match cli.benchmark.as_str() {
        "flat" => build_flat_graph(cli.num_tasks, 2, 1),
        "neural-net" => {
            let (g, inputs, _layers) = build_neural_net_graph(
                cli.neural_net_layers,
                cli.neural_net_width,
                cli.neural_net_fan_in,
            );
            (g, inputs)
        }
        other => {
            eprintln!("Unknown benchmark: {other}. Use 'flat' or 'neural-net'.");
            std::process::exit(1);
        }
    };

    eprintln!(
        "Graph built: {} tasks, benchmark={}",
        graph.get_num_tasks(),
        cli.benchmark
    );

    // Create per-worker channels.
    let num_workers = cli.num_workers;
    let mut worker_txs = Vec::with_capacity(num_workers);
    let mut worker_rxs = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<TaskIndex>();
        worker_txs.push(tx);
        worker_rxs.push(Mutex::new(rx));
    }

    let ready_queue = ServerReadyQueue {
        worker_txs: worker_txs.clone(),
        round_robin: AtomicU64::new(0),
    };

    // Build the job.
    let (jcb, initial_ready) = build_job(
        JobId::new(),
        ResourceGroupId::new(),
        &graph,
        job_inputs,
        ready_queue,
        MockDb,
        MockInstancePool::new(),
    )
    .expect("build_job should succeed");

    eprintln!("Job built: {} initially ready tasks", initial_ready.len());

    // Seed initial ready tasks round-robin across worker channels.
    for (i, &idx) in initial_ready.iter().enumerate() {
        worker_txs[i % num_workers]
            .send(idx)
            .expect("channel should be open during seeding");
    }

    let done = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(ServerStats::new());

    let service = BenchService {
        jcb: Arc::new(jcb),
        worker_rxs,
        done: done.clone(),
        stats: Arc::clone(&stats),
    };

    let addr = format!("[::1]:{}", cli.port).parse()?;
    eprintln!("Server listening on {addr} ({num_workers} worker channels)");

    Server::builder()
        .add_service(BenchSchedulerServer::new(service))
        .serve_with_shutdown(addr, async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if done.load(Ordering::Relaxed) {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    break;
                }
            }
        })
        .await?;

    stats.report().await;
    eprintln!("Server shutting down.");
    Ok(())
}
