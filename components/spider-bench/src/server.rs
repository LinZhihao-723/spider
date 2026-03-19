use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Instant;

use async_trait::async_trait;
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
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

use spider_bench::{
    Compression,
    GetReadyTaskRequest, GetReadyTaskResponse,
    HEADER_ERROR_MESSAGE, HEADER_SUCCESS, HEADER_TASK_INDEX, HEADER_TASK_INSTANCE_ID,
    MockDb, MockInstancePool,
    RegisterTaskInstanceRequest,
    SubmitTaskResultResponse,
    avg_of,
    build_flat_graph, build_neural_net_graph,
    job_state_to_string,
    percentile,
};

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser)]
#[command(name = "cache-server", about = "Cache-layer benchmark server (REST)")]
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

    #[arg(long, default_value_t = 128)]
    num_workers: usize,

    #[arg(long, value_enum, default_value_t = Compression::None)]
    compression: Compression,

    #[arg(long, default_value_t = 1024)]
    input_size: usize,
}

// =============================================================================
// Server ready queue
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
// Server-side stats
// =============================================================================

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
        eprintln!("    avg:                         {:>10.3} ms", avg_of(&register_ms));
        eprintln!("    p50:                         {:>10.3} ms", percentile(&register_ms, 50.0));
        eprintln!("    p95:                         {:>10.3} ms", percentile(&register_ms, 95.0));
        eprintln!("    p99:                         {:>10.3} ms", percentile(&register_ms, 99.0));
        eprintln!("  --- SubmitTaskResult (server) ---");
        eprintln!("    count:                       {:>10}", submit_ms.len());
        eprintln!("    avg:                         {:>10.3} ms", avg_of(&submit_ms));
        eprintln!("    p50:                         {:>10.3} ms", percentile(&submit_ms, 50.0));
        eprintln!("    p95:                         {:>10.3} ms", percentile(&submit_ms, 95.0));
        eprintln!("    p99:                         {:>10.3} ms", percentile(&submit_ms, 99.0));
        eprintln!();
    }
}

// =============================================================================
// App state
// =============================================================================

struct AppState {
    jcb: Arc<Jcb>,
    worker_rxs: Vec<Mutex<tokio::sync::mpsc::UnboundedReceiver<TaskIndex>>>,
    done: Arc<AtomicBool>,
    stats: Arc<ServerStats>,
}

// =============================================================================
// Route handlers
// =============================================================================

async fn handle_get_ready_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GetReadyTaskRequest>,
) -> Json<GetReadyTaskResponse> {
    if state.done.load(Ordering::Relaxed) {
        return Json(GetReadyTaskResponse {
            has_task: false,
            task_index: 0,
        });
    }

    let worker_id = req.worker_id as usize;
    if worker_id >= state.worker_rxs.len() {
        return Json(GetReadyTaskResponse {
            has_task: false,
            task_index: 0,
        });
    }

    let result = state.worker_rxs[worker_id].lock().await.try_recv();
    match result {
        Ok(task_index) => Json(GetReadyTaskResponse {
            has_task: true,
            task_index: task_index as u64,
        }),
        Err(_) => Json(GetReadyTaskResponse {
            has_task: false,
            task_index: 0,
        }),
    }
}

/// Returns raw binary body (rmp_serde bytes of inputs) with task_instance_id in a header.
async fn handle_register_task_instance(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterTaskInstanceRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    let task_index = req.task_index as usize;
    let ctx = state
        .jcb
        .create_task_instance(TaskId::TaskIndex(task_index))
        .await
        .expect("create_task_instance should succeed");

    state
        .stats
        .register_durations
        .lock()
        .await
        .push(start.elapsed());

    let inputs_bytes =
        rmp_serde::to_vec(&ctx.inputs).expect("input serialization should succeed");

    let mut headers = HeaderMap::new();
    headers.insert(
        HEADER_TASK_INSTANCE_ID,
        ctx.task_instance_id.to_string().parse().expect("u64 should be a valid header value"),
    );

    (StatusCode::OK, headers, inputs_bytes)
}

/// Accepts raw binary body (rmp_serde bytes of outputs) with metadata in headers.
async fn handle_submit_task_result(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Json<SubmitTaskResultResponse> {
    let start = Instant::now();

    let task_instance_id: u64 = headers
        .get(HEADER_TASK_INSTANCE_ID)
        .expect("missing x-task-instance-id header")
        .to_str()
        .expect("header should be valid str")
        .parse()
        .expect("header should be valid u64");
    let task_index: usize = headers
        .get(HEADER_TASK_INDEX)
        .expect("missing x-task-index header")
        .to_str()
        .expect("header should be valid str")
        .parse()
        .expect("header should be valid usize");
    let success: bool = headers
        .get(HEADER_SUCCESS)
        .expect("missing x-success header")
        .to_str()
        .expect("header should be valid str")
        .parse()
        .expect("header should be valid bool");

    let job_state = if success {
        let outputs: Vec<TaskOutput> =
            rmp_serde::from_slice(&body).expect("output deserialization should succeed");
        state
            .jcb
            .complete_task_instance(task_instance_id, task_index, outputs)
            .await
            .expect("complete_task_instance should succeed")
    } else {
        let error_message = headers
            .get(HEADER_ERROR_MESSAGE)
            .map(|v| v.to_str().unwrap_or("").to_owned())
            .unwrap_or_default();
        state
            .jcb
            .fail_task_instance(
                task_instance_id,
                TaskId::TaskIndex(task_index),
                error_message,
            )
            .await
            .expect("fail_task_instance should succeed")
    };

    state
        .stats
        .submit_durations
        .lock()
        .await
        .push(start.elapsed());

    if job_state.is_terminal() {
        state.done.store(true, Ordering::Relaxed);
    }

    Json(SubmitTaskResultResponse {
        job_state: job_state_to_string(job_state),
    })
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let (graph, job_inputs) = match cli.benchmark.as_str() {
        "flat" => build_flat_graph(cli.num_tasks, 2, 1, cli.input_size, cli.compression),
        "neural-net" => {
            let (g, inputs, _layers) = build_neural_net_graph(
                cli.neural_net_layers,
                cli.neural_net_width,
                cli.neural_net_fan_in,
                cli.input_size,
                cli.compression,
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

    for (i, &idx) in initial_ready.iter().enumerate() {
        worker_txs[i % num_workers]
            .send(idx)
            .expect("channel should be open during seeding");
    }

    let done = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(ServerStats::new());

    let app_state = Arc::new(AppState {
        jcb: Arc::new(jcb),
        worker_rxs,
        done: done.clone(),
        stats: Arc::clone(&stats),
    });

    let app = Router::new()
        .route("/get_ready_task", post(handle_get_ready_task))
        .route("/register_task_instance", post(handle_register_task_instance))
        .route("/submit_task_result", post(handle_submit_task_result))
        .with_state(app_state);

    let addr = format!("[::1]:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("Server listening on {addr} ({num_workers} worker channels)");

    let done_for_shutdown = done.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if done_for_shutdown.load(Ordering::Relaxed) {
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
