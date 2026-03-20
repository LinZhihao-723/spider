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

#[derive(Parser)]
#[command(name = "cache-server")]
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

// --- Ready queue: per-worker channels with round-robin ---

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
        let n = self.worker_txs.len();
        for idx in task_ids {
            let w = self.round_robin.fetch_add(1, Ordering::Relaxed) as usize % n;
            let _ = self.worker_txs[w].send(idx);
        }
        Ok(())
    }
    async fn send_commit_ready(&self, _: JobId) -> Result<(), InternalError> {
        Ok(())
    }
    async fn send_cleanup_ready(&self, _: JobId) -> Result<(), InternalError> {
        Ok(())
    }
}

// --- Server stats ---

struct ServerStats {
    register_durations: Mutex<Vec<std::time::Duration>>,
    submit_durations: Mutex<Vec<std::time::Duration>>,
    first_register: Mutex<Option<Instant>>,
    last_terminal: Mutex<Option<Instant>>,
}

impl ServerStats {
    fn new() -> Self {
        Self {
            register_durations: Mutex::new(Vec::new()),
            submit_durations: Mutex::new(Vec::new()),
            first_register: Mutex::new(None),
            last_terminal: Mutex::new(None),
        }
    }
    async fn record_register(&self, d: std::time::Duration) {
        let mut first = self.first_register.lock().await;
        if first.is_none() {
            *first = Some(Instant::now());
        }
        drop(first);
        self.register_durations.lock().await.push(d);
    }
    async fn record_submit(&self, d: std::time::Duration, terminal: bool) {
        self.submit_durations.lock().await.push(d);
        if terminal {
            *self.last_terminal.lock().await = Some(Instant::now());
        }
    }
    async fn report(&self) {
        let to_ms = |d: &std::time::Duration| d.as_secs_f64() * 1000.0;
        let first = self.first_register.lock().await;
        let last = self.last_terminal.lock().await;
        let total = match (*first, *last) {
            (Some(f), Some(l)) => to_ms(&l.duration_since(f)),
            _ => 0.0,
        };
        let reg = self.register_durations.lock().await;
        let sub = self.submit_durations.lock().await;
        let mut reg_ms: Vec<f64> = reg.iter().map(to_ms).collect();
        let mut sub_ms: Vec<f64> = sub.iter().map(to_ms).collect();
        reg_ms.sort_by(|a, b| a.partial_cmp(b).expect("cmp"));
        sub_ms.sort_by(|a, b| a.partial_cmp(b).expect("cmp"));

        eprintln!();
        eprintln!("=== Server-side stats ===");
        eprintln!("  server_total_time: {:.2} ms", total);
        eprintln!("  server_register_avg: {:.3} ms", avg_of(&reg_ms));
        eprintln!("  server_submit_avg: {:.3} ms", avg_of(&sub_ms));
        eprintln!("  --- Register (server) ---");
        eprintln!("    count: {}", reg_ms.len());
        eprintln!("    avg: {:.3} ms  p50: {:.3} ms  p95: {:.3} ms  p99: {:.3} ms",
            avg_of(&reg_ms), percentile(&reg_ms, 50.0), percentile(&reg_ms, 95.0), percentile(&reg_ms, 99.0));
        eprintln!("  --- Submit (server) ---");
        eprintln!("    count: {}", sub_ms.len());
        eprintln!("    avg: {:.3} ms  p50: {:.3} ms  p95: {:.3} ms  p99: {:.3} ms",
            avg_of(&sub_ms), percentile(&sub_ms, 50.0), percentile(&sub_ms, 95.0), percentile(&sub_ms, 99.0));
        eprintln!();
    }
}

// --- App state ---

struct AppState {
    jcb: Arc<Jcb>,
    worker_rxs: Vec<Mutex<tokio::sync::mpsc::UnboundedReceiver<TaskIndex>>>,
    done: Arc<AtomicBool>,
    stats: Arc<ServerStats>,
}

// --- Handlers ---

async fn handle_get_ready_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GetReadyTaskRequest>,
) -> Json<GetReadyTaskResponse> {
    if state.done.load(Ordering::Relaxed) {
        return Json(GetReadyTaskResponse { has_task: false, task_index: 0 });
    }
    let wid = req.worker_id as usize;
    if wid >= state.worker_rxs.len() {
        return Json(GetReadyTaskResponse { has_task: false, task_index: 0 });
    }
    match state.worker_rxs[wid].lock().await.try_recv() {
        Ok(idx) => Json(GetReadyTaskResponse { has_task: true, task_index: idx as u64 }),
        Err(_) => Json(GetReadyTaskResponse { has_task: false, task_index: 0 }),
    }
}

async fn handle_register_task_instance(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterTaskInstanceRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let ctx = state.jcb
        .create_task_instance(TaskId::TaskIndex(req.task_index as usize))
        .await
        .expect("create_task_instance should succeed");
    state.stats.record_register(start.elapsed()).await;

    let body = rmp_serde::to_vec(&ctx.inputs).expect("serialize inputs");
    let mut headers = HeaderMap::new();
    headers.insert(HEADER_TASK_INSTANCE_ID,
        ctx.task_instance_id.to_string().parse().expect("valid header"));
    (StatusCode::OK, headers, body)
}

async fn handle_submit_task_result(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Json<SubmitTaskResultResponse> {
    let start = Instant::now();
    let iid: u64 = headers.get(HEADER_TASK_INSTANCE_ID).expect("hdr").to_str().expect("str").parse().expect("u64");
    let tidx: usize = headers.get(HEADER_TASK_INDEX).expect("hdr").to_str().expect("str").parse().expect("usize");
    let ok: bool = headers.get(HEADER_SUCCESS).expect("hdr").to_str().expect("str").parse().expect("bool");

    let js = if ok {
        let outputs: Vec<TaskOutput> = rmp_serde::from_slice(&body).expect("deser outputs");
        state.jcb.complete_task_instance(iid, tidx, outputs).await.expect("complete")
    } else {
        let msg = headers.get(HEADER_ERROR_MESSAGE).map(|v| v.to_str().unwrap_or("").to_owned()).unwrap_or_default();
        state.jcb.fail_task_instance(iid, TaskId::TaskIndex(tidx), msg).await.expect("fail")
    };

    let terminal = js.is_terminal();
    state.stats.record_submit(start.elapsed(), terminal).await;
    if terminal { state.done.store(true, Ordering::Relaxed); }

    Json(SubmitTaskResultResponse { job_state: job_state_to_string(js) })
}

// --- Main ---

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let (graph, job_inputs) = match cli.benchmark.as_str() {
        "flat" => build_flat_graph(cli.num_tasks, 2, 1, cli.input_size, cli.compression),
        "neural-net" => {
            let (g, i, _) = build_neural_net_graph(
                cli.neural_net_layers, cli.neural_net_width, cli.neural_net_fan_in,
                cli.input_size, cli.compression);
            (g, i)
        }
        o => { eprintln!("Unknown: {o}"); std::process::exit(1); }
    };
    eprintln!("Graph: {} tasks, benchmark={}", graph.get_num_tasks(), cli.benchmark);

    let nw = cli.num_workers;
    let mut txs = Vec::with_capacity(nw);
    let mut rxs = Vec::with_capacity(nw);
    for _ in 0..nw {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<TaskIndex>();
        txs.push(tx);
        rxs.push(Mutex::new(rx));
    }

    let rq = ServerReadyQueue { worker_txs: txs.clone(), round_robin: AtomicU64::new(0) };
    let (jcb, ready) = build_job(JobId::new(), ResourceGroupId::new(), &graph, job_inputs, rq, MockDb, MockInstancePool::new())
        .expect("build_job");
    eprintln!("Job: {} ready", ready.len());

    for (i, &idx) in ready.iter().enumerate() {
        txs[i % nw].send(idx).expect("seed");
    }

    let done = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(ServerStats::new());
    let state = Arc::new(AppState { jcb: Arc::new(jcb), worker_rxs: rxs, done: done.clone(), stats: stats.clone() });

    let app = Router::new()
        .route("/get_ready_task", post(handle_get_ready_task))
        .route("/register_task_instance", post(handle_register_task_instance))
        .route("/submit_task_result", post(handle_submit_task_result))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("Listening on {addr} ({nw} workers)");

    let d = done.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if d.load(Ordering::Relaxed) {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    break;
                }
            }
        }).await?;

    stats.report().await;
    eprintln!("Server done.");
    Ok(())
}
