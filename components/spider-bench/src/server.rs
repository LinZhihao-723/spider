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
use tokio::sync::{Mutex, mpsc};
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
}

// =============================================================================
// Server ready queue — feeds task indices into an mpsc channel
// =============================================================================

type Jcb = spider_storage::cache::JobControlBlock<ServerReadyQueue, MockDb, MockInstancePool>;

struct ServerReadyQueue {
    tx: mpsc::UnboundedSender<TaskIndex>,
    ready_count: AtomicU64,
}

#[async_trait]
impl ReadyQueueConnector for ServerReadyQueue {
    async fn send_task_ready(
        &self,
        _job_id: JobId,
        task_ids: Vec<TaskIndex>,
    ) -> Result<(), InternalError> {
        self.ready_count
            .fetch_add(task_ids.len() as u64, Ordering::Relaxed);
        for idx in task_ids {
            let _ = self.tx.send(idx);
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

struct BenchService {
    jcb: Arc<Jcb>,
    rx: Arc<Mutex<mpsc::UnboundedReceiver<TaskIndex>>>,
    done: Arc<AtomicBool>,
}

#[tonic::async_trait]
impl BenchScheduler for BenchService {
    async fn get_ready_task(
        &self,
        _request: Request<proto::GetReadyTaskRequest>,
    ) -> Result<Response<proto::GetReadyTaskResponse>, Status> {
        if self.done.load(Ordering::Relaxed) {
            return Ok(Response::new(proto::GetReadyTaskResponse {
                has_task: false,
                task_index: 0,
            }));
        }

        // Use try_recv to avoid holding the mutex across an await point, which would
        // serialize all GetReadyTask calls.
        let result = self.rx.lock().await.try_recv();
        match result {
            Ok(task_index) => Ok(Response::new(proto::GetReadyTaskResponse {
                has_task: true,
                task_index: task_index as u64,
            })),
            Err(_) => {
                // No task available right now. Return has_task=false so the client retries.
                // This is NOT the same as "job is done" — the client distinguishes by checking
                // the done state from SubmitTaskResult responses.
                Ok(Response::new(proto::GetReadyTaskResponse {
                    has_task: false,
                    task_index: 0,
                }))
            }
        }
    }

    async fn register_task_instance(
        &self,
        request: Request<proto::RegisterTaskInstanceRequest>,
    ) -> Result<Response<proto::RegisterTaskInstanceResponse>, Status> {
        let req = request.into_inner();
        let task_index = req.task_index as usize;

        let ctx = self
            .jcb
            .create_task_instance(TaskId::TaskIndex(task_index))
            .await
            .map_err(|e| Status::internal(format!("create_task_instance failed: {e:?}")))?;

        let inputs_bytes = rmp_serde::to_vec(&ctx.inputs)
            .map_err(|e| Status::internal(format!("failed to serialize inputs: {e}")))?;

        Ok(Response::new(proto::RegisterTaskInstanceResponse {
            task_instance_id: ctx.task_instance_id,
            inputs: inputs_bytes,
        }))
    }

    async fn submit_task_result(
        &self,
        request: Request<proto::SubmitTaskResultRequest>,
    ) -> Result<Response<proto::SubmitTaskResultResponse>, Status> {
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

    // Create the ready queue channel.
    let (tx, rx) = mpsc::unbounded_channel::<TaskIndex>();
    let ready_queue = ServerReadyQueue {
        tx: tx.clone(),
        ready_count: AtomicU64::new(0),
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

    // Seed initial ready tasks.
    for &idx in &initial_ready {
        tx.send(idx).expect("channel should be open");
    }

    let done = Arc::new(AtomicBool::new(false));

    let service = BenchService {
        jcb: Arc::new(jcb),
        rx: Arc::new(Mutex::new(rx)),
        done: done.clone(),
    };

    let addr = format!("[::1]:{}", cli.port).parse()?;
    eprintln!("Server listening on {addr}");

    // Run server until the job completes.
    Server::builder()
        .add_service(BenchSchedulerServer::new(service))
        .serve_with_shutdown(addr, async move {
            // Poll until done.
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if done.load(Ordering::Relaxed) {
                    // Grace period for in-flight RPCs.
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    break;
                }
            }
        })
        .await?;

    eprintln!("Server shutting down.");
    Ok(())
}
