use std::{sync::Arc, time::Instant};

use clap::Parser;
use tokio::sync::Mutex;

use spider_bench::{
    TaskLatency, make_1kb_payload, report_stats,
    proto::bench_scheduler_client::BenchSchedulerClient,
    proto::{
        GetReadyTaskRequest, RegisterTaskInstanceRequest, SubmitTaskResultRequest,
    },
};

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser)]
#[command(name = "worker-client", about = "Cache-layer benchmark worker client")]
struct Cli {
    #[arg(long, default_value = "http://[::1]:50051")]
    server_addr: String,

    #[arg(long, default_value_t = 128)]
    num_workers: usize,
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let channel = tonic::transport::Channel::from_shared(cli.server_addr.clone())?
        .connect()
        .await?;

    eprintln!("Connected to server at {}", cli.server_addr);

    let latencies: Arc<Mutex<Vec<TaskLatency>>> = Arc::new(Mutex::new(Vec::new()));
    let (done_tx, done_rx) = tokio::sync::watch::channel(false);
    let done_tx = Arc::new(done_tx);

    let exec_start = Instant::now();

    let mut handles = Vec::with_capacity(cli.num_workers);
    for worker_id in 0..cli.num_workers {
        let client = BenchSchedulerClient::new(channel.clone());
        let latencies = Arc::clone(&latencies);
        let done_tx = Arc::clone(&done_tx);
        let mut done_rx = done_rx.clone();

        handles.push(tokio::spawn(async move {
            worker_loop(worker_id as u32, client, latencies, &done_tx, &mut done_rx).await;
        }));
    }

    for handle in handles {
        handle.await.expect("worker task should not panic");
    }

    let total_execution = exec_start.elapsed();

    let latencies = Arc::try_unwrap(latencies)
        .expect("all workers should have finished")
        .into_inner();

    report_stats(
        "Benchmark",
        cli.num_workers,
        total_execution,
        &latencies,
    );

    Ok(())
}

async fn worker_loop(
    worker_id: u32,
    mut client: BenchSchedulerClient<tonic::transport::Channel>,
    latencies: Arc<Mutex<Vec<TaskLatency>>>,
    done_tx: &tokio::sync::watch::Sender<bool>,
    done_rx: &mut tokio::sync::watch::Receiver<bool>,
) {
    loop {
        if *done_rx.borrow() {
            break;
        }

        // Step 1: Get a ready task from this worker's dedicated queue.
        let task_index = loop {
            if *done_rx.borrow() {
                return;
            }
            let ready_resp = client
                .get_ready_task(GetReadyTaskRequest { worker_id })
                .await
                .expect("GetReadyTask RPC should succeed")
                .into_inner();

            if ready_resp.has_task {
                break ready_resp.task_index;
            }
            // Queue temporarily empty — yield and retry.
            tokio::task::yield_now().await;
        };

        // Step 2: Register task instance (timed).
        let register_start = Instant::now();
        let register_resp = client
            .register_task_instance(RegisterTaskInstanceRequest { task_index })
            .await
            .expect("RegisterTaskInstance RPC should succeed")
            .into_inner();
        let register_duration = register_start.elapsed();

        let _task_instance_id = register_resp.task_instance_id;
        // Inputs are received but not used — we just produce dummy outputs.

        // Step 3: Produce 1KB output.
        let outputs = vec![make_1kb_payload()];
        let outputs_bytes = rmp_serde::to_vec(&outputs)
            .expect("output serialization should succeed");

        // Step 4: Submit result (timed).
        let submit_start = Instant::now();
        let submit_resp = client
            .submit_task_result(SubmitTaskResultRequest {
                task_instance_id: register_resp.task_instance_id,
                task_index,
                success: true,
                outputs: outputs_bytes,
                error_message: String::new(),
            })
            .await
            .expect("SubmitTaskResult RPC should succeed")
            .into_inner();
        let submit_duration = submit_start.elapsed();

        // Record latency.
        latencies.lock().await.push(TaskLatency {
            task_index: task_index as usize,
            register_duration,
            submit_duration,
        });

        // Check if job reached terminal state.
        if spider_bench::proto_job_state_is_terminal(submit_resp.job_state) {
            let _ = done_tx.send(true);
            break;
        }
    }
}
