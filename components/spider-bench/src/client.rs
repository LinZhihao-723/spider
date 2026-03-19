use std::{sync::Arc, time::Instant};

use clap::Parser;
use tokio::sync::Mutex;

use spider_bench::{
    Compression,
    GetReadyTaskRequest, GetReadyTaskResponse,
    HEADER_ERROR_MESSAGE, HEADER_SUCCESS, HEADER_TASK_INDEX, HEADER_TASK_INSTANCE_ID,
    RegisterTaskInstanceRequest,
    SubmitTaskResultResponse,
    TaskLatency,
    compress_bytes, job_state_is_terminal, make_random_payload, report_stats,
};

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser)]
#[command(name = "worker-client", about = "Cache-layer benchmark worker client (REST)")]
struct Cli {
    #[arg(long, default_value = "http://[::1]:50051")]
    server_addr: String,

    #[arg(long, default_value_t = 128)]
    num_workers: usize,

    #[arg(long, value_enum, default_value_t = Compression::None)]
    compression: Compression,

    #[arg(long, default_value_t = 1024)]
    input_size: usize,
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let http_client = reqwest::Client::new();

    eprintln!(
        "Connecting to server at {} (compression={})",
        cli.server_addr, cli.compression
    );

    // Wait for the server to be reachable before spawning workers.
    wait_for_server(&http_client, &cli.server_addr).await;
    eprintln!("Server is ready.");

    let latencies: Arc<Mutex<Vec<TaskLatency>>> = Arc::new(Mutex::new(Vec::new()));
    let (done_tx, done_rx) = tokio::sync::watch::channel(false);
    let done_tx = Arc::new(done_tx);

    let exec_start = Instant::now();

    let mut handles = Vec::with_capacity(cli.num_workers);
    for worker_id in 0..cli.num_workers {
        let http_client = http_client.clone();
        let base_url = cli.server_addr.clone();
        let latencies = Arc::clone(&latencies);
        let done_tx = Arc::clone(&done_tx);
        let mut done_rx = done_rx.clone();
        let compression = cli.compression;
        let input_size = cli.input_size;

        handles.push(tokio::spawn(async move {
            worker_loop(
                worker_id as u32,
                &http_client,
                &base_url,
                compression,
                input_size,
                latencies,
                &done_tx,
                &mut done_rx,
            )
            .await;
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
        cli.compression,
        total_execution,
        &latencies,
    );

    Ok(())
}

/// Polls the server until it responds, retrying with backoff. Gives up after 60 seconds.
async fn wait_for_server(http_client: &reqwest::Client, base_url: &str) {
    let url = format!("{base_url}/get_ready_task");
    let deadline = Instant::now() + std::time::Duration::from_secs(60);

    loop {
        match http_client
            .post(&url)
            .json(&GetReadyTaskRequest { worker_id: 0 })
            .send()
            .await
        {
            Ok(_) => return,
            Err(e) => {
                if Instant::now() > deadline {
                    panic!("Server at {base_url} not reachable after 60s: {e}");
                }
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }
    }
}

async fn worker_loop(
    worker_id: u32,
    http_client: &reqwest::Client,
    base_url: &str,
    compression: Compression,
    input_size: usize,
    latencies: Arc<Mutex<Vec<TaskLatency>>>,
    done_tx: &tokio::sync::watch::Sender<bool>,
    done_rx: &mut tokio::sync::watch::Receiver<bool>,
) {
    let get_ready_url = format!("{base_url}/get_ready_task");
    let register_url = format!("{base_url}/register_task_instance");
    let submit_url = format!("{base_url}/submit_task_result");

    loop {
        if *done_rx.borrow() {
            break;
        }

        // Step 1: Get a ready task. Retry on connection errors and empty queue.
        let task_index = loop {
            if *done_rx.borrow() {
                return;
            }
            let result = http_client
                .post(&get_ready_url)
                .json(&GetReadyTaskRequest { worker_id })
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let ready: GetReadyTaskResponse = resp
                        .json()
                        .await
                        .expect("GetReadyTask response should deserialize");
                    if ready.has_task {
                        break ready.task_index;
                    }
                    tokio::task::yield_now().await;
                }
                Err(e) if e.is_connect() || e.is_timeout() => {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                Err(e) => {
                    eprintln!("GetReadyTask error (worker {worker_id}): {e}");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        };

        // Step 2: Register task instance (timed).
        // Request is JSON (task_index), response is raw binary body + instance_id in header.
        let register_start = Instant::now();
        let register_resp = http_client
            .post(&register_url)
            .json(&RegisterTaskInstanceRequest { task_index })
            .send()
            .await
            .expect("RegisterTaskInstance request should succeed");

        let task_instance_id: u64 = register_resp
            .headers()
            .get(HEADER_TASK_INSTANCE_ID)
            .expect("response should have x-task-instance-id header")
            .to_str()
            .expect("header should be valid str")
            .parse()
            .expect("header should be valid u64");

        // Read raw binary body (inputs). Not timed beyond the HTTP round-trip.
        let _inputs_bytes = register_resp
            .bytes()
            .await
            .expect("should read response body");
        let register_duration = register_start.elapsed();

        // Step 3: Produce output (outside timing).
        let raw_output = make_random_payload(input_size);
        let compressed_output = compress_bytes(&raw_output, compression);
        let outputs: Vec<Vec<u8>> = vec![compressed_output];
        let outputs_bytes =
            rmp_serde::to_vec(&outputs).expect("output serialization should succeed");

        // Step 4: Submit result (timed).
        // Request is raw binary body with metadata in headers. Response is JSON.
        let submit_start = Instant::now();
        let submit_resp: SubmitTaskResultResponse = http_client
            .post(&submit_url)
            .header(HEADER_TASK_INSTANCE_ID, task_instance_id.to_string())
            .header(HEADER_TASK_INDEX, task_index.to_string())
            .header(HEADER_SUCCESS, "true")
            .header(HEADER_ERROR_MESSAGE, "")
            .body(outputs_bytes)
            .send()
            .await
            .expect("SubmitTaskResult request should succeed")
            .json()
            .await
            .expect("SubmitTaskResult response should deserialize");
        let submit_duration = submit_start.elapsed();

        latencies.lock().await.push(TaskLatency {
            task_index: task_index as usize,
            register_duration,
            submit_duration,
        });

        if job_state_is_terminal(&submit_resp.job_state) {
            let _ = done_tx.send(true);
            break;
        }
    }
}
