use std::{sync::Arc, time::Instant};

use clap::Parser;
use tokio::sync::Mutex;

use spider_bench::{
    Compression,
    GetReadyTaskRequest, GetReadyTaskResponse,
    RegisterTaskInstanceRequest, RegisterTaskInstanceResponse,
    SubmitTaskResultRequest, SubmitTaskResultResponse,
    TaskLatency,
    compress_bytes, hex_encode,
    job_state_is_terminal, make_random_payload, report_stats,
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

    /// Size of each output payload in bytes.
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

        // Step 1: Get a ready task from this worker's dedicated queue. Retry if empty.
        let task_index = loop {
            if *done_rx.borrow() {
                return;
            }
            let resp: GetReadyTaskResponse = http_client
                .post(&get_ready_url)
                .json(&GetReadyTaskRequest { worker_id })
                .send()
                .await
                .expect("GetReadyTask request should succeed")
                .json()
                .await
                .expect("GetReadyTask response should deserialize");

            if resp.has_task {
                break resp.task_index;
            }
            tokio::task::yield_now().await;
        };

        // Step 2: Register task instance (timed — includes network round-trip).
        let register_start = Instant::now();
        let register_resp: RegisterTaskInstanceResponse = http_client
            .post(&register_url)
            .json(&RegisterTaskInstanceRequest { task_index })
            .send()
            .await
            .expect("RegisterTaskInstance request should succeed")
            .json()
            .await
            .expect("RegisterTaskInstance response should deserialize");
        let register_duration = register_start.elapsed();

        // Inputs received are already compressed (stored that way in the cache).
        // Decompression happens here, outside timing, simulating real worker behavior.
        let _inputs_compressed = register_resp.inputs_b64; // would decompress in real use

        // Step 3: Produce randomized 1KB output, compress it (outside timing).
        // The compressed bytes are what the server stores in the cache.
        let raw_output = make_random_payload(input_size);
        let compressed_output = compress_bytes(&raw_output, compression);
        let outputs: Vec<Vec<u8>> = vec![compressed_output];
        let outputs_bytes =
            rmp_serde::to_vec(&outputs).expect("output serialization should succeed");
        let outputs_hex = hex_encode(&outputs_bytes);

        // Step 4: Submit result (timed — includes network round-trip).
        let submit_start = Instant::now();
        let submit_resp: SubmitTaskResultResponse = http_client
            .post(&submit_url)
            .json(&SubmitTaskResultRequest {
                task_instance_id: register_resp.task_instance_id,
                task_index,
                success: true,
                outputs_b64: outputs_hex,
                error_message: String::new(),
            })
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
