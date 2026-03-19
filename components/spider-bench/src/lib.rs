use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use spider_core::{
    job::JobState,
    task::{
        BytesTypeDescriptor, DataTypeDescriptor, ExecutionPolicy, TaskDescriptor,
        TaskGraph as CoreTaskGraph, TaskIndex, TaskInputOutputIndex, ValueTypeDescriptor,
    },
    types::{
        id::{JobId, TaskInstanceId},
        io::{TaskInput, TaskOutput},
    },
};
use spider_storage::{
    cache::{
        SharedTaskControlBlock, SharedTerminationTaskControlBlock, TaskInstancePoolConnector,
        error::InternalError,
    },
    db::DbError,
};

// =============================================================================
// REST API types (JSON-serializable)
// =============================================================================

#[derive(Serialize, Deserialize)]
pub struct GetReadyTaskRequest {
    pub worker_id: u32,
}

#[derive(Serialize, Deserialize)]
pub struct GetReadyTaskResponse {
    pub has_task: bool,
    pub task_index: u64,
}

#[derive(Serialize, Deserialize)]
pub struct RegisterTaskInstanceRequest {
    pub task_index: u64,
}

/// Response for RegisterTaskInstance is raw binary (rmp_serde bytes of inputs) with metadata
/// in headers: `X-Task-Instance-Id`.
/// (No JSON struct needed for the response body.)

#[derive(Serialize, Deserialize)]
pub struct SubmitTaskResultResponse {
    pub job_state: String,
}

/// Headers used for binary transport.
pub const HEADER_TASK_INSTANCE_ID: &str = "x-task-instance-id";
pub const HEADER_TASK_INDEX: &str = "x-task-index";
pub const HEADER_SUCCESS: &str = "x-success";
pub const HEADER_ERROR_MESSAGE: &str = "x-error-message";

// =============================================================================
// Job state helpers
// =============================================================================

pub fn job_state_to_string(state: JobState) -> String {
    match state {
        JobState::Ready => "ready",
        JobState::Running => "running",
        JobState::CommitReady => "commit_ready",
        JobState::CleanupReady => "cleanup_ready",
        JobState::Succeeded => "succeeded",
        JobState::Failed => "failed",
        JobState::Cancelled => "cancelled",
    }
    .to_owned()
}

pub fn job_state_is_terminal(state: &str) -> bool {
    matches!(state, "succeeded" | "failed" | "cancelled")
}

// =============================================================================
// Compression (client-side only — server stores compressed bytes as-is)
// =============================================================================

/// Whether to compress payloads stored in the cache. Compression and decompression happen
/// on the client side only. The server stores and serves compressed bytes opaquely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Compression {
    None,
    Zstd,
}

impl std::fmt::Display for Compression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Zstd => write!(f, "zstd"),
        }
    }
}

/// Compress raw bytes. Returns the input unchanged if compression is `None`.
pub fn compress_bytes(data: &[u8], compression: Compression) -> Vec<u8> {
    match compression {
        Compression::None => data.to_vec(),
        Compression::Zstd => {
            zstd::encode_all(std::io::Cursor::new(data), 3)
                .expect("zstd compression should succeed")
        }
    }
}

/// Decompress bytes. Returns the input unchanged if compression is `None`.
pub fn decompress_bytes(data: &[u8], compression: Compression) -> Vec<u8> {
    match compression {
        Compression::None => data.to_vec(),
        Compression::Zstd => {
            zstd::decode_all(std::io::Cursor::new(data))
                .expect("zstd decompression should succeed")
        }
    }
}


// =============================================================================
// Graph builders (ported from spider-storage tests)
// =============================================================================

fn bytes_type() -> DataTypeDescriptor {
    DataTypeDescriptor::Value(ValueTypeDescriptor::Bytes(BytesTypeDescriptor {}))
}

pub fn make_1kb_payload() -> Vec<u8> {
    vec![0xab_u8; 1024]
}

/// Generate a payload of `size` bytes filled with random data.
pub fn make_random_payload(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let mut buf = vec![0u8; size];
    rng.fill(buf.as_mut_slice());
    buf
}

pub fn build_flat_graph(
    num_tasks: usize,
    num_inputs_per_task: usize,
    num_outputs_per_task: usize,
    input_size: usize,
    compression: Compression,
) -> (CoreTaskGraph, Vec<TaskInput>) {
    let mut graph = CoreTaskGraph::default();
    for i in 0..num_tasks {
        graph
            .insert_task(TaskDescriptor {
                tdl_package: "pkg".into(),
                tdl_function: format!("fn_{i}"),
                inputs: vec![bytes_type(); num_inputs_per_task],
                outputs: vec![bytes_type(); num_outputs_per_task],
                input_sources: None,
                execution_policy: ExecutionPolicy::default(),
            })
            .expect("flat graph task insertion should succeed");
    }
    // Job inputs are pre-compressed so the cache stores compressed bytes directly.
    let job_inputs: Vec<TaskInput> = (0..num_tasks * num_inputs_per_task)
        .map(|_| TaskInput::ValuePayload(compress_bytes(&make_random_payload(input_size), compression)))
        .collect();
    (graph, job_inputs)
}

pub fn build_neural_net_graph(
    num_layers: usize,
    width: usize,
    fan_in: usize,
    input_size: usize,
    compression: Compression,
) -> (CoreTaskGraph, Vec<TaskInput>, Vec<Vec<TaskIndex>>) {
    let mut graph = CoreTaskGraph::default();
    let mut layers: Vec<Vec<TaskIndex>> = Vec::with_capacity(num_layers);

    let mut layer_0 = Vec::with_capacity(width);
    for i in 0..width {
        let idx = graph
            .insert_task(TaskDescriptor {
                tdl_package: "pkg".into(),
                tdl_function: format!("L0_{i}"),
                inputs: vec![bytes_type(); fan_in],
                outputs: vec![bytes_type()],
                input_sources: None,
                execution_policy: ExecutionPolicy::default(),
            })
            .expect("neural net layer 0 task insertion should succeed");
        layer_0.push(idx);
    }
    layers.push(layer_0);

    let half = fan_in / 2;
    for layer_idx in 1..num_layers {
        let prev_layer = &layers[layer_idx - 1];
        let mut current_layer = Vec::with_capacity(width);

        for p in 0..width {
            let input_sources: Vec<TaskInputOutputIndex> = (0..fan_in)
                .map(|k| {
                    let src_pos = (p + width - half + k) % width;
                    TaskInputOutputIndex {
                        task_idx: prev_layer[src_pos],
                        position: 0,
                    }
                })
                .collect();

            let idx = graph
                .insert_task(TaskDescriptor {
                    tdl_package: "pkg".into(),
                    tdl_function: format!("L{layer_idx}_{p}"),
                    inputs: vec![bytes_type(); fan_in],
                    outputs: vec![bytes_type()],
                    input_sources: Some(input_sources),
                    execution_policy: ExecutionPolicy::default(),
                })
                .expect("neural net layer task insertion should succeed");
            current_layer.push(idx);
        }
        layers.push(current_layer);
    }

    // Job inputs are pre-compressed so the cache stores compressed bytes directly.
    let job_inputs: Vec<TaskInput> = (0..width * fan_in)
        .map(|_| TaskInput::ValuePayload(compress_bytes(&make_random_payload(input_size), compression)))
        .collect();

    (graph, job_inputs, layers)
}

// =============================================================================
// Mock implementations (for server-side use)
// =============================================================================

pub struct MockDb;

#[async_trait]
impl spider_storage::db::InternalJobOrchestration for MockDb {
    async fn set_state(&self, _job_id: JobId, _state: JobState) -> Result<(), DbError> {
        Ok(())
    }

    async fn commit_outputs(
        &self,
        _job_id: JobId,
        _job_outputs: Vec<TaskOutput>,
    ) -> Result<JobState, DbError> {
        Ok(JobState::Succeeded)
    }

    async fn cancel(&self, _job_id: JobId) -> Result<JobState, DbError> {
        Ok(JobState::Cancelled)
    }

    async fn fail(&self, _job_id: JobId, _error_message: String) -> Result<(), DbError> {
        Ok(())
    }

    async fn delete_expired_terminated_jobs(
        &self,
        _expire_after: std::time::Duration,
    ) -> Result<Vec<JobId>, DbError> {
        Ok(Vec::new())
    }
}

pub struct MockInstancePool {
    next_id: AtomicU64,
}

impl MockInstancePool {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
        }
    }
}

impl Default for MockInstancePool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TaskInstancePoolConnector for MockInstancePool {
    fn get_next_available_task_instance_id(&self) -> TaskInstanceId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    async fn register_task_instance(
        &self,
        _task_instance_id: TaskInstanceId,
        _task: SharedTaskControlBlock,
    ) -> Result<(), InternalError> {
        Ok(())
    }

    async fn register_termination_task_instance(
        &self,
        _task_instance_id: TaskInstanceId,
        _termination_task: SharedTerminationTaskControlBlock,
    ) -> Result<(), InternalError> {
        Ok(())
    }
}

// =============================================================================
// Stats utilities
// =============================================================================

#[derive(Debug)]
pub struct TaskLatency {
    pub task_index: TaskIndex,
    pub register_duration: Duration,
    pub submit_duration: Duration,
}

pub fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn avg_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn report_stats(
    test_name: &str,
    num_workers: usize,
    compression: Compression,
    total_execution: Duration,
    latencies: &[TaskLatency],
) {
    let to_ms = |d: &Duration| d.as_secs_f64() * 1000.0;

    let mut register_ms: Vec<f64> = latencies.iter().map(|l| to_ms(&l.register_duration)).collect();
    let mut submit_ms: Vec<f64> = latencies.iter().map(|l| to_ms(&l.submit_duration)).collect();
    register_ms.sort_by(|a, b| a.partial_cmp(b).expect("latencies should be comparable"));
    submit_ms.sort_by(|a, b| a.partial_cmp(b).expect("latencies should be comparable"));

    eprintln!();
    eprintln!("=== {test_name} ({num_workers} workers, REST, compression={compression}) ===");
    eprintln!(
        "  total_execution:             {:>10.2} ms",
        to_ms(&total_execution)
    );
    eprintln!("  tasks_completed:              {:>10}", latencies.len());
    eprintln!("  --- RegisterTaskInstance (REST) ---");
    eprintln!("    avg:                         {:>10.3} ms", avg_of(&register_ms));
    eprintln!("    p50:                         {:>10.3} ms", percentile(&register_ms, 50.0));
    eprintln!("    p95:                         {:>10.3} ms", percentile(&register_ms, 95.0));
    eprintln!("    p99:                         {:>10.3} ms", percentile(&register_ms, 99.0));
    eprintln!("  --- SubmitTaskResult (REST) ---");
    eprintln!("    avg:                         {:>10.3} ms", avg_of(&submit_ms));
    eprintln!("    p50:                         {:>10.3} ms", percentile(&submit_ms, 50.0));
    eprintln!("    p95:                         {:>10.3} ms", percentile(&submit_ms, 95.0));
    eprintln!("    p99:                         {:>10.3} ms", percentile(&submit_ms, 99.0));
    eprintln!();
}
