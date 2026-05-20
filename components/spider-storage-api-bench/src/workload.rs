use serde::{Deserialize, Serialize};
use spider_core::{
    task::{
        DataTypeDescriptor,
        ExecutionPolicy,
        TaskDescriptor,
        TaskGraph,
        TaskInputOutputIndex,
        TdlContext,
        ValueTypeDescriptor,
    },
    types::io::TaskInput,
};
use spider_tdl::wire::TaskInputsSerializer;

/// Workload family selected by the benchmark client.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum WorkloadKind {
    Flat,
    Deep,
    Mixed,
}

impl std::str::FromStr for WorkloadKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "flat" => Ok(Self::Flat),
            "deep" => Ok(Self::Deep),
            "mixed" => Ok(Self::Mixed),
            _ => anyhow::bail!("unknown workload `{value}`"),
        }
    }
}

/// Serialized job payload ready for [`spider_storage::state::ServiceState::register_job`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobPayload {
    pub workload_kind: WorkloadKind,
    pub serialized_task_graph: String,
    pub serialized_inputs: Vec<u8>,
    pub task_count: usize,
}

/// Builds the configured sequence of benchmark jobs.
///
/// # Returns
///
/// A vector of serialized benchmark job payloads on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`build_flat_job`] fails.
/// * [`build_deep_job`] fails.
pub fn build_jobs(
    workload_kind: WorkloadKind,
    job_count: usize,
    task_count: usize,
    payload_bytes: usize,
    flat_percent: u8,
) -> anyhow::Result<Vec<JobPayload>> {
    if flat_percent > 100 {
        anyhow::bail!("flat_percent must be between 0 and 100");
    }

    (0..job_count)
        .map(|job_index| {
            let kind = match workload_kind {
                WorkloadKind::Flat => WorkloadKind::Flat,
                WorkloadKind::Deep => WorkloadKind::Deep,
                WorkloadKind::Mixed => {
                    if (job_index * 100 / job_count) < usize::from(flat_percent) {
                        WorkloadKind::Flat
                    } else {
                        WorkloadKind::Deep
                    }
                }
            };
            match kind {
                WorkloadKind::Flat => build_flat_job(task_count, payload_bytes),
                WorkloadKind::Deep => build_deep_job(task_count, payload_bytes),
                WorkloadKind::Mixed => unreachable!("mixed is expanded before job construction"),
            }
        })
        .collect()
}

/// Builds a flat task graph with independent tasks.
///
/// # Returns
///
/// A serialized benchmark job payload on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`TaskGraph::new`]'s return values on failure.
/// * Forwards [`TaskGraph::insert_task`]'s return values on failure.
/// * Forwards [`serialize_job`]'s return values on failure.
pub fn build_flat_job(task_count: usize, payload_bytes: usize) -> anyhow::Result<JobPayload> {
    let mut graph = TaskGraph::new(None, None)?;
    let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
    for _ in 0..task_count {
        graph.insert_task(TaskDescriptor {
            tdl_context: task_context("flat_task"),
            execution_policy: Some(execution_policy()),
            inputs: vec![bytes_type.clone()],
            outputs: vec![bytes_type.clone()],
            input_sources: None,
        })?;
    }
    serialize_job(WorkloadKind::Flat, &graph, task_count, payload_bytes)
}

/// Builds a deep task graph with a single dependency chain.
///
/// # Returns
///
/// A serialized benchmark job payload on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`TaskGraph::new`]'s return values on failure.
/// * Forwards [`TaskGraph::insert_task`]'s return values on failure.
/// * Forwards [`serialize_job`]'s return values on failure.
pub fn build_deep_job(task_count: usize, payload_bytes: usize) -> anyhow::Result<JobPayload> {
    let mut graph = TaskGraph::new(None, None)?;
    let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
    let mut previous_task = None;
    for _ in 0..task_count {
        let input_sources = previous_task.map(|task_idx| {
            vec![TaskInputOutputIndex {
                task_idx,
                position: 0,
            }]
        });
        let task_idx = graph.insert_task(TaskDescriptor {
            tdl_context: task_context("deep_task"),
            execution_policy: Some(execution_policy()),
            inputs: vec![bytes_type.clone()],
            outputs: vec![bytes_type.clone()],
            input_sources,
        })?;
        previous_task = Some(task_idx);
    }
    serialize_job(WorkloadKind::Deep, &graph, 1, payload_bytes)
}

fn serialize_job(
    workload_kind: WorkloadKind,
    graph: &TaskGraph,
    num_inputs: usize,
    payload_bytes: usize,
) -> anyhow::Result<JobPayload> {
    let mut serializer = TaskInputsSerializer::new();
    for _ in 0..num_inputs {
        serializer.append(TaskInput::ValuePayload(vec![0; payload_bytes]))?;
    }
    let framed = serializer.release();
    // Compress at the client so the server can pass the bytes through to the DB column verbatim
    // and only pay the decompression cost at JCB build time.
    let serialized_inputs = zstd::encode_all(framed.as_slice(), INPUTS_ZSTD_LEVEL)?;
    Ok(JobPayload {
        workload_kind,
        serialized_task_graph: graph.to_json()?,
        serialized_inputs,
        task_count: graph.get_num_tasks(),
    })
}

/// zstd level used for the client-side compression of the framed inputs blob. Matches the level
/// the server used to apply to the same payload pre-change so the comparison stays
/// apples-to-apples.
const INPUTS_ZSTD_LEVEL: i32 = 3;

fn task_context(task_func: &str) -> TdlContext {
    TdlContext {
        package: "storage_api_bench".to_owned(),
        task_func: task_func.to_owned(),
    }
}

const fn execution_policy() -> ExecutionPolicy {
    ExecutionPolicy {
        max_num_retry: 0,
        max_num_instances: 1,
        timeout_policy: spider_core::task::TimeoutPolicy {
            soft_timeout_ms: 150_000,
            hard_timeout_ms: 300_000,
        },
    }
}

#[cfg(test)]
mod tests {
    use spider_core::task::TaskGraph;

    use super::{WorkloadKind, build_deep_job, build_flat_job, build_jobs};

    #[test]
    fn flat_job_has_one_graph_input_per_task() -> anyhow::Result<()> {
        let payload = build_flat_job(4, 16)?;
        let graph = TaskGraph::from_json(&payload.serialized_task_graph)?;
        assert_eq!(4, graph.get_num_tasks());
        assert_eq!(4, graph.get_task_graph_input_indices().len());
        assert_eq!(WorkloadKind::Flat, payload.workload_kind);
        Ok(())
    }

    #[test]
    fn deep_job_has_single_graph_input() -> anyhow::Result<()> {
        let payload = build_deep_job(4, 16)?;
        let graph = TaskGraph::from_json(&payload.serialized_task_graph)?;
        assert_eq!(4, graph.get_num_tasks());
        assert_eq!(1, graph.get_task_graph_input_indices().len());
        assert_eq!(WorkloadKind::Deep, payload.workload_kind);
        Ok(())
    }

    #[test]
    fn mixed_workload_uses_flat_percent() -> anyhow::Result<()> {
        let jobs = build_jobs(WorkloadKind::Mixed, 10, 4, 16, 70)?;
        let flat_count = jobs
            .iter()
            .filter(|job| job.workload_kind == WorkloadKind::Flat)
            .count();
        let deep_count = jobs
            .iter()
            .filter(|job| job.workload_kind == WorkloadKind::Deep)
            .count();
        assert_eq!(7, flat_count);
        assert_eq!(3, deep_count);
        Ok(())
    }
}
