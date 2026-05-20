use spider_core::{task::TaskGraph, types::io::TaskInput};
use spider_tdl::wire::TaskInputsSerializer;

use super::error::InternalError;

/// zstd compression level used when framing inputs locally via [`ValidatedJobSubmission::create`].
const INPUTS_ZSTD_LEVEL: i32 = 3;

/// A validated wrapper around a task graph and the corresponding job inputs.
///
/// Inputs are stored as an opaque zstd-compressed TDL-framed byte blob. Callers feed the blob
/// straight to the storage layer (which writes it to the DB column verbatim) and only pay the
/// decompression + unframe cost at JCB build time.
///
/// At construction time the type guarantees:
///
/// * The task graph contains at least one task. The input count is **not** validated here — that
///   check is deferred to the JCB build path, since the blob is opaque until then.
#[derive(Debug)]
pub struct ValidatedJobSubmission {
    task_graph: TaskGraph,
    inputs_blob: Vec<u8>,
}

impl ValidatedJobSubmission {
    /// Creates a new validated job submission from an already-framed-and-compressed inputs blob.
    ///
    /// Fast path used by the storage service: the bytes received over the wire are zstd-compressed
    /// TDL-framed inputs, so we keep them as-is until JCB build time.
    ///
    /// # Errors
    ///
    /// Returns [`InternalError::TaskGraphEmpty`] if the task graph contains no tasks.
    pub fn create_from_compressed_bytes(
        task_graph: TaskGraph,
        inputs_blob: Vec<u8>,
    ) -> Result<Self, InternalError> {
        if task_graph.get_num_tasks() == 0 {
            return Err(InternalError::TaskGraphEmpty);
        }
        Ok(Self {
            task_graph,
            inputs_blob,
        })
    }

    /// Creates a new validated job submission from in-memory `TaskInput`s.
    ///
    /// Convenience path used by tests and any caller that hasn't already framed and compressed
    /// the inputs. Validates the input count against the task graph (since the inputs are
    /// already parsed here), then frames + zstd-compresses them so the on-disk representation
    /// matches the fast path.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`InternalError::TaskGraphEmpty`] if the task graph contains no tasks.
    /// * [`InternalError::TaskGraphInputSizeMismatch`] if the number of inputs does not match the
    ///   number of graph inputs.
    pub fn create(task_graph: TaskGraph, inputs: Vec<TaskInput>) -> Result<Self, InternalError> {
        let num_tasks = task_graph.get_num_tasks();
        if num_tasks == 0 {
            return Err(InternalError::TaskGraphEmpty);
        }
        let expected_num_inputs = task_graph.get_task_graph_input_indices().len();
        let actual_num_inputs = inputs.len();
        if expected_num_inputs != actual_num_inputs {
            return Err(InternalError::TaskGraphInputSizeMismatch {
                expected: expected_num_inputs,
                actual: actual_num_inputs,
            });
        }
        let mut serializer = TaskInputsSerializer::new();
        for input in inputs {
            serializer
                .append(input)
                .map_err(|e| InternalError::TaskGraphCorrupted(e.to_string()))?;
        }
        let framed = serializer.release();
        let inputs_blob = zstd::encode_all(framed.as_slice(), INPUTS_ZSTD_LEVEL)
            .map_err(|e| InternalError::TaskGraphCorrupted(e.to_string()))?;
        Ok(Self {
            task_graph,
            inputs_blob,
        })
    }

    /// # Returns
    ///
    /// A reference to the validated task graph.
    #[must_use]
    pub const fn task_graph(&self) -> &TaskGraph {
        &self.task_graph
    }

    /// # Returns
    ///
    /// A reference to the zstd-compressed TDL-framed inputs blob. Callers writing the blob to
    /// persistent storage should use this directly; callers needing parsed `TaskInput`s should go
    /// through [`Self::into_parts`] and decompress.
    #[must_use]
    pub fn inputs_blob(&self) -> &[u8] {
        &self.inputs_blob
    }

    /// Consumes the wrapper and returns the owned task graph and inputs blob.
    ///
    /// # Returns
    ///
    /// A tuple of `(task_graph, inputs_blob)`. The blob is still zstd-compressed TDL frames —
    /// the JCB build path is responsible for decompressing and unframing before use.
    #[must_use]
    pub fn into_parts(self) -> (TaskGraph, Vec<u8>) {
        (self.task_graph, self.inputs_blob)
    }
}

#[cfg(test)]
mod tests {
    use spider_core::{
        task::{
            DataTypeDescriptor,
            ExecutionPolicy,
            TaskDescriptor,
            TaskGraph as SubmittedTaskGraph,
            TdlContext,
            ValueTypeDescriptor,
        },
        types::io::TaskInput,
    };
    use spider_tdl::wire::unframe;

    use super::{super::error::InternalError, *};

    fn create_single_input_task_graph() -> SubmittedTaskGraph {
        let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
        let mut graph =
            SubmittedTaskGraph::new(None, None).expect("task graph creation should succeed");
        graph
            .insert_task(TaskDescriptor {
                tdl_context: TdlContext {
                    package: "test_pkg".to_owned(),
                    task_func: "test_fn".to_owned(),
                },
                execution_policy: Some(ExecutionPolicy::default()),
                inputs: vec![bytes_type],
                outputs: vec![],
                input_sources: None,
            })
            .expect("task insertion should succeed");
        graph
    }

    #[test]
    fn valid_job_submission_succeeds() {
        let graph = create_single_input_task_graph();
        let inputs = vec![TaskInput::ValuePayload(vec![1u8; 4])];
        let result = ValidatedJobSubmission::create(graph, inputs);
        assert!(result.is_ok(), "valid submission should succeed");
    }

    #[test]
    fn empty_task_graph_fails() {
        let graph =
            SubmittedTaskGraph::new(None, None).expect("task graph creation should succeed");
        let inputs = vec![];
        let result = ValidatedJobSubmission::create(graph, inputs);
        assert!(
            matches!(result, Err(InternalError::TaskGraphEmpty)),
            "empty task graph should return EmptyTaskGraph"
        );
    }

    #[test]
    fn mismatched_input_count_fails() {
        let graph = create_single_input_task_graph();
        let inputs = vec![];
        let result = ValidatedJobSubmission::create(graph, inputs);
        assert!(
            matches!(
                result,
                Err(InternalError::TaskGraphInputSizeMismatch {
                    expected: 1,
                    actual: 0
                })
            ),
            "mismatched input count should return TaskGraphInputSizeMismatch"
        );
    }

    #[test]
    fn into_parts_returns_compressed_blob_that_round_trips() {
        let graph = create_single_input_task_graph();
        let inputs = vec![TaskInput::ValuePayload(vec![1u8; 4])];
        let submission =
            ValidatedJobSubmission::create(graph, inputs).expect("submission should be valid");
        let (graph, inputs_blob) = submission.into_parts();
        assert_eq!(graph.get_num_tasks(), 1, "task graph should have 1 task");

        let framed = zstd::decode_all(inputs_blob.as_slice()).expect("zstd decode");
        let payloads = unframe(&framed).expect("unframe");
        assert_eq!(payloads.len(), 1, "round trip should yield 1 input");
        assert_eq!(payloads[0].as_slice(), &[1u8; 4]);
    }
}
