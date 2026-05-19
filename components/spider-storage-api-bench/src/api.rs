use std::{fmt::Display, net::IpAddr, str::FromStr, time::Duration};

use serde::{Deserialize, Serialize};
use spider_core::{
    job::JobState,
    task::{TaskIndex, TdlContext, TimeoutPolicy},
    types::{
        id::{ExecutionManagerId, JobId, ResourceGroupId, TaskInstanceId},
        io::ExecutionContext,
    },
};
use spider_storage::{cache::TaskId, ready_queue::ReadyQueueEntry, state::StorageServerError};

/// Serde adapters that JSON-encode byte payloads as base64 strings.
///
/// Default `serde_json` encodes `Vec<u8>` as a JSON array of decimal numbers (`[123, 45, ...]`),
/// expanding each byte to roughly 3.5 characters. Base64 expands each byte to 4/3 ≈ 1.33 — about
/// 2.6× smaller for large payloads — and keeps each blob as a single JSON string. The gRPC
/// transport bypasses serde entirely (it uses hand-written proto converters in `grpc.rs`), so
/// these adapters only affect the REST wire format.
mod base64_serde {
    use base64::{Engine, engine::general_purpose::STANDARD};
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub mod bytes {
        use super::{Deserialize, Deserializer, Engine, Error, STANDARD, Serializer};

        pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_str(&STANDARD.encode(bytes))
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Vec<u8>, D::Error> {
            let encoded = String::deserialize(deserializer)?;
            STANDARD
                .decode(encoded.as_bytes())
                .map_err(D::Error::custom)
        }
    }

    pub mod vec_bytes {
        use serde::{de::SeqAccess, ser::SerializeSeq};

        use super::{Deserializer, Engine, Error, STANDARD, Serializer};

        pub fn serialize<S: Serializer>(
            value: &[Vec<u8>],
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            let mut seq = serializer.serialize_seq(Some(value.len()))?;
            for bytes in value {
                seq.serialize_element(&STANDARD.encode(bytes))?;
            }
            seq.end()
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Vec<Vec<u8>>, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Vec<Vec<u8>>;

                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("a sequence of base64-encoded byte strings")
                }

                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                    while let Some(encoded) = seq.next_element::<String>()? {
                        out.push(
                            STANDARD
                                .decode(encoded.as_bytes())
                                .map_err(A::Error::custom)?,
                        );
                    }
                    Ok(out)
                }
            }
            deserializer.deserialize_seq(Visitor)
        }
    }
}

/// Result type used by transport-neutral API helpers.
pub type ApiResult<ResponseType> = Result<ResponseType, ApiError>;

/// Error envelope shared by REST, gRPC, and benchmark clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiError {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
}

impl ApiError {
    #[must_use]
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::BadRequest,
            message: message.into(),
            retryable: false,
        }
    }

    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::Internal,
            message: message.into(),
            retryable: true,
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for ApiError {}

impl From<StorageServerError> for ApiError {
    fn from(error: StorageServerError) -> Self {
        let message = error.to_string();
        match error {
            StorageServerError::BadRequest(_) => Self {
                code: ErrorCode::BadRequest,
                message,
                retryable: false,
            },
            StorageServerError::JobNotFound(_) => Self {
                code: ErrorCode::NotFound,
                message,
                retryable: false,
            },
            StorageServerError::StaleSession | StorageServerError::JobAlreadyExists(_) => Self {
                code: ErrorCode::Conflict,
                message,
                retryable: false,
            },
            StorageServerError::Stopping(_) => Self {
                code: ErrorCode::Unavailable,
                message,
                retryable: true,
            },
            StorageServerError::Cache(_)
            | StorageServerError::Db(_)
            | StorageServerError::Task(_)
            | StorageServerError::Tdl(_) => Self {
                code: ErrorCode::Internal,
                message,
                retryable: true,
            },
        }
    }
}

/// Stable error code used across transports.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    BadRequest,
    Conflict,
    Internal,
    NotFound,
    Unavailable,
}

/// Empty JSON/protobuf-compatible response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct EmptyResponse {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct GetSessionRequest {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionResponse {
    pub session_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct StartMetricsSessionRequest {
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StartMetricsSessionResponse {
    pub metrics_session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EndMetricsSessionRequest {
    pub metrics_session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AddResourceGroupRequest {
    pub external_id: String,
    #[serde(with = "base64_serde::bytes")]
    pub password: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerifyResourceGroupRequest {
    pub resource_group_id: String,
    #[serde(with = "base64_serde::bytes")]
    pub password: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceGroupResponse {
    pub resource_group_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterJobRequest {
    pub resource_group_id: String,
    pub serialized_task_graph: String,
    #[serde(with = "base64_serde::bytes")]
    pub serialized_inputs: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobIdRequest {
    pub job_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobIdResponse {
    pub job_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobStateResponse {
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobOutputsResponse {
    #[serde(with = "base64_serde::vec_bytes")]
    pub outputs: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobErrorResponse {
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PollReadyTasksRequest {
    pub max_tasks: usize,
    pub wait_ms: u64,
}

impl PollReadyTasksRequest {
    #[must_use]
    pub const fn wait(&self) -> Duration {
        Duration::from_millis(self.wait_ms)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadyTaskEntryDto {
    pub resource_group_id: String,
    pub job_id: String,
    pub task_index: TaskIndex,
}

impl From<ReadyQueueEntry<TaskIndex>> for ReadyTaskEntryDto {
    fn from(entry: ReadyQueueEntry<TaskIndex>) -> Self {
        Self {
            resource_group_id: format_id(&entry.resource_group_id),
            job_id: format_id(&entry.job_id),
            task_index: entry.task_kind,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadyTasksResponse {
    pub tasks: Vec<ReadyTaskEntryDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminationTaskEntryDto {
    pub resource_group_id: String,
    pub job_id: String,
}

impl<TaskKind> From<ReadyQueueEntry<TaskKind>> for TerminationTaskEntryDto {
    fn from(entry: ReadyQueueEntry<TaskKind>) -> Self {
        Self {
            resource_group_id: format_id(&entry.resource_group_id),
            job_id: format_id(&entry.job_id),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminationTasksResponse {
    pub tasks: Vec<TerminationTaskEntryDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskIdDto {
    Index { task_index: TaskIndex },
    Commit,
    Cleanup,
}

impl From<TaskId> for TaskIdDto {
    fn from(task_id: TaskId) -> Self {
        match task_id {
            TaskId::Index(task_index) => Self::Index { task_index },
            TaskId::Commit => Self::Commit,
            TaskId::Cleanup => Self::Cleanup,
        }
    }
}

impl From<TaskIdDto> for TaskId {
    fn from(task_id: TaskIdDto) -> Self {
        match task_id {
            TaskIdDto::Index { task_index } => Self::Index(task_index),
            TaskIdDto::Commit => Self::Commit,
            TaskIdDto::Cleanup => Self::Cleanup,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CreateTaskInstanceRequest {
    pub session_id: u64,
    pub job_id: String,
    pub task_id: TaskIdDto,
    pub execution_manager_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionContextResponse {
    pub task_instance_id: TaskInstanceId,
    pub tdl_context: TdlContext,
    pub timeout_policy: TimeoutPolicy,
    #[serde(with = "base64_serde::bytes")]
    pub serialized_inputs: Vec<u8>,
}

impl From<ExecutionContext> for ExecutionContextResponse {
    fn from(context: ExecutionContext) -> Self {
        Self {
            task_instance_id: context.task_instance_id,
            tdl_context: context.tdl_context,
            timeout_policy: context.timeout_policy,
            serialized_inputs: context.serialized_inputs,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SucceedTaskInstanceRequest {
    pub session_id: u64,
    pub job_id: String,
    pub task_instance_id: TaskInstanceId,
    pub task_index: TaskIndex,
    #[serde(with = "base64_serde::bytes")]
    pub serialized_outputs: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SucceedTerminationTaskInstanceRequest {
    pub session_id: u64,
    pub job_id: String,
    pub task_instance_id: TaskInstanceId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailTaskInstanceRequest {
    pub session_id: u64,
    pub job_id: String,
    pub task_instance_id: TaskInstanceId,
    pub task_id: TaskIdDto,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterExecutionManagerRequest {
    pub ip_address: IpAddr,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionManagerRequest {
    pub execution_manager_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionManagerResponse {
    pub execution_manager_id: String,
}

#[must_use]
pub fn format_id<IdType: Display>(id: &IdType) -> String {
    id.to_string()
}

pub(crate) fn parse_resource_group_id(value: &str) -> ApiResult<ResourceGroupId> {
    parse_id(value)
}

pub(crate) fn parse_job_id(value: &str) -> ApiResult<JobId> {
    parse_id(value)
}

pub(crate) fn parse_execution_manager_id(value: &str) -> ApiResult<ExecutionManagerId> {
    parse_id(value)
}

fn parse_id<IdType>(value: &str) -> ApiResult<IdType>
where
    IdType: FromStr,
    IdType::Err: Display, {
    IdType::from_str(value).map_err(|e| ApiError::bad_request(format!("invalid id `{value}`: {e}")))
}

#[must_use]
pub fn job_state_response(state: JobState) -> JobStateResponse {
    JobStateResponse {
        state: state.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use spider_core::types::id::JobId;
    use spider_storage::cache::TaskId;

    use super::{TaskIdDto, format_id, parse_job_id};

    #[test]
    fn id_formats_and_parses_as_uuid() {
        let id = JobId::new();
        let encoded = format_id(&id);
        let decoded = parse_job_id(&encoded).expect("formatted id should parse");
        assert_eq!(id, decoded);
    }

    #[test]
    fn task_id_dto_roundtrips_index_commit_and_cleanup() {
        for task_id in [TaskId::Index(7), TaskId::Commit, TaskId::Cleanup] {
            let dto = TaskIdDto::from(task_id);
            let decoded = TaskId::from(dto);
            assert_eq!(task_id, decoded);
        }
    }
}
