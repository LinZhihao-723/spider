use std::fmt::Display;

use spider_core::job::JobState;
use spider_core::types::id::ExecutionManagerId;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use spider_core::types::id::SessionId;

#[derive(thiserror::Error, Debug)]
pub enum DbError {
    #[error("resource group `{0:?}` does not exist")]
    ResourceGroupNotFound(ResourceGroupId),

    #[error("external resource group `{0}` does not exist")]
    ExternalResourceGroupNotFound(String),

    #[error("resource group `{0:?}` already exists")]
    ResourceGroupAlreadyExists(String),

    #[error("resource group `{0:?}` password is incorrect")]
    InvalidPassword(ResourceGroupId),

    #[error("job `{0:?}` does not exist")]
    JobNotFound(JobId),

    #[error("execution manager `{0:?}` is illegal")]
    IllegalExecutionManagerId(ExecutionManagerId),

    #[error("execution manager `{0:?}` is already dead")]
    ExecutionManagerAlreadyDead(ExecutionManagerId),

    #[error("job in state {current}, expect state {expected}")]
    UnexpectedJobState {
        current: JobState,
        expected: ExpectedStates,
    },

    #[error("storage session {mine} is fenced off by the newer session {current}")]
    Fenced { mine: SessionId, current: SessionId },

    #[error("database corrupted: {0}")]
    CorruptedDbState(String),

    #[error("Task graph serialization failure: {0}")]
    TaskGraphSerializationFailure(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Task graph deserialization failure: {0}")]
    TaskGraphDeserializationFailure(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Value serialization failure: {0}")]
    ValueSerializationFailure(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Value deserialization failure: {0}")]
    ValueDeserializationFailure(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("the database rejected a child insert's payload: {0}")]
    PayloadRejected(#[source] sqlx::error::Error),

    #[error("job outputs of {size} serialized bytes exceed the limit of {limit} bytes")]
    OutputsTooLarge {
        job_id: JobId,
        size: u64,
        limit: u64,
    },

    #[cfg(feature = "test-hooks")]
    #[error("injected database failure (deterministic payload error: {deterministic_payload})")]
    Injected { deterministic_payload: bool },

    #[error(transparent)]
    Sql(#[from] sqlx::error::Error),
}

impl DbError {
    /// # Returns
    ///
    /// Whether the error is a payload rejection that the same statement would get again, such as
    /// a packet or a value that is too large.
    #[must_use]
    pub const fn is_deterministic_payload_error(&self) -> bool {
        match self {
            Self::PayloadRejected(_) => true,
            #[cfg(feature = "test-hooks")]
            Self::Injected {
                deterministic_payload,
            } => *deterministic_payload,
            _ => false,
        }
    }

    pub fn task_graph_ser<SerializationError: serde::ser::Error + Send + Sync + 'static>(
        e: SerializationError,
    ) -> Self {
        Self::TaskGraphSerializationFailure(Box::new(e))
    }

    pub fn task_graph_de<DeserializationError: std::error::Error + Send + Sync + 'static>(
        e: DeserializationError,
    ) -> Self {
        Self::TaskGraphDeserializationFailure(Box::new(e))
    }

    pub fn value_ser<SerializationError: serde::ser::Error + Send + Sync + 'static>(
        e: SerializationError,
    ) -> Self {
        Self::ValueSerializationFailure(Box::new(e))
    }

    pub fn value_de<DeserializationError: serde::de::Error + Send + Sync + 'static>(
        e: DeserializationError,
    ) -> Self {
        Self::ValueDeserializationFailure(Box::new(e))
    }
}

#[derive(Debug)]
pub struct ExpectedStates(pub Vec<JobState>);

impl Display for ExpectedStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let states = self
            .0
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "{states}")
    }
}
