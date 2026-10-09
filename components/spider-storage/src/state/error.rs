use spider_core::types::id::JobId;
use spider_core::types::id::SessionId;
use spider_tdl::error::TdlError;

use crate::cache::error::CacheError;
use crate::cache::error::ChildPreparationError;
use crate::db::DbError;

/// Errors that can occur during storage server operations.
#[derive(thiserror::Error, Debug)]
pub enum StorageServerError {
    #[error(transparent)]
    Cache(CacheError),

    #[error(transparent)]
    Db(#[from] DbError),

    #[error(transparent)]
    Tdl(#[from] TdlError),

    #[error("current storage session is {0}")]
    StaleSession(SessionId),

    #[error("server is shutting down: {0}")]
    Stopping(String),

    #[error("job not found in cache: {0:?}")]
    JobNotFound(JobId),

    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("serde: {0}")]
    Serde(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl From<CacheError> for StorageServerError {
    fn from(error: CacheError) -> Self {
        match error {
            CacheError::ServiceStopping => Self::Stopping(error.to_string()),
            error => Self::Cache(error),
        }
    }
}

/// Enums for the reasons why a task's success report is turned into a failure of the task.
#[derive(thiserror::Error, Debug)]
pub enum ReportRejection {
    #[error("task outputs of {size} bytes exceed the limit of {limit} bytes")]
    OutputsTooLarge { size: u64, limit: u64 },

    #[error("child jobs at depth {depth} exceed the maximum depth {limit}")]
    TooDeep { depth: u32, limit: u32 },

    #[error("{count} child jobs exceed the limit of {limit} per report")]
    TooManyChildren { count: usize, limit: usize },

    #[error("child job {index} of {size} bytes exceeds the limit of {limit} bytes")]
    ChildTooLarge { index: usize, size: u64, limit: u64 },

    #[error("child jobs of {size} bytes exceed the limit of {limit} bytes per report")]
    ChildrenTooLarge { size: u64, limit: u64 },

    #[error(
        "child jobs would bring the job to {count} child jobs of {bytes} bytes, over the limits \
         of {count_limit} child jobs and {bytes_limit} bytes per job"
    )]
    JobChildrenTooLarge {
        count: usize,
        bytes: u64,
        count_limit: usize,
        bytes_limit: u64,
    },

    #[error("child job {index} is invalid: {source}")]
    InvalidChild {
        index: usize,
        #[source]
        source: ChildPreparationError,
    },
}
