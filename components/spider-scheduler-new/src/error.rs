//! The error types used in this crate.

/// Errors returned by [`crate::storage_client::SchedulerStorageClient`] operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageClientError {
    /// The storage server returned an error.
    #[error("storage server error: {0}")]
    Server(String),

    /// The storage transport failed or returned malformed data.
    #[error("storage transport error: {0}")]
    Transport(String),
}

/// Errors returned by the prototype scheduler core.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Forwarded from the storage client.
    #[error(transparent)]
    Storage(#[from] StorageClientError),

    /// A queue the core publishes into is closed, so nothing it publishes can ever be taken. No
    /// later tick can recover from it.
    #[error("fatal publication failure: {0}")]
    FatalPublication(MakeAssignmentError),

    /// An invariant of the core was violated.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Errors returned by [`crate::scheduling_unit::RgSchedulingUnit::try_make_assignment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MakeAssignmentError {
    /// The resource group has nothing left to schedule this tick.
    #[error("no task to assign")]
    NoTask,

    /// The resource group's queue occupancy has reached the admission threshold.
    #[error("dispatch queue is full")]
    DispatchQueueFull,

    /// The resource group's dispatch queue is closed and can no longer accept assignments.
    #[error("dispatch queue is closed")]
    DispatchQueueClosed,

    /// The global dispatch queue is closed and can no longer carry hints.
    #[error("broadcast queue is closed")]
    BroadcastQueueClosed,
}

/// Errors returned by the prototype's test harness.
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// A gRPC transport operation failed.
    #[error("harness transport error: {0}")]
    Transport(String),

    /// The harness server could not bind a local port.
    #[error("harness bind error: {0}")]
    Bind(String),

    /// The harness was given a configuration it cannot run.
    #[error("invalid harness configuration: {0}")]
    Config(String),

    /// A protobuf message could not be converted to its core representation.
    #[error("malformed wire message: {0}")]
    InvalidMessage(String),

    /// A component of the harness itself failed.
    #[error("harness internal error: {0}")]
    Internal(String),
}
