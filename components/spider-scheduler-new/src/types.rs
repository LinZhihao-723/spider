//! The data types the prototype scheduler exchanges with the storage layer and its own components.

pub use spider_core::task::TaskIndex;
pub use spider_core::types::id::JobId;
pub use spider_core::types::id::ResourceGroupId;
pub use spider_core::types::id::SessionId;
pub use spider_core::types::id::TaskAssignmentId;
pub use spider_core::types::id::TaskId;
pub use spider_core::types::scheduler::TaskAssignment;

/// A ready task drained from the storage-owned inbound queue.
///
/// The storage client flattens storage's three ready lanes (regular, commit, and cleanup tasks)
/// into this uniform entry, resolving each to its [`TaskId`] so the core can treat every ready task
/// identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InboundEntry {
    /// The resource group that owns the job.
    pub resource_group_id: ResourceGroupId,

    /// The job the task belongs to.
    pub job_id: JobId,

    /// The ready task.
    pub task_id: TaskId,
}

/// The way a job reached its terminal state, determining which task finalizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinalizeKind {
    /// The job completed and must run its commit task.
    Commit,

    /// The job was cancelled or failed and must run its cleanup task.
    Cleanup,
}

impl From<FinalizeKind> for TaskId {
    fn from(kind: FinalizeKind) -> Self {
        match kind {
            FinalizeKind::Commit => Self::Commit,
            FinalizeKind::Cleanup => Self::Cleanup,
        }
    }
}
