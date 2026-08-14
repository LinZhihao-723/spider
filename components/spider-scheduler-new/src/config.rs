//! The tunables of the prototype scheduler core.

use std::num::NonZeroU64;
use std::num::NonZeroUsize;

use serde::Deserialize;
use serde::Serialize;

/// The configuration of the prototype scheduler core.
///
/// The sharing coefficient α of the admission policy is fixed at 1 and is deliberately absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreConfig {
    /// The total dispatch buffer size shared by every resource group.
    pub dispatch_queue_capacity: NonZeroUsize,

    /// The number of active jobs each resource group may hold, applied per group rather than as a
    /// global budget.
    pub active_job_list_capacity: NonZeroUsize,

    /// The maximum number of regular tasks fetched from the inbound queue in a single poll.
    pub ready_task_capacity: NonZeroUsize,

    /// The maximum number of commit-ready jobs fetched from the inbound queue in a single poll.
    pub commit_ready_task_capacity: NonZeroUsize,

    /// The maximum number of cleanup-ready jobs fetched from the inbound queue in a single poll.
    pub cleanup_ready_task_capacity: NonZeroUsize,

    /// The time an inbound queue poll may block on the storage side, in milliseconds.
    pub storage_poll_timeout_ms: u64,

    /// The cadence of the core's tick loop, in milliseconds.
    pub tick_interval_ms: NonZeroU64,
}
