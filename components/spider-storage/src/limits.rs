//! The limits that the storage service enforces on job trees and on task reports.

/// The maximum number of events in one tree state transaction.
pub const MAX_EVENTS_PER_ROUND: usize = 256;

/// The maximum number of bytes, outputs plus child blobs, that one tree state transaction writes.
/// The first event that writes anything is always admitted.
pub const MAX_ROUND_PAYLOAD: u64 = 16 * 1024 * 1024;

/// The maximum number of children of one job, over all of its tasks.
pub const MAX_CHILDREN_PER_JOB: usize = 10_000;

/// The maximum number of bytes of the children of one job, over all of its tasks.
pub const MAX_JOB_CHILD_BYTES: u64 = 16 * 1024 * 1024;

/// The maximum number of characters of a failure message that the failed job's error stores.
pub const MAX_ERROR_MESSAGE_CHARS: usize = 64 * 1024;

/// The maximum depth of a job in its tree. A root is at depth 0.
pub const MAX_DEPTH: u32 = 8;

/// The maximum number of compressed bytes of the children of one success report.
pub const MAX_REPORT_CHILD_BYTES: u64 = 16 * 1024 * 1024;

/// The maximum number of bytes of the outputs of one success report.
pub const MAX_REPORT_OUTPUT_BYTES: u64 = 4 * 1024 * 1024;

/// The maximum size of a decoded message of the task instance management service.
pub const MAX_TASK_INSTANCE_MESSAGE_SIZE: usize = 32 * 1024 * 1024;
