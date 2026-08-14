//! The prototype scheduler's view of the storage-owned inbound queue.

use std::time::Duration;

use async_trait::async_trait;

use crate::error::StorageClientError;
use crate::types::InboundEntry;
use crate::types::SessionId;

/// The storage-owned inbound queue, as the prototype core sees it.
///
/// Modeled as a trait so the core can be driven by a real storage client or by the harness's fake
/// storage. Every lane reports the session it was served under, which is what the core's session
/// bump detection keys on.
#[async_trait]
pub trait SchedulerStorageClient: Send + Sync + Clone + 'static {
    /// Polls the regular-task lane of the inbound queue.
    ///
    /// # Parameters
    ///
    /// * `max_items` - The maximum number of entries to return from a single poll.
    /// * `wait` - The maximum duration the poll may block waiting for entries.
    ///
    /// # Returns
    ///
    /// A tuple on success, containing:
    ///
    /// * The storage session the poll was served under.
    /// * The ready regular tasks drained from the lane.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageClientError::Server`] if the storage service returns an error.
    /// * [`StorageClientError::Transport`] if the storage transport fails or returns malformed
    ///   data.
    async fn poll_ready(
        &self,
        max_items: usize,
        wait: Duration,
    ) -> Result<(SessionId, Vec<InboundEntry>), StorageClientError>;

    /// Polls the commit-ready lane of the inbound queue.
    ///
    /// # Parameters
    ///
    /// * `max_items` - The maximum number of entries to return from a single poll.
    /// * `wait` - The maximum duration the poll may block waiting for entries.
    ///
    /// # Returns
    ///
    /// A tuple on success, containing:
    ///
    /// * The storage session the poll was served under.
    /// * The commit-ready jobs drained from the lane, each carrying
    ///   [`crate::types::TaskId::Commit`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageClientError::Server`] if the storage service returns an error.
    /// * [`StorageClientError::Transport`] if the storage transport fails or returns malformed
    ///   data.
    async fn poll_commit_ready(
        &self,
        max_items: usize,
        wait: Duration,
    ) -> Result<(SessionId, Vec<InboundEntry>), StorageClientError>;

    /// Polls the cleanup-ready lane of the inbound queue.
    ///
    /// # Parameters
    ///
    /// * `max_items` - The maximum number of entries to return from a single poll.
    /// * `wait` - The maximum duration the poll may block waiting for entries.
    ///
    /// # Returns
    ///
    /// A tuple on success, containing:
    ///
    /// * The storage session the poll was served under.
    /// * The cleanup-ready jobs drained from the lane, each carrying
    ///   [`crate::types::TaskId::Cleanup`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageClientError::Server`] if the storage service returns an error.
    /// * [`StorageClientError::Transport`] if the storage transport fails or returns malformed
    ///   data.
    async fn poll_cleanup_ready(
        &self,
        max_items: usize,
        wait: Duration,
    ) -> Result<(SessionId, Vec<InboundEntry>), StorageClientError>;
}
