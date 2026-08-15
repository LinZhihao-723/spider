//! The per-resource-group dispatch queues shared between the core and the dispatch service.
//!
//! An assignment is stored exactly once, in the queue of the resource group that owns it. The table
//! below is the only structure both sides touch, so it is `Arc`-backed and lock-free; everything
//! the core keeps about a resource group beyond these endpoints stays core-private.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use dashmap::DashMap;

use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskAssignment;

/// The read side of one resource group's dispatch queue.
///
/// Clones share one queue and one hint counter, so the value is both what a pinned execution
/// manager blocks on and what the core publishes as a hint to general execution managers.
#[derive(Clone, Debug)]
pub struct RgDispatchQueueReader(Arc<RgDispatchQueueReaderInner>);

impl RgDispatchQueueReader {
    /// # Returns
    ///
    /// The resource group this queue belongs to.
    pub(crate) fn rg_id(&self) -> ResourceGroupId {
        self.0.rg_id
    }

    /// # Returns
    ///
    /// The session in which the resource group was created.
    pub(crate) fn session_id(&self) -> SessionId {
        self.0.session_id
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for the resource group.
    pub(crate) fn len(&self) -> usize {
        self.0.receiver.len()
    }

    /// # Returns
    ///
    /// The group's outstanding hint counter, shared with the core's scheduling unit.
    pub(crate) fn living_hint(&self) -> &Arc<AtomicUsize> {
        &self.0.living_hint
    }

    /// Attempts a single non-blocking pop.
    ///
    /// Used by a pinned execution manager to learn whether an assignment is already waiting for it,
    /// which is what tells a request that only cost a dispatch apart from one that had to wait for
    /// work to exist. Like [`Self::recv_pinned`], it leaves the hint counter untouched.
    ///
    /// # Returns
    ///
    /// The next assignment, or [`None`] if the queue is empty or closed.
    pub(crate) fn try_recv_pinned(&self) -> Option<TaskAssignment> {
        self.0.receiver.try_recv().ok()
    }

    /// Blocks until an assignment arrives or `wait_time` expires.
    ///
    /// Used by a pinned execution manager, which is steered by nothing but this queue and therefore
    /// leaves the hint counter untouched.
    ///
    /// # Returns
    ///
    /// The next assignment, or [`None`] if none arrived before `wait_time` expired or the queue was
    /// closed.
    pub(crate) async fn recv_pinned(&self, wait_time: Duration) -> Option<TaskAssignment> {
        tokio::time::timeout(wait_time, self.0.receiver.recv())
            .await
            .ok()?
            .ok()
    }

    /// Consumes one outstanding hint and attempts a single non-blocking pop.
    ///
    /// Synchronous by design: the caller must not yield between receiving the hint and this call's
    /// decrement, or a cancelled request would strand the decrement and permanently over-state the
    /// group's coverage.
    ///
    /// # Returns
    ///
    /// The next assignment, or [`None`] if the hint was stale and the queue is empty.
    pub(crate) fn consume_hint_and_try_recv(&self) -> Option<TaskAssignment> {
        self.0
            .living_hint
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        self.0.receiver.try_recv().ok()
    }

    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created reader over `receiver`.
    fn new(
        receiver: async_channel::Receiver<TaskAssignment>,
        rg_id: ResourceGroupId,
        session_id: SessionId,
    ) -> Self {
        Self(Arc::new(RgDispatchQueueReaderInner {
            receiver,
            living_hint: Arc::new(AtomicUsize::new(0)),
            rg_id,
            session_id,
        }))
    }
}

/// Both ends of one resource group's dispatch queue.
#[derive(Clone, Debug)]
pub struct RgDispatchQueueEndpoints {
    /// The write side, cloned into the core's scheduling unit for the group.
    pub(crate) sender: async_channel::Sender<TaskAssignment>,

    /// The read side, cloned into every execution manager request that touches the group.
    pub(crate) reader: RgDispatchQueueReader,
}

/// The registry of dispatch queue endpoints, shared between the core and the dispatch service.
///
/// Either side may be the first to name a resource group -- a pinned execution manager can connect
/// before any task of its group has been scheduled -- so both lookups create the group on demand.
///
/// Per-group queues are unbounded. The admission threshold is what limits a group's occupancy, so a
/// channel bound would be a second, redundant limit whose only possible effect is to reject a send
/// the design's coverage proof requires to succeed.
#[derive(Clone, Debug)]
pub struct ResourceGroupTable {
    table: Arc<DashMap<ResourceGroupId, RgDispatchQueueEndpoints>>,
}

impl ResourceGroupTable {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created, empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: Arc::new(DashMap::new()),
        }
    }

    /// # Returns
    ///
    /// The read side of `rg_id`'s dispatch queue, creating the group in `session_id` if it has
    /// none.
    pub(crate) fn get_dispatch_queue_reader(
        &self,
        rg_id: ResourceGroupId,
        session_id: SessionId,
    ) -> RgDispatchQueueReader {
        self.get_or_create(rg_id, session_id).reader
    }

    /// # Returns
    ///
    /// Both ends of `rg_id`'s dispatch queue, creating the group in `session_id` if it has none.
    pub(crate) fn get_or_create(
        &self,
        rg_id: ResourceGroupId,
        session_id: SessionId,
    ) -> RgDispatchQueueEndpoints {
        self.table
            .entry(rg_id)
            .or_insert_with(|| {
                let (sender, receiver) = async_channel::unbounded();
                RgDispatchQueueEndpoints {
                    sender,
                    reader: RgDispatchQueueReader::new(receiver, rg_id, session_id),
                }
            })
            .value()
            .clone()
    }

    /// # Returns
    ///
    /// The number of resource groups the table currently holds.
    pub(crate) fn len(&self) -> usize {
        self.table.len()
    }

    /// Drops every resource group.
    ///
    /// An execution manager still blocked on a dropped group's reader keeps that queue alive, but
    /// every assignment in it fails the dispatch service's session check.
    pub(crate) fn clear(&self) {
        self.table.clear();
    }
}

impl Default for ResourceGroupTable {
    fn default() -> Self {
        Self::new()
    }
}

/// The `Arc`-shared body of an [`RgDispatchQueueReader`].
#[derive(Debug)]
struct RgDispatchQueueReaderInner {
    receiver: async_channel::Receiver<TaskAssignment>,
    living_hint: Arc<AtomicUsize>,
    rg_id: ResourceGroupId,
    session_id: SessionId,
}
