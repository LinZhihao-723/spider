//! The per-resource-group dispatch queues shared between the core and the dispatch service.
//!
//! An assignment is stored exactly once, in the queue of the resource group that owns it. The table
//! below is the only structure both sides touch, so it is `Arc`-backed and lock-free; everything
//! the core keeps about a resource group beyond these endpoints stays core-private.
//!
//! Each side of a queue has its own view: a shared read side an execution manager pulls through,
//! and a single-owner write side the group's scheduling unit publishes through. The hint counter
//! lives behind the read side, and the write side reaches it only through the reader clone it
//! holds, so the two can never be paired across resource groups.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dashmap::DashMap;

use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskAssignment;

/// Both ends of one resource group's dispatch queue.
#[derive(Clone, Debug)]
pub struct RgDispatchQueueEndpoints {
    /// The write side, handed to the core's scheduling unit for the group.
    pub(crate) sender: async_channel::Sender<TaskAssignment>,

    /// The read side, cloned into every execution manager request that touches the group.
    pub(crate) reader: RgDispatchQueueReader,
}

impl RgDispatchQueueEndpoints {
    /// # Returns
    ///
    /// A newly created write side over this group's queue, paired with this group's reader.
    pub(crate) fn writer(&self) -> RgDispatchQueueWriter {
        RgDispatchQueueWriter {
            sender: self.sender.clone(),
            reader: self.reader.clone(),
        }
    }
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

/// The read side of one resource group's dispatch queue.
///
/// Clones share one queue and one hint counter, so the value is both what a pinned execution
/// manager blocks on and what the core publishes as a hint to general execution managers.
///
/// [`Self::consume_hint_and_try_recv`] is the only way the read side may touch the hint counter.
#[derive(Clone, Debug)]
pub(crate) struct RgDispatchQueueReader {
    inner: Arc<RgDispatchQueueReaderInner>,
}

impl RgDispatchQueueReader {
    /// # Returns
    ///
    /// The resource group this queue belongs to.
    pub(crate) fn rg_id(&self) -> ResourceGroupId {
        self.inner.rg_id
    }

    /// # Returns
    ///
    /// The session in which the resource group was created.
    pub(crate) fn session_id(&self) -> SessionId {
        self.inner.session_id
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
        self.inner.receiver.try_recv().ok()
    }

    /// Blocks until an assignment arrives or `wait_time` expires.
    ///
    /// Used by a pinned execution manager, which is steered by nothing but this queue and therefore
    /// leaves the hint counter untouched.
    ///
    /// A closed queue is deliberately collapsed into the empty case rather than surfaced as an
    /// error: a session bump clears the resource group table, so a caller already blocked here
    /// wakes to find the queue closed, and the only thing that can mean is that the session it
    /// waited for is over. Closure is fatal only on the write side.
    ///
    /// # Returns
    ///
    /// The next assignment, or [`None`] if none arrived before `wait_time` expired or the queue was
    /// closed.
    pub(crate) async fn recv_pinned(&self, wait_time: Duration) -> Option<TaskAssignment> {
        tokio::time::timeout(wait_time, self.inner.receiver.recv())
            .await
            .ok()?
            .ok()
    }

    /// Consumes one outstanding hint and attempts a single non-blocking pop.
    ///
    /// The caller must hold exactly one outstanding hint for this group. A decrement against no
    /// hint clamps at zero, and neither over- nor under-counting is detectable afterward.
    ///
    /// A closed queue is deliberately collapsed into the empty case rather than surfaced as an
    /// error, for the reason given on [`Self::recv_pinned`]: the hint outlived the session that
    /// published it, and the caller is simply served nothing.
    ///
    /// # Cancel safety
    ///
    /// This method is synchronous, so its decrement and its pop cannot be separated by
    /// cancellation. The caller must still not yield between receiving the hint and calling it: a
    /// future can only be dropped at an await point, so an await in that window would let a
    /// cancelled request consume the hint without decrementing the counter, permanently
    /// overstating the group's coverage.
    ///
    /// # Returns
    ///
    /// The next assignment, or [`None`] if the hint was stale and the queue is empty or closed.
    pub(crate) fn consume_hint_and_try_recv(&self) -> Option<TaskAssignment> {
        self.inner.decrement_living_hint();
        self.inner.receiver.try_recv().ok()
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
        Self {
            inner: Arc::new(RgDispatchQueueReaderInner {
                receiver,
                living_hint: AtomicUsize::new(0),
                rg_id,
                session_id,
            }),
        }
    }
}

/// The write side of one resource group's dispatch queue.
///
/// Not clonable: one scheduling unit owns one write side. It is the only view that may write the
/// group's hint counter, which it reaches through the reader it was built with, so the counter it
/// maintains always belongs to the group it publishes into.
#[derive(Debug)]
pub(crate) struct RgDispatchQueueWriter {
    sender: async_channel::Sender<TaskAssignment>,
    reader: RgDispatchQueueReader,
}

impl RgDispatchQueueWriter {
    /// Publishes `assignment` into the group's dispatch queue.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`async_channel::Sender::try_send`]'s return values on failure.
    pub(crate) fn try_send(
        &self,
        assignment: TaskAssignment,
    ) -> Result<(), async_channel::TrySendError<TaskAssignment>> {
        self.sender.try_send(assignment)
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for the resource group.
    pub(crate) fn queue_len(&self) -> usize {
        self.sender.len()
    }

    /// # Returns
    ///
    /// The group's outstanding hint count.
    pub(crate) fn living_hint(&self) -> usize {
        self.reader.inner.living_hint()
    }

    /// Credits the group with one more outstanding hint.
    pub(crate) fn increment_living_hint(&self) {
        self.reader.inner.increment_living_hint();
    }

    /// # Returns
    ///
    /// A hint for this group, which is what the global dispatch queue carries.
    pub(crate) fn hint(&self) -> RgDispatchQueueReader {
        self.reader.clone()
    }
}

/// The `Arc`-shared body of an [`RgDispatchQueueReader`].
///
/// One allocation backs both sides of a group's queue: the write side holds a reader clone rather
/// than a handle of its own on the hint counter.
#[derive(Debug)]
struct RgDispatchQueueReaderInner {
    receiver: async_channel::Receiver<TaskAssignment>,
    living_hint: AtomicUsize,
    rg_id: ResourceGroupId,
    session_id: SessionId,
}

impl RgDispatchQueueReaderInner {
    /// # Returns
    ///
    /// The group's outstanding hint count.
    fn living_hint(&self) -> usize {
        self.living_hint.load(Ordering::Acquire)
    }

    /// Credits the group with one more outstanding hint.
    fn increment_living_hint(&self) {
        self.living_hint.fetch_add(1, Ordering::Release);
    }

    /// Takes one outstanding hint back off the group.
    ///
    /// A decrement against no hint would otherwise wrap the count to [`usize::MAX`], which no later
    /// increment could ever bring back down, so it clamps at zero instead.
    fn decrement_living_hint(&self) {
        let _ = self
            .living_hint
            .try_update(Ordering::AcqRel, Ordering::Acquire, |hint| {
                Some(hint.saturating_sub(1))
            });
    }
}
