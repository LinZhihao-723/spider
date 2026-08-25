//! The dispatch queue subsystem: every queue an assignment travels through on its way to an
//! execution manager, and the path an execution manager pulls it out of.
//!
//! An assignment is stored exactly once, in the queue of the resource group that owns it. The
//! registry below owns every channel involved -- one queue per resource group, plus the broadcast
//! queue of payload-free hints that steers general execution managers -- and is the only structure
//! the core and the dispatch service both touch, so it is `Arc`-backed and lock-free; everything
//! the core keeps about a resource group beyond these endpoints stays core-private.
//!
//! Each side of a per-group queue has its own view: a shared read side an execution manager pulls
//! through, and a single-owner write side the group's scheduling unit publishes through. The hint
//! counter lives behind the read side, and the write side reaches it only through the reader clone
//! it holds, so the two can never be paired across resource groups.
//!
//! Assignments never travel through the broadcast queue. A hint says only "there may be work in
//! this resource group", which is what makes exactly-once dispatch structural rather than
//! protocol-enforced.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dashmap::DashMap;
use spider_core::session::SessionTracker;

use crate::error::MakeAssignmentError;
use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskAssignment;

/// The registry of dispatch queues, shared between the core and the dispatch service.
///
/// The registry owns one queue per resource group and the broadcast queue of payload-free hints
/// that steers general execution managers. Clones share one registry, so a hint published through
/// any of them reaches a general execution manager waiting on any other.
///
/// Either side may be the first to name a resource group -- a pinned execution manager can connect
/// before any task of its group has been scheduled -- so both lookups create the group on demand,
/// stamped with the session the registry currently serves.
///
/// Per-group queues are unbounded. The admission threshold is what limits a group's occupancy, so a
/// channel bound would be a second, redundant limit whose only possible effect is to reject a send
/// the design's coverage proof requires to succeed.
///
/// The broadcast queue is unbounded for a stronger reason: the coverage invariant requires that a
/// hint send never be rejected and no static bound can guarantee that. A group's hint counter is
/// only ever decremented by a general execution manager, so a pinned execution manager draining
/// that group's queue leaves the counter above the queue size: outstanding hints are bounded by the
/// sum of each group's *peak* occupancy, not by the dispatch buffer's capacity, and that sum can
/// exceed the buffer. The counter still caps each group's share at its peak occupancy, so the
/// broadcast queue cannot grow without bound.
///
/// # NOTE
///
/// The registry holds both ends of the broadcast queue, and an `async_channel` closes only once all
/// of its senders or all of its receivers are dropped. The broadcast queue therefore cannot close
/// while the registry is alive: its closure means the registry itself is gone, i.e. the scheduler
/// is shutting down. Closure stays fatal on the write path, but is not reachable in a running
/// scheduler.
#[derive(Clone, Debug)]
pub struct DispatchQueueRegistry {
    inner: Arc<DispatchQueueRegistryInner>,
}

impl DispatchQueueRegistry {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created registry holding no resource group and no hint, stamping every group it
    /// creates with `session_tracker`'s current session.
    #[must_use]
    pub fn new(session_tracker: SessionTracker) -> Self {
        let (broadcast_sender, broadcast_receiver) = async_channel::unbounded();
        Self {
            inner: Arc::new(DispatchQueueRegistryInner {
                table: DashMap::new(),
                session_tracker,
                broadcast_sender,
                broadcast_receiver,
            }),
        }
    }

    /// # Returns
    ///
    /// The session every group the registry creates is stamped with.
    pub(crate) fn current_session(&self) -> SessionId {
        self.inner.session_tracker.current()
    }

    /// # Returns
    ///
    /// The read side of `rg_id`'s dispatch queue, creating the group if it has none.
    pub(crate) fn get_dispatch_queue_reader(
        &self,
        rg_id: ResourceGroupId,
    ) -> RgDispatchQueueReader {
        self.get_or_create(rg_id).reader
    }

    /// # Returns
    ///
    /// The write side of `rg_id`'s dispatch queue, creating the group if it has none. The write
    /// side carries the broadcast queue's send end, so the hints covering what it publishes are
    /// published with it.
    pub(crate) fn get_dispatch_queue_writer(
        &self,
        rg_id: ResourceGroupId,
    ) -> RgDispatchQueueWriter {
        self.get_or_create(rg_id)
            .writer(self.inner.broadcast_sender.clone())
    }

    /// # Returns
    ///
    /// The number of resource groups the registry currently holds.
    pub(crate) fn len(&self) -> usize {
        self.inner.table.len()
    }

    /// # Returns
    ///
    /// The number of hints waiting in the broadcast queue.
    pub(crate) fn num_outstanding_hints(&self) -> usize {
        self.inner.broadcast_receiver.len()
    }

    /// Attempts a single non-blocking hint pop.
    ///
    /// # Returns
    ///
    /// The next hint, or [`None`] if no hint is outstanding.
    pub(crate) fn try_next_hint(&self) -> Option<Hint> {
        self.inner.broadcast_receiver.try_recv().ok()
    }

    /// Blocks until a hint arrives or `wait_time` expires.
    ///
    /// The wait is bounded rather than open-ended because the broadcast queue cannot close while
    /// the registry is alive, so an unbounded wait on an empty queue would never return.
    ///
    /// # Cancel safety
    ///
    /// This method is cancel-safe: a hint leaves the broadcast queue only once the receive
    /// completes, so dropping the returned future loses no hint. The caller must still not yield
    /// between being handed a hint and consuming it, for the reason given on
    /// [`Hint::consume_and_try_recv`].
    ///
    /// # Returns
    ///
    /// The next hint, or [`None`] if none arrived before `wait_time` expired or the broadcast queue
    /// was closed.
    pub(crate) async fn next_hint(&self, wait_time: Duration) -> Option<Hint> {
        tokio::time::timeout(wait_time, self.inner.broadcast_receiver.recv())
            .await
            .ok()?
            .ok()
    }

    /// Drops every resource group and discards every outstanding hint.
    ///
    /// An execution manager still blocked on a dropped group's reader keeps that queue alive, but
    /// every assignment in it fails the dispatch service's session check.
    ///
    /// The broadcast queue outlives the flush -- the registry holds both of its ends -- so its
    /// hints have to be discarded explicitly, or hints naming groups this call drops would steer
    /// general execution managers of the next session. The groups are dropped before the hints are
    /// discarded: the core is the only publisher of hints and is itself executing this call, so no
    /// publication can race it, and dropping first means that even a hint published after the call
    /// began would be one this call discards.
    ///
    /// Discarding a hint deliberately leaves its group's hint counter as it is: the counter belongs
    /// to a group no reader of the next session consults, and dropping a [`Hint`] is free, since
    /// only [`Hint::consume_and_try_recv`] ever lowers a counter.
    pub(crate) fn clear(&self) {
        self.inner.table.clear();
        while self.inner.broadcast_receiver.try_recv().is_ok() {}
    }

    /// Closes the broadcast queue, so that every later hint publication is rejected.
    ///
    /// No production path closes the queue; this exists so that a test can drive the core into the
    /// failure the closure represents.
    #[cfg(test)]
    pub(crate) fn close_broadcast_queue(&self) {
        self.inner.broadcast_sender.close();
    }

    /// Closes `rg_id`'s dispatch queue, creating the group if it has none, so that every later
    /// publication into it is rejected.
    ///
    /// No production path closes a group's queue on its own; this exists so that a test can drive
    /// the core into the failure the closure represents.
    #[cfg(test)]
    pub(crate) fn close_dispatch_queue(&self, rg_id: ResourceGroupId) {
        self.get_or_create(rg_id).sender.close();
    }

    /// Puts `assignment` straight into `rg_id`'s dispatch queue, creating the group if it has none,
    /// without the hint [`RgDispatchQueueWriter::try_send`] would decide on alongside it.
    ///
    /// Queuing an assignment that no hint covers is the coverage break design §8.1 rules out, which
    /// is why no production path can reach the group's sender. This exists so that a test can stand
    /// in for the occupancy a group carries over from an earlier tick.
    ///
    /// # Returns
    ///
    /// Whether the group's queue accepted the assignment.
    #[cfg(test)]
    pub(crate) fn preload_queue(&self, rg_id: ResourceGroupId, assignment: TaskAssignment) -> bool {
        self.get_or_create(rg_id)
            .sender
            .try_send(assignment)
            .is_ok()
    }

    /// # Returns
    ///
    /// Both ends of `rg_id`'s dispatch queue, creating the group if it has none.
    fn get_or_create(&self, rg_id: ResourceGroupId) -> RgDispatchQueueEndpoints {
        let session_id = self.current_session();
        self.inner
            .table
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
}

/// What one dispatch request was served, together with the class it falls into.
///
/// The class is what separates the scheduler's own cost from the supply of work: only an immediate
/// request measures how long the scheduler took to hand an assignment over, while a waited one is
/// dominated by how long the request had to wait for an assignment to exist at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// An assignment was already available, and was served without the request ever awaiting on an
    /// empty queue.
    Immediate(TaskAssignment),

    /// The request awaited on an empty queue, and an assignment arrived before its wait expired.
    Waited(TaskAssignment),

    /// No assignment arrived before the request's wait expired.
    Empty,
}

impl DispatchOutcome {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created outcome carrying `assignment`, classified as waited if the request awaited
    /// on an empty queue at any point before it was served.
    #[must_use]
    pub const fn served(assignment: TaskAssignment, waited: bool) -> Self {
        if waited {
            Self::Waited(assignment)
        } else {
            Self::Immediate(assignment)
        }
    }

    /// # Returns
    ///
    /// The assignment served, or [`None`] if the request was served none.
    #[must_use]
    pub const fn into_assignment(self) -> Option<TaskAssignment> {
        match self {
            Self::Immediate(assignment) | Self::Waited(assignment) => Some(assignment),
            Self::Empty => None,
        }
    }
}

/// The path an execution manager takes to pull its next assignment.
#[derive(Clone, Debug)]
pub struct DispatchService {
    registry: DispatchQueueRegistry,
}

impl DispatchService {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created dispatch service serving from `registry`.
    #[must_use]
    pub const fn new(registry: DispatchQueueRegistry) -> Self {
        Self { registry }
    }

    /// Serves an execution manager pinned to `rg_id`.
    ///
    /// # Returns
    ///
    /// The next assignment of `rg_id`, or [`None`] if none became available within `wait_time`.
    pub async fn next_task_pinned(
        &self,
        rg_id: ResourceGroupId,
        wait_time: Duration,
    ) -> Option<TaskAssignment> {
        self.next_task_pinned_classified(rg_id, wait_time)
            .await
            .into_assignment()
    }

    /// Serves an execution manager pinned to `rg_id`, reporting which class the request fell into.
    ///
    /// The class is an observation of the path [`Self::next_task_pinned`] already takes rather than
    /// a second path: a non-blocking pop precedes every wait, so the wait is entered only once the
    /// queue has been seen empty, and neither how long the request is willing to wait nor what it
    /// returns changes. A retry that follows a stale-session assignment carries whatever the
    /// request has already waited on, so having waited once makes the whole request a waited one.
    ///
    /// # Returns
    ///
    /// The next assignment of `rg_id` and its class, or [`DispatchOutcome::Empty`] if none became
    /// available within `wait_time`.
    pub async fn next_task_pinned_classified(
        &self,
        rg_id: ResourceGroupId,
        wait_time: Duration,
    ) -> DispatchOutcome {
        let deadline = tokio::time::Instant::now() + wait_time;
        let mut waited = false;
        loop {
            // Re-fetched per attempt because a retry follows a stale-session assignment, which is
            // evidence of a bump: reusing the reader would serve from a registry entry the bump has
            // already replaced. A bump landing while this call is blocked is instead picked up by
            // the execution manager's next request.
            let reader = self.registry.get_dispatch_queue_reader(rg_id);
            let mut served = reader.try_recv_pinned();
            if served.is_none() {
                waited = true;
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                served = reader.recv_pinned(remaining).await;
            }
            let Some(assignment) = served else {
                return DispatchOutcome::Empty;
            };

            if assignment.session_id == self.registry.current_session() {
                return DispatchOutcome::served(assignment, waited);
            }
        }
    }

    /// Serves a general execution manager, which may receive an assignment of any resource group.
    ///
    /// # Returns
    ///
    /// The next assignment of any resource group, or [`None`] if none became available within
    /// `wait_time`.
    pub async fn next_task_general(&self, wait_time: Duration) -> Option<TaskAssignment> {
        self.next_task_general_classified(wait_time)
            .await
            .into_assignment()
    }

    /// Serves a general execution manager, reporting which class the request fell into.
    ///
    /// As on the pinned path, the class is an observation rather than a second path: a non-blocking
    /// hint pop precedes every wait on the broadcast queue. Traversing stale hints costs nothing
    /// but pops, so a request that discards several of them without ever finding the queue empty is
    /// still an immediate one.
    ///
    /// # Returns
    ///
    /// The next assignment of any resource group and its class, or [`DispatchOutcome::Empty`] if
    /// none became available within `wait_time`.
    pub async fn next_task_general_classified(&self, wait_time: Duration) -> DispatchOutcome {
        let deadline = tokio::time::Instant::now() + wait_time;
        let mut waited = false;
        loop {
            let mut hint = self.registry.try_next_hint();
            if hint.is_none() {
                waited = true;
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                hint = self.registry.next_hint(remaining).await;
            }
            let Some(hint) = hint else {
                return DispatchOutcome::Empty;
            };

            // No await point may separate the receive above from the decrement inside
            // `consume_and_try_recv`: a future can only be dropped at an await, so this is what
            // makes hint-pop-and-decrement atomic against request cancellation.
            let current_session_id = self.registry.current_session();
            if hint.session_id() != current_session_id {
                tracing::trace!(
                    rg_id = ? hint.rg_id(),
                    hint_session_id = hint.session_id(),
                    current_session_id,
                    "Discarding a hint published in a stale session."
                );
                continue;
            }
            let Some(assignment) = hint.consume_and_try_recv() else {
                continue;
            };

            if assignment.session_id == current_session_id {
                return DispatchOutcome::served(assignment, waited);
            }
        }
    }
}

/// Both ends of one resource group's dispatch queue.
///
/// Module-private, and so is its sender: the only way out of this module for the write side is an
/// [`RgDispatchQueueWriter`], which is what makes an assignment queued without a hint
/// unconstructible.
#[derive(Clone, Debug)]
struct RgDispatchQueueEndpoints {
    /// The write side, from which the group's scheduling unit's write view is built.
    sender: async_channel::Sender<TaskAssignment>,

    /// The read side, cloned into every execution manager request that touches the group.
    reader: RgDispatchQueueReader,
}

impl RgDispatchQueueEndpoints {
    /// # Returns
    ///
    /// A newly created writer pairing the group's sender with the group's own reader and with
    /// `broadcast_sender`.
    fn writer(&self, broadcast_sender: async_channel::Sender<Hint>) -> RgDispatchQueueWriter {
        RgDispatchQueueWriter {
            sender: self.sender.clone(),
            broadcast_sender,
            reader: self.reader.clone(),
        }
    }
}

/// One outstanding hint of one resource group: the right to pop that group's queue once on behalf
/// of a general execution manager.
///
/// A hint can only come out of the broadcast queue. Its field and its constructor are both
/// module-private, so no holder of an [`RgDispatchQueueReader`] can mint one, which is what keeps
/// the pinned path -- which holds a reader and no hint -- from lowering a group's hint counter.
///
/// Spending a hint consumes it, and the type is deliberately neither [`Clone`] nor [`Copy`], so
/// "a hint is spent at most once" is enforced by the type system rather than by its caller.
/// Dropping a hint instead is free and leaves the counter as it is, which is what lets the dispatch
/// service discard a hint of a stale session without spending it.
#[derive(Debug)]
pub(crate) struct Hint {
    reader: RgDispatchQueueReader,
}

impl Hint {
    /// # Returns
    ///
    /// The resource group the hint names.
    pub(crate) fn rg_id(&self) -> ResourceGroupId {
        self.reader.rg_id()
    }

    /// # Returns
    ///
    /// The session in which the hint's resource group was created.
    pub(crate) fn session_id(&self) -> SessionId {
        self.reader.session_id()
    }

    /// Spends the hint and attempts a single non-blocking pop of its group.
    ///
    /// A closed queue is deliberately collapsed into the empty case rather than surfaced as an
    /// error, for the reason given on [`RgDispatchQueueReader::recv_pinned`]: the hint outlived the
    /// session that published it, and the caller is simply served nothing.
    ///
    /// # Cancel safety
    ///
    /// This method is synchronous, so its decrement and its pop cannot be separated by
    /// cancellation. The caller must still not yield between receiving the hint and calling it: a
    /// future can only be dropped at an await point, so an await in that window would let a
    /// cancelled request drop the hint without decrementing the counter, permanently overstating
    /// the group's coverage.
    ///
    /// # Returns
    ///
    /// The next assignment, or [`None`] if the hint was stale and the queue is empty or closed.
    pub(crate) fn consume_and_try_recv(self) -> Option<TaskAssignment> {
        self.reader.inner.decrement_living_hint();
        self.reader.inner.receiver.try_recv().ok()
    }

    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created hint naming `reader`'s resource group.
    const fn new(reader: RgDispatchQueueReader) -> Self {
        Self { reader }
    }
}

/// The read side of one resource group's dispatch queue.
///
/// Clones share one queue and one hint counter, so the value is both what a pinned execution
/// manager blocks on and what a [`Hint`] carries to a general execution manager.
///
/// The read side never touches the hint counter itself: [`Hint::consume_and_try_recv`] is the only
/// path that lowers it.
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
    /// error: a session bump clears the registry's table, so a caller already blocked here wakes to
    /// find the queue closed, and the only thing that can mean is that the session it waited for is
    /// over. Closure is fatal only on the write side.
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

    /// # Returns
    ///
    /// The group's outstanding hint count, which no production path reads: the write side observes
    /// it only from inside [`RgDispatchQueueWriter::try_send`].
    #[cfg(test)]
    pub(crate) fn living_hint(&self) -> usize {
        self.inner.living_hint()
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

/// The write side of one resource group's dispatch queue, and of the broadcast queue the group's
/// hints go into.
///
/// Not clonable: one scheduling unit owns one write side. It is the only view that may write the
/// group's hint counter, which it reaches through the reader it was built with, so the counter it
/// maintains always belongs to the group it publishes into.
#[derive(Debug)]
pub(crate) struct RgDispatchQueueWriter {
    sender: async_channel::Sender<TaskAssignment>,
    broadcast_sender: async_channel::Sender<Hint>,
    reader: RgDispatchQueueReader,
}

impl RgDispatchQueueWriter {
    /// Publishes `assignment` into the group's dispatch queue and, when the group's coverage
    /// requires it, a hint into the broadcast queue.
    ///
    /// Publishing the assignment and deciding on its hint are one operation because the order of
    /// the two is load-bearing: the assignment must already be in the queue when the hint count is
    /// compared against the queue size, or a general execution manager can consume a hint for an
    /// assignment that is not yet visible.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`MakeAssignmentError::DispatchQueueClosed`] if the group's queue or the broadcast queue
    ///   is closed. The two are indistinguishable to the caller, and a running scheduler never
    ///   observes the latter: the registry holds both ends of the broadcast queue, so its closure
    ///   means the registry itself is gone.
    pub(crate) fn try_send(&self, assignment: TaskAssignment) -> Result<(), MakeAssignmentError> {
        self.sender
            .try_send(assignment)
            .map_err(|_| MakeAssignmentError::DispatchQueueClosed)?;

        let Some(hint) = self.try_make_hint() else {
            return Ok(());
        };
        self.broadcast_sender
            .try_send(hint)
            .map_err(|_| MakeAssignmentError::DispatchQueueClosed)
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for the resource group.
    pub(crate) fn queue_len(&self) -> usize {
        self.sender.len()
    }

    /// Takes out one hint for the resource group, unless its outstanding hints already cover its
    /// queue.
    ///
    /// The caller must already have published the assignment this hint covers, and must publish the
    /// returned hint into the broadcast queue. Within the check, the queue occupancy `S` is
    /// sampled before the hint count `H`, which the memory-ordering argument in design §8.1 depends
    /// on: a decrement landing between the two reads can only lower `H`, which strengthens the
    /// condition that triggers publication.
    ///
    /// # Returns
    ///
    /// The hint taken out, through which a general execution manager pops from this group, or
    /// [`None`] if the group's outstanding hints already cover its queue and no hint is needed.
    ///
    /// # NOTE
    ///
    /// The count is raised before the returned hint is published, so a hint that is taken out but
    /// never published leaves `H` permanently overstated, which suppresses the group's later hints
    /// and can strand its assignments for want of a general execution manager. That is a different
    /// thing from the transient over-read design §8.1 blesses as safe: there the unobserved change
    /// is a decrement, which corresponds to an execution manager already counted in `m`.
    fn try_make_hint(&self) -> Option<Hint> {
        let dispatch_queue_size = self.queue_len();
        if self.reader.inner.living_hint() >= dispatch_queue_size {
            return None;
        }

        self.reader.inner.increment_living_hint();
        Some(Hint::new(self.reader.clone()))
    }
}

/// The `Arc`-shared body of a [`DispatchQueueRegistry`].
///
/// One allocation backs the whole subsystem: every group's queue endpoints, the session those
/// groups are stamped with, and both ends of the broadcast queue.
#[derive(Debug)]
struct DispatchQueueRegistryInner {
    table: DashMap<ResourceGroupId, RgDispatchQueueEndpoints>,
    session_tracker: SessionTracker,
    broadcast_sender: async_channel::Sender<Hint>,
    broadcast_receiver: async_channel::Receiver<Hint>,
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
    /// The caller must hold exactly one outstanding hint for this group. A decrement with no hint
    /// outstanding wraps the count, permanently corrupting the group's hint accounting.
    fn decrement_living_hint(&self) {
        self.living_hint.fetch_sub(1, Ordering::AcqRel);
    }
}
