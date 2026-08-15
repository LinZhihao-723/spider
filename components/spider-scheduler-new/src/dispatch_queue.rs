//! The hint channel and the execution-manager-facing dispatch path.
//!
//! Assignments never travel through the hint channel; they stay in the queue of the resource group
//! that owns them. A hint says only "there may be work in this resource group", which is what makes
//! exactly-once dispatch structural rather than protocol-enforced.

use std::time::Duration;

use crate::resource_group::ResourceGroupTable;
use crate::resource_group::RgDispatchQueueReader;
use crate::session::SessionManager;
use crate::types::ResourceGroupId;
use crate::types::TaskAssignment;

/// The channel of payload-free hints that steers general execution managers.
///
/// Unbounded, because the coverage invariant requires that a hint send never be rejected and no
/// static bound can guarantee that. A group's hint counter is only ever decremented by a general
/// execution manager, so a pinned execution manager draining that group's queue leaves the counter
/// above the queue size: outstanding hints are bounded by the sum of each group's *peak* occupancy,
/// not by the dispatch buffer's capacity, and that sum can exceed the buffer. The counter still
/// caps each group's share at its peak occupancy, so the channel cannot grow without bound.
#[derive(Clone, Debug)]
pub struct GlobalDispatchQueue {
    sender: async_channel::Sender<RgDispatchQueueReader>,
    receiver: async_channel::Receiver<RgDispatchQueueReader>,
}

impl GlobalDispatchQueue {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created, empty hint channel.
    #[must_use]
    pub fn new() -> Self {
        let (sender, receiver) = async_channel::unbounded();
        Self { sender, receiver }
    }

    /// Publishes a hint for the resource group behind `reader`.
    ///
    /// # Returns
    ///
    /// Whether the hint was published.
    pub(crate) fn try_send(&self, reader: RgDispatchQueueReader) -> bool {
        self.sender.try_send(reader).is_ok()
    }

    /// Attempts a single non-blocking hint pop.
    ///
    /// # Returns
    ///
    /// The next hint, or [`None`] if no hint is outstanding or the channel was closed.
    pub(crate) fn try_recv(&self) -> Option<RgDispatchQueueReader> {
        self.receiver.try_recv().ok()
    }

    /// Blocks until a hint arrives or `wait_time` expires.
    ///
    /// # Returns
    ///
    /// The next hint, or [`None`] if none arrived before `wait_time` expired or the channel was
    /// closed.
    pub(crate) async fn recv(&self, wait_time: Duration) -> Option<RgDispatchQueueReader> {
        tokio::time::timeout(wait_time, self.receiver.recv())
            .await
            .ok()?
            .ok()
    }

    /// # Returns
    ///
    /// The number of outstanding hints.
    pub(crate) fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Discards every outstanding hint.
    pub(crate) fn drain(&self) {
        while self.receiver.try_recv().is_ok() {}
    }
}

impl Default for GlobalDispatchQueue {
    fn default() -> Self {
        Self::new()
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
    rg_table: ResourceGroupTable,
    global_queue: GlobalDispatchQueue,
    session_manager: SessionManager,
}

impl DispatchService {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created dispatch service serving from `rg_table` and `global_queue`.
    #[must_use]
    pub const fn new(
        rg_table: ResourceGroupTable,
        global_queue: GlobalDispatchQueue,
        session_manager: SessionManager,
    ) -> Self {
        Self {
            rg_table,
            global_queue,
            session_manager,
        }
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
            // evidence of a bump: reusing the reader would serve from a table entry the bump has
            // already replaced. A bump landing while this call is blocked is instead picked up by
            // the execution manager's next request.
            let reader = self
                .rg_table
                .get_dispatch_queue_reader(rg_id, self.session_manager.current());
            let mut served = reader.try_recv_pinned();
            if served.is_none() {
                waited = true;
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                served = reader.recv_pinned(remaining).await;
            }
            let Some(assignment) = served else {
                return DispatchOutcome::Empty;
            };

            if assignment.session_id == self.session_manager.current() {
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
    /// hint pop precedes every wait on the hint channel. Traversing stale hints costs nothing but
    /// pops, so a request that discards several of them without ever finding the channel empty is
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
            let mut hint = self.global_queue.try_recv();
            if hint.is_none() {
                waited = true;
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                hint = self.global_queue.recv(remaining).await;
            }
            let Some(reader) = hint else {
                return DispatchOutcome::Empty;
            };

            // No await point may separate the receive above from the decrement inside
            // `consume_hint_and_try_recv`: a future can only be dropped at an await, so this is
            // what makes hint-pop-and-decrement atomic against request cancellation.
            let current_session_id = self.session_manager.current();
            if reader.session_id() != current_session_id {
                tracing::trace!(
                    rg_id = ? reader.rg_id(),
                    hint_session_id = reader.session_id(),
                    current_session_id,
                    "Discarding a hint published in a stale session."
                );
                continue;
            }
            let Some(assignment) = reader.consume_hint_and_try_recv() else {
                continue;
            };

            if assignment.session_id == current_session_id {
                return DispatchOutcome::served(assignment, waited);
            }
        }
    }
}
