//! The globally shared view of storage's current session.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::types::SessionId;

/// The prototype's view of storage's current session.
///
/// Cloneable; clones share the same underlying counter, so the core and the dispatch path always
/// agree on which assignments are still valid.
#[derive(Clone, Debug, Default)]
pub struct SessionManager {
    session_id: Arc<AtomicU64>,
}

impl SessionManager {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created [`SessionManager`] holding `initial_session_id`.
    #[must_use]
    pub fn new(initial_session_id: SessionId) -> Self {
        Self {
            session_id: Arc::new(AtomicU64::new(initial_session_id)),
        }
    }

    /// # Returns
    ///
    /// The session every assignment is currently validated against.
    #[must_use]
    pub fn current(&self) -> SessionId {
        self.session_id.load(Ordering::Acquire)
    }

    /// Replaces the current session with `new_session_id`.
    pub fn bump(&self, new_session_id: SessionId) {
        self.session_id.store(new_session_id, Ordering::Release);
    }
}
