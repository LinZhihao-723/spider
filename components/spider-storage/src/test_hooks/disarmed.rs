//! The test hooks of a build without the `test-hooks` feature: no hook point ever fires.

use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::cache::error::CacheError;
use crate::db::DbError;
use crate::test_hooks::HookContext;
use crate::test_hooks::HookPoint;

/// A fault that a hook injects. Without the `test-hooks` feature, no fault exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectedFault {}

impl InjectedFault {
    /// # Returns
    ///
    /// Whether the fault is a pause that ended because the runtime's cancellation token was
    /// cancelled.
    #[must_use]
    pub const fn is_stopping(self) -> bool {
        match self {}
    }

    /// # Returns
    ///
    /// The database error that the fault injects on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * The fault itself, if it is not an injected database failure.
    #[must_use]
    pub const fn into_db_error(self) -> Result<DbError, Self> {
        match self {}
    }
}

impl From<InjectedFault> for CacheError {
    fn from(fault: InjectedFault) -> Self {
        match fault {}
    }
}

/// The hooks of one storage runtime, none of which can be armed.
#[derive(Clone, Default)]
pub struct TestHooks {}

impl TestHooks {
    /// Does nothing: there is no hook to release a runtime to.
    pub const fn register_runtime(&self, _token: &CancellationToken, _tracker: &TaskTracker) {}

    /// Fires no hook.
    ///
    /// # Returns
    ///
    /// A future that is ready at once.
    pub fn hit(
        &self,
        _point: HookPoint,
        _context: impl FnOnce() -> HookContext,
    ) -> std::future::Ready<Result<(), InjectedFault>> {
        std::future::ready(Ok(()))
    }

    /// Counts no hit.
    pub fn observe(&self, _point: HookPoint, _context: impl FnOnce() -> HookContext) {}
}
