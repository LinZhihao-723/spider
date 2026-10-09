use std::time::Duration;

use serde::Deserialize;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::cache::error::CacheError;
use crate::cache::error::InternalError;
use crate::config::DatabaseConfig;
use crate::db::DbStorage;
use crate::db::MariaDbStorageConnector;
use crate::db::SessionManagement;
use crate::inbound_queue::InboundQueueConfig;
use crate::inbound_queue::create_inbound_queue;
use crate::state::JobCache;
use crate::state::JobCacheGcConfig;
use crate::state::ServiceState;
use crate::state::ServiceStateParams;
use crate::state::ShutdownHandle;
use crate::state::StorageServerError;
use crate::state::create_job_cache_gc;
use crate::task_instance_pool::TaskInstancePoolConfig;
use crate::task_instance_pool::TaskInstancePoolConnector;
use crate::task_instance_pool::TaskInstancePoolHandle;
use crate::task_instance_pool::create_task_instance_pool;
use crate::test_hooks::TestHooks;

/// Runtime configuration for the storage service.
#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeConfig {
    pub db: DatabaseConfig,
    #[serde(default)]
    pub inbound_queue: InboundQueueConfig,
    #[serde(default)]
    pub task_instance_pool: TaskInstancePoolConfig,
    #[serde(default)]
    pub job_cache_gc: JobCacheGcConfig,
}

/// Runtime state for the storage service.
///
/// # Type Parameters
///
/// * `DbConnectorType` - The database connector type.
/// * `TaskInstancePoolConnectorType` - The task instance pool connector type.
pub struct Runtime<
    DbConnectorType: DbStorage + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
> {
    service_state: ServiceState<DbConnectorType, TaskInstancePoolConnectorType>,
    cancellation_token: CancellationToken,
    tracker: TaskTracker,
    task_instance_pool_join_handle: JoinHandle<Result<(), InternalError>>,
    job_cache_gc_join_handle: JoinHandle<Result<(), InternalError>>,
    stop_timeout: Duration,
}

impl<
    DbConnectorType: DbStorage + 'static,
    TaskInstancePoolConnectorType: TaskInstancePoolConnector + 'static,
> Runtime<DbConnectorType, TaskInstancePoolConnectorType>
{
    /// Stops the runtime.
    ///
    /// The background tasks, including every task in the service's task tracker, will be cancelled
    /// and joined. The errors of the background tasks are logged.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`StorageServerError::Stopping`] if any of the background tasks does not stop before
    ///   timeout, or exits on an error or a panic.
    pub async fn stop(self) -> Result<(), StorageServerError> {
        self.cancellation_token.cancel();
        self.tracker.close();

        let join_task_instance_pool = async {
            match self.task_instance_pool_join_handle.await {
                Ok(Ok(())) => {
                    tracing::info!("Task instance pool stopped.");
                    true
                }
                Ok(Err(e)) => {
                    tracing::error!(error = % e, "Task instance pool exited on error.");
                    false
                }
                Err(e) => {
                    tracing::error!(error = % e, "Task instance pool exited on panic.");
                    false
                }
            }
        };

        let join_job_cache_gc = async {
            match self.job_cache_gc_join_handle.await {
                Ok(Ok(())) => {
                    tracing::info!("Job cache GC stopped.");
                    true
                }
                Ok(Err(e)) => {
                    tracing::error!(error = % e, "Job cache GC exited on error.");
                    false
                }
                Err(e) => {
                    tracing::error!(error = % e, "Job cache GC exited on panic.");
                    false
                }
            }
        };

        let (is_task_instance_pool_stopped, is_job_cache_gc_stopped, ()) =
            tokio::time::timeout(self.stop_timeout, async {
                tokio::join!(
                    join_task_instance_pool,
                    join_job_cache_gc,
                    self.tracker.wait()
                )
            })
            .await
            .map_err(|_| {
                StorageServerError::Stopping("background task stop timed out".to_owned())
            })?;
        if !(is_task_instance_pool_stopped && is_job_cache_gc_stopped) {
            return Err(StorageServerError::Stopping(
                "background task exited on an error or a panic".to_owned(),
            ));
        }

        Ok(())
    }

    /// # Returns
    ///
    /// A clone of the runtime's [`ServiceState`].
    #[must_use]
    pub fn get_service_state(
        &self,
    ) -> ServiceState<DbConnectorType, TaskInstancePoolConnectorType> {
        self.service_state.clone()
    }

    /// # Returns
    ///
    /// The runtime's test hooks.
    #[must_use]
    pub fn test_hooks(&self) -> &TestHooks {
        self.service_state.test_hooks()
    }
}

/// Creates a storage server runtime from the given configurations, with no test hook armed.
///
/// # Returns
///
/// Forwards [`create_runtime_with_hooks`]'s return values on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`create_runtime_with_hooks`]'s return values on failure.
pub async fn create_runtime(
    config: &RuntimeConfig,
) -> Result<
    (
        Runtime<MariaDbStorageConnector, TaskInstancePoolHandle>,
        CancellationToken,
    ),
    StorageServerError,
> {
    create_runtime_with_hooks(config, TestHooks::default()).await
}

/// Creates a storage server runtime from the given configurations, with `hooks` as its test hooks:
/// connects to the database, starts the background tasks, and recovers the persisted jobs.
///
/// If a step fails after a background task has been spawned, every spawned task is cancelled and
/// joined before the error is returned.
///
/// # Returns
///
/// A tuple on success, containing:
///
/// * The newly created runtime instance.
/// * The runtime's cancellation token.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`MariaDbStorageConnector::connect`]'s return values on failure.
/// * Forwards [`create_inbound_queue`]'s return values on failure.
/// * Forwards [`create_task_instance_pool`]'s return values on failure.
/// * Forwards [`create_job_cache_gc`]'s return values on failure.
/// * Forwards [`ServiceState::recover_jobs`]'s return values on failure.
pub async fn create_runtime_with_hooks(
    config: &RuntimeConfig,
    hooks: TestHooks,
) -> Result<
    (
        Runtime<MariaDbStorageConnector, TaskInstancePoolHandle>,
        CancellationToken,
    ),
    StorageServerError,
> {
    let cancellation_token = CancellationToken::new();
    let shutdown = ShutdownHandle::new(cancellation_token.clone());
    let stop_timeout = Duration::from_secs(STOP_BACKGROUND_TASKS_TIMEOUT_SEC);
    let db = MariaDbStorageConnector::connect(&config.db).await?;
    let session_id = db.session_id();
    let (inbound_queue_sender, inbound_queue_receiver) =
        create_inbound_queue(&config.inbound_queue).map_err(CacheError::from)?;
    let (task_instance_pool_connector, task_instance_pool_join_handle) = create_task_instance_pool(
        inbound_queue_sender.clone(),
        db.clone(),
        shutdown.clone(),
        &config.task_instance_pool,
    )
    .map_err(CacheError::from)?;

    let job_cache = JobCache::new();
    let (job_cache_gc_handle, job_cache_gc_join_handle) =
        match create_job_cache_gc(job_cache.clone(), shutdown.clone(), &config.job_cache_gc) {
            Ok(created) => created,
            Err(e) => {
                let partial = PartialRuntime {
                    cancellation_token,
                    tracker: None,
                    task_instance_pool: task_instance_pool_join_handle,
                    job_cache_gc: None,
                    stop_timeout,
                };
                return Err(partial.abort(CacheError::from(e).into()).await);
            }
        };

    let service_state = ServiceState::new(ServiceStateParams {
        db,
        session_id,
        job_cache,
        inbound_queue_sender,
        inbound_queue_receiver,
        task_instance_pool_connector,
        job_cache_gc_handle,
        shutdown,
        hooks,
    });
    let tracker = service_state.tracker().clone();

    if let Err(e) = service_state.recover_jobs().await {
        drop(service_state);
        let partial = PartialRuntime {
            cancellation_token,
            tracker: Some(tracker),
            task_instance_pool: task_instance_pool_join_handle,
            job_cache_gc: Some(job_cache_gc_join_handle),
            stop_timeout,
        };
        return Err(partial.abort(e).await);
    }

    Ok((
        Runtime {
            service_state,
            cancellation_token: cancellation_token.clone(),
            tracker,
            task_instance_pool_join_handle,
            job_cache_gc_join_handle,
            stop_timeout,
        },
        cancellation_token,
    ))
}

const STOP_BACKGROUND_TASKS_TIMEOUT_SEC: u64 = 30;

/// The background tasks that a runtime build has spawned so far.
struct PartialRuntime {
    cancellation_token: CancellationToken,
    tracker: Option<TaskTracker>,
    task_instance_pool: JoinHandle<Result<(), InternalError>>,
    job_cache_gc: Option<JoinHandle<Result<(), InternalError>>>,
    stop_timeout: Duration,
}

impl PartialRuntime {
    /// Cancels every spawned background task and waits for them to stop, within the stop timeout.
    /// A failure to stop is logged.
    ///
    /// # Returns
    ///
    /// `error`, the error that failed the build.
    async fn abort(self, error: StorageServerError) -> StorageServerError {
        tracing::error!(
            error = % error,
            "Failed to build the storage runtime. Stopping its background tasks."
        );
        self.cancellation_token.cancel();
        let tracker = self.tracker.unwrap_or_default();
        tracker.close();
        let join_all = async {
            for join_handle in std::iter::once(self.task_instance_pool).chain(self.job_cache_gc) {
                match join_handle.await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::error!(error = % e, "Background task exited on error.");
                    }
                    Err(e) => tracing::error!(error = % e, "Background task exited on panic."),
                }
            }
            tracker.wait().await;
        };
        if tokio::time::timeout(self.stop_timeout, join_all)
            .await
            .is_err()
        {
            tracing::error!("Background tasks of the failed runtime build did not stop in time.");
        }
        error
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::task::JoinHandle;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::cache::error::InternalError;
    use crate::db::SessionManagement;
    use crate::inbound_queue::InboundQueueConfig;
    use crate::inbound_queue::create_inbound_queue;
    use crate::state::JobCache;
    use crate::state::ServiceState;
    use crate::state::ServiceStateParams;
    use crate::state::StorageServerError;
    use crate::state::test_utils::MockDbConnector;
    use crate::state::test_utils::MockTaskInstancePoolConnector;

    type TestServerRuntime = Runtime<MockDbConnector, MockTaskInstancePoolConnector>;

    fn create_test_runtime(
        cancellation_token: CancellationToken,
        mock_task_instance_pool_handle: JoinHandle<Result<(), InternalError>>,
        stop_timeout_sec: u64,
    ) -> TestServerRuntime {
        let db = MockDbConnector::default();
        let session_id = db.session_id();
        let (sender, receiver) =
            create_inbound_queue(&InboundQueueConfig::default()).expect("inbound queue creation");
        let job_cache = JobCache::new();
        let shutdown = ShutdownHandle::new(cancellation_token.clone());
        let (job_cache_gc_handle, job_cache_gc_join_handle) = create_job_cache_gc(
            job_cache.clone(),
            shutdown.clone(),
            &JobCacheGcConfig::default(),
        )
        .expect("job cache GC creation");
        let service_state = ServiceState::new(ServiceStateParams {
            db,
            session_id,
            job_cache,
            inbound_queue_sender: sender,
            inbound_queue_receiver: receiver,
            task_instance_pool_connector: MockTaskInstancePoolConnector,
            job_cache_gc_handle,
            shutdown,
            hooks: TestHooks::default(),
        });

        // Wired with a real job cache GC task, which should always be terminated without errors.
        let tracker = service_state.tracker().clone();
        Runtime {
            service_state,
            cancellation_token,
            tracker,
            task_instance_pool_join_handle: mock_task_instance_pool_handle,
            job_cache_gc_join_handle,
            stop_timeout: Duration::from_secs(stop_timeout_sec),
        }
    }

    #[tokio::test]
    async fn stop_runtime_on_success() -> anyhow::Result<()> {
        let cancellation_token = CancellationToken::new();
        let task_cancellation_token = cancellation_token.clone();
        let mock_task_instance_pool_handle: JoinHandle<Result<(), InternalError>> =
            tokio::spawn(async move {
                task_cancellation_token.cancelled().await;
                Ok(())
            });

        let runtime = create_test_runtime(
            cancellation_token,
            mock_task_instance_pool_handle,
            STOP_BACKGROUND_TASKS_TIMEOUT_SEC,
        );
        runtime
            .stop()
            .await
            .expect("stop_background_tasks should succeed");
        Ok(())
    }

    #[tokio::test]
    async fn stop_runtime_on_timeout() -> anyhow::Result<()> {
        let cancellation_token = CancellationToken::new();
        let mock_task_instance_pool_handle: JoinHandle<Result<(), InternalError>> =
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(10)).await;
                Ok(())
            });

        let runtime = create_test_runtime(cancellation_token, mock_task_instance_pool_handle, 0);
        let result = runtime.stop().await;

        assert!(
            matches!(result, Err(StorageServerError::Stopping(_))),
            "timeout should return Stopping"
        );
        Ok(())
    }

    #[tokio::test]
    async fn stop_runtime_on_task_error() -> anyhow::Result<()> {
        let cancellation_token = CancellationToken::new();
        let mock_task_instance_pool_handle: JoinHandle<Result<(), InternalError>> =
            tokio::spawn(async move {
                Err(InternalError::TaskInstancePoolCorrupted(
                    "test failure".to_owned(),
                ))
            });

        let runtime = create_test_runtime(
            cancellation_token,
            mock_task_instance_pool_handle,
            STOP_BACKGROUND_TASKS_TIMEOUT_SEC,
        );
        let result = runtime.stop().await;

        assert!(
            matches!(result, Err(StorageServerError::Stopping(_))),
            "pool task failure should return Stopping"
        );
        Ok(())
    }

    #[tokio::test]
    async fn stop_runtime_on_task_panic() -> anyhow::Result<()> {
        let cancellation_token = CancellationToken::new();
        let mock_task_instance_pool_handle: JoinHandle<Result<(), InternalError>> =
            tokio::spawn(async move { panic!("test panic") });

        let runtime = create_test_runtime(
            cancellation_token,
            mock_task_instance_pool_handle,
            STOP_BACKGROUND_TASKS_TIMEOUT_SEC,
        );
        let result = runtime.stop().await;

        assert!(
            matches!(result, Err(StorageServerError::Stopping(_))),
            "pool task panic should return Stopping"
        );
        Ok(())
    }
}
