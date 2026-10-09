//! An in-process storage runtime on its own scratch database, with test hooks, for job-tree tests.

use std::net::IpAddr;
use std::time::Duration;

use spider_core::job::JobState;
use spider_core::task::DataTypeDescriptor;
use spider_core::task::ExecutionPolicy;
use spider_core::task::TaskDescriptor;
use spider_core::task::TaskIndex;
use spider_core::task::TaskInputOutputIndex;
use spider_core::task::TdlContext;
use spider_core::task::ValueTypeDescriptor;
use spider_core::types::id::ExecutionManagerId;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use spider_core::types::io::TaskGraphInput;
use spider_core::types::io::TaskGraphInputBuilder;
use spider_core::types::resource_group::ExternalResourceGroupCredentials;
use spider_storage::db::MariaDbStorageConnector;
use spider_storage::inbound_queue::InboundQueueConfig;
use spider_storage::state::JobCacheGcConfig;
use spider_storage::state::Runtime;
use spider_storage::state::ServiceState;
use spider_storage::state::create_runtime_with_hooks;
use spider_storage::state::runtime::RuntimeConfig;
use spider_storage::task_instance_pool::TaskInstancePoolConfig;
use spider_storage::task_instance_pool::TaskInstancePoolHandle;
use spider_storage::test_hooks::TestHooks;
use sqlx::MySqlPool;

use super::mariadb_infra::ScratchDatabase;
use super::mariadb_infra::create_raw_pool;
use super::task_graph_builder::SubmittedTaskGraph;
use super::task_graph_builder::compress_job_inputs;
use super::task_graph_builder::compress_task_graph;

/// The runtime type that [`TestEnv`] runs.
pub type TestRuntime = Runtime<MariaDbStorageConnector, TaskInstancePoolHandle>;

/// The service type of a [`TestRuntime`].
pub type TestService = ServiceState<MariaDbStorageConnector, TaskInstancePoolHandle>;

/// A storage runtime on its own scratch database, with its test hooks, a raw pool for row
/// assertions, a resource group and an execution manager.
pub struct TestEnv {
    pub scratch: ScratchDatabase,
    pub config: RuntimeConfig,
    pub runtime: TestRuntime,
    pub service: TestService,
    pub hooks: TestHooks,
    pub pool: MySqlPool,
    pub resource_group_id: ResourceGroupId,
    pub execution_manager_id: ExecutionManagerId,
}

impl TestEnv {
    /// Creates a scratch database and starts a runtime on it, then resends the ready tasks, as the
    /// storage binary does after it starts serving.
    ///
    /// # Panics
    ///
    /// Panics if any step fails.
    pub async fn start() -> Self {
        Self::start_with_task_instance_pool(TaskInstancePoolConfig::default()).await
    }

    /// Like [`Self::start`], with `task_instance_pool` as the runtime's task instance pool
    /// configuration.
    ///
    /// # Panics
    ///
    /// Panics if any step fails.
    pub async fn start_with_task_instance_pool(task_instance_pool: TaskInstancePoolConfig) -> Self {
        let scratch = ScratchDatabase::create().await;
        let config = RuntimeConfig {
            db: scratch.config(),
            inbound_queue: InboundQueueConfig::default(),
            task_instance_pool,
            job_cache_gc: JobCacheGcConfig::default(),
        };
        let pool = create_raw_pool(&config.db).await;
        let hooks = TestHooks::default();
        let runtime = start_runtime(&config, hooks.clone()).await;
        let service = runtime.get_service_state();
        let resource_group_id = service
            .add_resource_group(ExternalResourceGroupCredentials::new(
                format!("job-tree-test-{}", rand::random::<u64>()),
                b"test-password".to_vec(),
            ))
            .await
            .expect("resource group creation should succeed");
        let execution_manager_id = service
            .register_execution_manager(IpAddr::from([127, 0, 0, 1]), None)
            .await
            .expect("execution manager registration should succeed")
            .0;
        Self {
            scratch,
            config,
            runtime,
            service,
            hooks,
            pool,
            resource_group_id,
            execution_manager_id,
        }
    }

    /// # Returns
    ///
    /// The state of `job_id` in the runtime's job cache, root or not, or [`None`] if the job is not
    /// cached.
    pub async fn cached_job_state(&self, job_id: JobId) -> Option<JobState> {
        Some(
            self.service
                .tree_env()
                .job_cache
                .get(job_id)
                .await?
                .state()
                .await,
        )
    }

    /// # Returns
    ///
    /// Whether the runtime's service has shut down on a fatal error.
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        self.service.shutdown().is_fatal()
    }

    /// Stops the runtime and starts a new one on the same database, with fresh hooks, then resends
    /// the ready tasks.
    ///
    /// # Panics
    ///
    /// Panics if the old runtime fails to stop cleanly or the new runtime fails to start.
    #[must_use]
    pub async fn restart(self) -> Self {
        self.stop().await.start(TestHooks::default()).await
    }

    /// Releases the hooks and stops the runtime, keeping the database.
    ///
    /// # Panics
    ///
    /// Panics if the runtime fails to stop cleanly.
    pub async fn stop(self) -> StoppedEnv {
        let Self {
            scratch,
            config,
            runtime,
            service,
            hooks,
            pool,
            resource_group_id,
            execution_manager_id,
        } = self;
        hooks.release_all();
        drop(service);
        runtime
            .stop()
            .await
            .expect("the runtime should stop cleanly");
        StoppedEnv {
            scratch,
            config,
            pool,
            resource_group_id,
            execution_manager_id,
        }
    }

    /// Starts a second runtime on the same database while this one keeps running, as a storage
    /// process that supersedes a live one, and makes it this environment's runtime.
    ///
    /// # Returns
    ///
    /// The superseded runtime and its service.
    ///
    /// # Panics
    ///
    /// Panics if the new runtime fails to start.
    pub async fn start_successor(&mut self) -> (TestRuntime, TestService) {
        let hooks = TestHooks::default();
        let runtime = start_runtime(&self.config, hooks.clone()).await;
        let service = runtime.get_service_state();
        self.hooks = hooks;
        (
            std::mem::replace(&mut self.runtime, runtime),
            std::mem::replace(&mut self.service, service),
        )
    }

    /// Stops the runtime and drops the scratch database.
    ///
    /// # Panics
    ///
    /// Panics if the runtime fails to stop or the database fails to drop.
    pub async fn finish(self) {
        self.hooks.release_all();
        drop(self.service);
        self.runtime
            .stop()
            .await
            .expect("the runtime should stop cleanly");
        self.pool.close().await;
        self.scratch.drop_database().await;
    }

    /// Registers a job.
    ///
    /// # Returns
    ///
    /// The ID of the registered job.
    ///
    /// # Panics
    ///
    /// Panics if the registration fails.
    pub async fn register_job(
        &self,
        task_graph: &SubmittedTaskGraph,
        task_graph_input: &TaskGraphInput,
    ) -> JobId {
        self.service
            .register_job(
                self.resource_group_id,
                compress_task_graph(task_graph).expect("task graph compression should succeed"),
                compress_job_inputs(task_graph_input).expect("input compression should succeed"),
            )
            .await
            .expect("job registration should succeed")
    }

    /// Polls the task lane until it has returned `num_entries` ready tasks of `job_id`, ignoring
    /// other jobs' entries.
    ///
    /// # Returns
    ///
    /// The task indices, in the order they were polled.
    ///
    /// # Panics
    ///
    /// Panics if the entries do not arrive within 10 seconds.
    pub async fn poll_ready_tasks(&self, job_id: JobId, num_entries: usize) -> Vec<TaskIndex> {
        let mut task_indices = Vec::new();
        tokio::time::timeout(Duration::from_secs(10), async {
            while task_indices.len() < num_entries {
                let entries = self
                    .service
                    .poll_ready_tasks(num_entries - task_indices.len(), Duration::from_secs(1))
                    .await
                    .expect("polling should succeed");
                task_indices.extend(
                    entries
                        .into_iter()
                        .filter(|entry| entry.job_id == job_id)
                        .map(|entry| entry.task_kind),
                );
            }
        })
        .await
        .expect("the ready tasks should be published");
        task_indices
    }

    /// # Returns
    ///
    /// The state of `job_id`'s row in the database.
    ///
    /// # Panics
    ///
    /// Panics if the query fails.
    pub async fn row_state(&self, job_id: JobId) -> JobState {
        sqlx::query_scalar::<_, JobState>("SELECT `state` FROM `jobs` WHERE `id` = ?;")
            .bind(job_id)
            .fetch_one(&self.pool)
            .await
            .expect("the row state query should succeed")
    }

    /// # Returns
    ///
    /// The error message stored in `job_id`'s row in the database.
    ///
    /// # Panics
    ///
    /// Panics if the query fails or the row has no error message.
    pub async fn row_error(&self, job_id: JobId) -> String {
        sqlx::query_scalar::<_, Option<String>>(
            "SELECT `error_message` FROM `jobs` WHERE `id` = ?;",
        )
        .bind(job_id)
        .fetch_one(&self.pool)
        .await
        .expect("the row error query should succeed")
        .expect("the row should have an error message")
    }
}

/// A [`TestEnv`] whose runtime is stopped: its scratch database, a raw pool on it, and the
/// resource group and execution manager registered in it.
pub struct StoppedEnv {
    pub scratch: ScratchDatabase,
    pub config: RuntimeConfig,
    pub pool: MySqlPool,
    pub resource_group_id: ResourceGroupId,
    pub execution_manager_id: ExecutionManagerId,
}

impl StoppedEnv {
    /// Starts a runtime on the database with `hooks`, then resends the ready tasks.
    ///
    /// # Panics
    ///
    /// Panics if the runtime fails to start.
    pub async fn start(self, hooks: TestHooks) -> TestEnv {
        let runtime = start_runtime(&self.config, hooks.clone()).await;
        let service = runtime.get_service_state();
        TestEnv {
            scratch: self.scratch,
            config: self.config,
            runtime,
            service,
            hooks,
            pool: self.pool,
            resource_group_id: self.resource_group_id,
            execution_manager_id: self.execution_manager_id,
        }
    }

    /// Starts a runtime on the database with `hooks`, whose armed hooks must fail the start.
    ///
    /// # Panics
    ///
    /// Panics if the runtime starts.
    pub async fn start_failing(&self, hooks: TestHooks) {
        assert!(
            create_runtime_with_hooks(&self.config, hooks)
                .await
                .is_err(),
            "the runtime should fail to start"
        );
    }

    /// Sets the state of `job_id`'s row, bypassing the storage service.
    ///
    /// # Panics
    ///
    /// Panics if the update does not change exactly one row.
    pub async fn set_row_state(&self, job_id: JobId, state: JobState) {
        let result = sqlx::query("UPDATE `jobs` SET `state` = ? WHERE `id` = ?;")
            .bind(state)
            .bind(job_id)
            .execute(&self.pool)
            .await
            .expect("the row update should succeed");
        assert_eq!(result.rows_affected(), 1, "job {job_id:?} should change");
    }
}

/// Builds a graph of `num_tasks` tasks under `execution_policy`, where every task but the first
/// consumes the previous task's output, with one byte-typed input for the first task.
///
/// # Returns
///
/// The task graph and its inputs.
///
/// # Panics
///
/// Panics if the graph fails to build.
#[must_use]
pub fn build_chain_task_graph(
    num_tasks: usize,
    execution_policy: ExecutionPolicy,
) -> (SubmittedTaskGraph, TaskGraphInput) {
    let bytes_type = DataTypeDescriptor::Value(ValueTypeDescriptor::bytes());
    let execution_policy = Some(execution_policy);
    let mut graph =
        SubmittedTaskGraph::new(None, None).expect("chain task graph creation should succeed");
    for task_index in 0..num_tasks {
        graph
            .insert_task(TaskDescriptor {
                tdl_context: TdlContext {
                    package: "test".to_owned(),
                    task_func: format!("n{task_index}"),
                },
                execution_policy: execution_policy.clone(),
                inputs: vec![bytes_type.clone()],
                outputs: vec![bytes_type.clone()],
                input_sources: task_index.checked_sub(1).map(|parent| {
                    vec![TaskInputOutputIndex {
                        task_idx: parent,
                        position: 0,
                    }]
                }),
            })
            .expect("chain task insertion should succeed");
    }
    let mut inputs = TaskGraphInputBuilder::new();
    inputs
        .append_task_input(&[0u8; 4])
        .expect("task input appending should succeed");
    (graph, inputs.build())
}

/// Starts a runtime with `hooks` and resends its ready tasks.
///
/// # Returns
///
/// The runtime.
///
/// # Panics
///
/// Panics if the runtime fails to start or the resend fails.
async fn start_runtime(config: &RuntimeConfig, hooks: TestHooks) -> TestRuntime {
    let (runtime, _) = create_runtime_with_hooks(config, hooks)
        .await
        .expect("the runtime should start");
    runtime
        .get_service_state()
        .resend_ready_tasks()
        .await
        .expect("the ready-task resend should succeed");
    runtime
}
