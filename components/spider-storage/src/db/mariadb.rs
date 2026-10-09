use std::net::IpAddr;
use std::time::Duration;

use async_trait::async_trait;
use const_format::formatcp;
use secrecy::ExposeSecret;
use spider_core::job::JobState;
use spider_core::types::id::ExecutionManagerId;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use spider_core::types::id::SchedulerId;
use spider_core::types::id::SessionId;
use spider_core::types::io::SerializedTaskOutputs;
use spider_core::types::io::TaskOutput;
use spider_core::types::resource_group::ExternalResourceGroupCredentials;
use spider_core::types::scheduler::RegisteredScheduler;
use spider_derive::MySqlEnum;
use spider_utils::config::Host;
use sqlx::Connection;
use sqlx::MySqlConnection;
use sqlx::MySqlPool;
use sqlx::mysql::MySqlDatabaseError;
use subtle::ConstantTimeEq;

use crate::config::DatabaseConfig;
use crate::db::ChildJobInsert;
use crate::db::DbError;
use crate::db::DbStorage;
use crate::db::ExecutionManagerLivenessManagement;
use crate::db::ExternalJobOrchestration;
use crate::db::InternalJobOrchestration;
use crate::db::JobStateUpdate;
use crate::db::RecoverableJobContext;
use crate::db::ResourceGroupManagement;
use crate::db::SchedulerRegistrationManagement;
use crate::db::SessionManagement;
use crate::db::TreeTransition;
use crate::db::error::ExpectedStates;
use crate::job_submission::ValidatedJobSubmission;

/// A cloneable storage connector for `MariaDB` database that implements Spider's DB protocols.
#[derive(Clone)]
pub struct MariaDbStorageConnector {
    pool: MySqlPool,
    session_id: SessionId,
    fence_id: u64,
    max_statement_payload: u64,
    max_outputs_size: u64,
    schema_ddl_statements_at_connect: usize,
}

impl MariaDbStorageConnector {
    /// Connects to database, initializes or upgrades the tables, and fences off every storage
    /// session that connected earlier with the same fence ID.
    ///
    /// # Parameters
    ///
    /// * `config`: Database configuration parameters for connecting to the database.
    ///
    /// # Returns
    ///
    /// A newly created [`MariaDbStorageConnector`] instance for connection on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`sqlx::mysql::MySqlPoolOptions::connect`]'s return values on failure.
    /// * Forwards [`sqlx::query::Query::execute`]'s return values on failure.
    /// * Forwards [`upgrade_jobs_schema`]'s return values on failure.
    /// * Forwards [`bump_fence`]'s return values on failure.
    /// * Forwards [`sqlx::query::QueryScalar::fetch_one`]'s return values on failure.
    pub async fn connect(config: &DatabaseConfig) -> Result<Self, DbError> {
        const SET_WAIT_TIMEOUT_QUERY: &str =
            formatcp!("SET SESSION wait_timeout = {SESSION_WAIT_TIMEOUT_SEC};");
        const BUMP_SESSION_ID_QUERY: &str = formatcp!(
            "INSERT INTO `{table}` () VALUES ();",
            table = SESSIONS_TABLE_NAME,
        );
        const SELECT_MAX_ALLOWED_PACKET_QUERY: &str = "SELECT @@max_allowed_packet;";
        const INSERT_FENCE_QUERY: &str = formatcp!(
            "INSERT IGNORE INTO `{table}` (`id`, `session_id`) VALUES (?, 0);",
            table = STORAGE_FENCE_TABLE_NAME,
        );

        let mysql_options = sqlx::mysql::MySqlConnectOptions::new()
            .host(&config.host)
            .port(config.port)
            .database(&config.name)
            .username(&config.credentials.username)
            .password(config.credentials.password.expose_secret());

        let pool = sqlx::mysql::MySqlPoolOptions::new()
            .max_connections(config.max_connections)
            .after_connect(|connection, _| {
                Box::pin(async move {
                    sqlx::query(SET_WAIT_TIMEOUT_QUERY)
                        .execute(connection)
                        .await?;
                    Ok(())
                })
            })
            .connect_with(mysql_options)
            .await?;

        // MariaDB does not support transactions for DDL statements. All DDL statements are
        // automatically committed. Thus, each table creation query is executed separately, and
        // atomicity is not guaranteed.
        sqlx::query(resource_groups_creation_query())
            .execute(&pool)
            .await?;
        sqlx::query(jobs_creation_query()).execute(&pool).await?;
        let schema_ddl_statements_at_connect = upgrade_jobs_schema(&pool).await?;
        sqlx::query(sessions_creation_query())
            .execute(&pool)
            .await?;
        sqlx::query(execution_managers_creation_query())
            .execute(&pool)
            .await?;
        sqlx::query(schedulers_creation_query())
            .execute(&pool)
            .await?;
        sqlx::query(storage_fence_creation_query())
            .execute(&pool)
            .await?;
        sqlx::query(INSERT_FENCE_QUERY)
            .bind(config.fence_id)
            .execute(&pool)
            .await?;

        let session_id = SessionId::from(
            sqlx::query(BUMP_SESSION_ID_QUERY)
                .execute(&pool)
                .await?
                .last_insert_id(),
        );
        bump_fence(&pool, config.fence_id, session_id).await?;
        let max_allowed_packet: u64 = sqlx::query_scalar(SELECT_MAX_ALLOWED_PACKET_QUERY)
            .fetch_one(&pool)
            .await?;

        Ok(Self {
            pool,
            session_id,
            fence_id: config.fence_id,
            max_statement_payload: MAX_STATEMENT_PAYLOAD.min(max_allowed_packet / 2),
            max_outputs_size: max_allowed_packet.saturating_sub(MAX_STATEMENT_OVERHEAD),
            schema_ddl_statements_at_connect,
        })
    }

    /// # Returns
    ///
    /// The number of DDL statements that [`Self::connect`] issued to upgrade the `jobs` table.
    #[must_use]
    pub const fn schema_ddl_statements_at_connect(&self) -> usize {
        self.schema_ddl_statements_at_connect
    }

    /// Checks, inside a transaction that writes, that no newer storage session has fenced this one
    /// off.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`DbError::Fenced`] if the fence row holds a session ID other than this connector's.
    /// * [`DbError::CorruptedDbState`] if the fence row does not exist.
    /// * Forwards [`sqlx::query::QueryScalar::fetch_optional`]'s return values on failure.
    async fn check_fence(&self, connection: &mut MySqlConnection) -> Result<(), DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `session_id` FROM `{table}` WHERE `id` = ? LOCK IN SHARE MODE;",
            table = STORAGE_FENCE_TABLE_NAME,
        );

        let current = sqlx::query_scalar::<_, SessionId>(QUERY)
            .bind(self.fence_id)
            .fetch_optional(connection)
            .await?
            .ok_or_else(|| {
                DbError::CorruptedDbState(format!(
                    "storage fence `{}` does not exist",
                    self.fence_id
                ))
            })?;
        if current != self.session_id {
            return Err(DbError::Fenced {
                mine: self.session_id,
                current,
            });
        }
        Ok(())
    }
}

#[async_trait]
impl ExternalJobOrchestration for MariaDbStorageConnector {
    async fn register(
        &self,
        resource_group_id: ResourceGroupId,
        job_submission: &ValidatedJobSubmission,
    ) -> Result<JobId, DbError> {
        const INSERT_QUERY: &str = formatcp!(
            "INSERT INTO `{table}` (`resource_group_id`, `compressed_serialized_task_graph`, \
             `compressed_serialized_job_inputs`) VALUES (?, ?, ?);",
            table = JOBS_TABLE_NAME,
        );

        let compressed_serialized_task_graph = job_submission.compressed_serialized_task_graph();
        let compressed_serialized_job_inputs = job_submission.compressed_serialized_job_inputs();

        let mut tx = self.pool.begin().await?;
        self.check_fence(&mut tx).await?;

        let job_id = sqlx::query(INSERT_QUERY)
            .bind(resource_group_id)
            .bind(compressed_serialized_task_graph)
            .bind(compressed_serialized_job_inputs)
            .execute(&mut *tx)
            .await
            .map_err(|e| match e {
                sqlx::Error::Database(e)
                    if e.try_downcast_ref::<MySqlDatabaseError>()
                        .is_some_and(|mysql_err| mysql_err.number() == MYSQL_ER_FK_CONSTRAINT) =>
                {
                    DbError::ResourceGroupNotFound(resource_group_id)
                }
                e => e.into(),
            })?
            .last_insert_id();

        tx.commit().await?;
        Ok(JobId::from(job_id))
    }

    async fn get_state(&self, job_id: JobId) -> Result<JobState, DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `state` FROM `{table}` WHERE `id` = ?;",
            table = JOBS_TABLE_NAME,
        );

        let row: Option<(JobState,)> = sqlx::query_as(QUERY)
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await?;

        let (state,) = row.ok_or(DbError::JobNotFound(job_id))?;
        Ok(state)
    }

    async fn get_state_and_parent(
        &self,
        job_id: JobId,
    ) -> Result<(JobState, Option<JobId>), DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `state`, `{parent}` FROM `{table}` WHERE `id` = ?;",
            parent = JOB_PARENT_ID_COLUMN_NAME,
            table = JOBS_TABLE_NAME,
        );

        let row: Option<(JobState, Option<JobId>)> = sqlx::query_as(QUERY)
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await?;

        row.ok_or(DbError::JobNotFound(job_id))
    }

    async fn get_outputs(&self, job_id: JobId) -> Result<Vec<TaskOutput>, DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `state`, `serialized_job_outputs` FROM `{table}` WHERE `id` = ?;",
            table = JOBS_TABLE_NAME,
        );

        let Some((state, serialized_outputs)) =
            sqlx::query_as::<_, (JobState, Option<Vec<u8>>)>(QUERY)
                .bind(job_id)
                .fetch_optional(&self.pool)
                .await?
        else {
            return Err(DbError::JobNotFound(job_id));
        };

        if state != JobState::Succeeded {
            return Err(DbError::UnexpectedJobState {
                current: state,
                expected: ExpectedStates(vec![JobState::Succeeded]),
            });
        }

        let outputs_bytes = serialized_outputs.ok_or_else(|| {
            DbError::CorruptedDbState(format!(
                "job `{job_id}` succeeded but has no serialized outputs"
            ))
        })?;
        let outputs = SerializedTaskOutputs::deserialize_from_raw(&outputs_bytes)
            .map_err(|e| DbError::ValueDeserializationFailure(Box::new(e)))?;
        Ok(outputs)
    }

    async fn get_error(&self, job_id: JobId) -> Result<String, DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `state`, `error_message` FROM `{table}` WHERE `id` = ?;",
            table = JOBS_TABLE_NAME,
        );

        let Some((state, error_message)) = sqlx::query_as::<_, (JobState, Option<String>)>(QUERY)
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await?
        else {
            return Err(DbError::JobNotFound(job_id));
        };

        if state != JobState::Failed {
            return Err(DbError::UnexpectedJobState {
                current: state,
                expected: ExpectedStates(vec![JobState::Failed]),
            });
        }

        let message = error_message.ok_or_else(|| {
            DbError::CorruptedDbState(format!("job `{job_id}` failed but has no error message"))
        })?;
        Ok(message)
    }
}

#[async_trait]
impl InternalJobOrchestration for MariaDbStorageConnector {
    async fn apply_tree_transition(
        &self,
        transition: TreeTransition<'_>,
    ) -> Result<Vec<JobId>, DbError> {
        let statements = UpdateStatement::group(
            transition.updates,
            self.max_statement_payload,
            self.max_outputs_size,
        )?;
        let insert_chunks =
            chunk_child_inserts(&transition.child_inserts, self.max_statement_payload);

        let mut tx = self.pool.begin().await?;
        let result: Result<Vec<JobId>, DbError> = async {
            self.check_fence(&mut tx).await?;
            for statement in &statements {
                statement.execute(&mut tx).await?;
            }
            let mut child_ids = Vec::with_capacity(transition.child_inserts.len());
            for chunk in insert_chunks {
                child_ids.extend(insert_children(&mut tx, chunk).await.map_err(|e| match e {
                    DbError::Sql(e) if is_payload_rejection(&e) => DbError::PayloadRejected(e),
                    e => e,
                })?);
            }
            Ok(child_ids)
        }
        .await;
        match result {
            Ok(child_ids) => {
                tx.commit().await?;
                Ok(child_ids)
            }
            Err(e) => {
                if let Err(rollback_error) = tx.rollback().await {
                    tracing::warn!(
                        error = % rollback_error,
                        "Failed to roll back a failed tree transition."
                    );
                }
                Err(e)
            }
        }
    }

    fn max_statement_payload(&self) -> u64 {
        self.max_statement_payload
    }

    async fn delete_expired_terminated_jobs(
        &self,
        expire_after_sec: u64,
    ) -> Result<Vec<JobId>, DbError> {
        const DELETE_BATCH_SIZE: usize = 1000;

        const SELECT_CANDIDATES_QUERY: &str = formatcp!(
            "SELECT `id` FROM `{table}` WHERE `state` IN \
             ('{succeeded_state}','{failed_state}','{cancelled_state}') AND `ended_at` < NOW() - \
             INTERVAL ? SECOND LIMIT {DELETE_BATCH_SIZE};",
            table = JOBS_TABLE_NAME,
            succeeded_state = JobState::Succeeded.as_str(),
            failed_state = JobState::Failed.as_str(),
            cancelled_state = JobState::Cancelled.as_str(),
        );

        let mut deleted_job_ids: Vec<JobId> = Vec::new();
        let mut tx = self.pool.begin().await?;
        self.check_fence(&mut tx).await?;

        loop {
            let candidate_ids: Vec<JobId> = sqlx::query_scalar(SELECT_CANDIDATES_QUERY)
                .bind(expire_after_sec)
                .fetch_all(&mut *tx)
                .await?;

            if candidate_ids.is_empty() {
                break;
            }

            let candidate_count = candidate_ids.len();
            let placeholders = std::iter::repeat_n("?", candidate_count)
                .collect::<Vec<_>>()
                .join(",");
            let delete_stmt = format!(
                "DELETE FROM `{table}` WHERE `id` IN ({placeholders}) AND `state` IN \
                 ('{succeeded_state}','{failed_state}','{cancelled_state}');",
                table = JOBS_TABLE_NAME,
                succeeded_state = JobState::Succeeded.as_str(),
                failed_state = JobState::Failed.as_str(),
                cancelled_state = JobState::Cancelled.as_str(),
            );
            let mut delete_query = sqlx::query(&delete_stmt);
            for job_id in &candidate_ids {
                delete_query = delete_query.bind(job_id);
            }
            let rows_affected = delete_query.execute(&mut *tx).await?.rows_affected();

            if rows_affected != candidate_count as u64 {
                return Err(DbError::CorruptedDbState(format!(
                    "expected to delete {candidate_count} rows but only {rows_affected} rows \
                     deleted"
                )));
            }

            deleted_job_ids.extend(candidate_ids);

            if candidate_count < DELETE_BATCH_SIZE {
                break;
            }
        }

        tx.commit().await?;
        Ok(deleted_job_ids)
    }

    async fn get_recoverable_jobs(&self) -> Result<Vec<RecoverableJobContext>, DbError> {
        const SELECT_QUERY: &str = formatcp!(
            "SELECT `id`, `resource_group_id`, `state`, `{parent}`, `{root}`, \
             `compressed_serialized_task_graph`, `compressed_serialized_job_inputs`, \
             `serialized_job_outputs` FROM `{table}` WHERE `state` IN \
             ('{ready_state}','{running_state}','{blocking_state}','{commit_ready_state}','\
             {cleanup_ready_state}');",
            parent = JOB_PARENT_ID_COLUMN_NAME,
            root = JOB_ROOT_ID_COLUMN_NAME,
            table = JOBS_TABLE_NAME,
            ready_state = JobState::Ready.as_str(),
            running_state = JobState::Running.as_str(),
            blocking_state = JobState::BlockingOnChildren.as_str(),
            commit_ready_state = JobState::CommitReady.as_str(),
            cleanup_ready_state = JobState::CleanupReady.as_str(),
        );

        sqlx::query_as::<_, RecoverableJobRowProjection>(SELECT_QUERY)
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(RecoverableJobRowProjection::into_recoverable_job_context)
            .collect()
    }

    async fn has_failed_child(&self, parent_id: JobId) -> Result<bool, DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `id` FROM `{table}` WHERE `{parent}` = ? AND `state` IN (?, ?) LIMIT 1;",
            table = JOBS_TABLE_NAME,
            parent = JOB_PARENT_ID_COLUMN_NAME,
        );

        let failed_child = sqlx::query_scalar::<_, JobId>(QUERY)
            .persistent(false)
            .bind(parent_id)
            .bind(JobState::Failed)
            .bind(JobState::Cancelled)
            .fetch_optional(&self.pool)
            .await?;
        Ok(failed_child.is_some())
    }
}

#[async_trait]
impl ResourceGroupManagement for MariaDbStorageConnector {
    async fn add(
        &self,
        credentials: ExternalResourceGroupCredentials,
    ) -> Result<ResourceGroupId, DbError> {
        const QUERY: &str = formatcp!(
            "INSERT INTO `{table}` (`external_id`, `password`) VALUES (?, ?);",
            table = RESOURCE_GROUPS_TABLE_NAME,
        );

        let resource_group_id = sqlx::query(QUERY)
            .bind(credentials.get_external_resource_group_id())
            .bind(credentials.get_password())
            .execute(&self.pool)
            .await
            .map_err(|e| match e {
                e if is_duplicate_entry(&e) => DbError::ResourceGroupAlreadyExists(
                    credentials.get_external_resource_group_id().to_owned(),
                ),
                e => e.into(),
            })?
            .last_insert_id();

        Ok(ResourceGroupId::from(resource_group_id))
    }

    async fn add_or_verify(
        &self,
        credentials: ExternalResourceGroupCredentials,
    ) -> Result<ResourceGroupId, DbError> {
        const INSERT_QUERY: &str = formatcp!(
            "INSERT INTO `{table}` (`external_id`, `password`) VALUES (?, ?);",
            table = RESOURCE_GROUPS_TABLE_NAME,
        );
        const SELECT_QUERY: &str = formatcp!(
            "SELECT `id`, `password` FROM `{table}` WHERE `external_id` = ? LOCK IN SHARE MODE;",
            table = RESOURCE_GROUPS_TABLE_NAME,
        );

        run_read_committed_tx(self.pool.clone(), async move |connection| {
            let mut tx = connection.begin().await?;
            if let Some((resource_group_id, stored_password)) =
                sqlx::query_as::<_, (ResourceGroupId, Vec<u8>)>(SELECT_QUERY)
                    .bind(credentials.get_external_resource_group_id())
                    .fetch_optional(&mut *tx)
                    .await?
            {
                if !bool::from(stored_password.ct_eq(credentials.get_password())) {
                    return Err(DbError::InvalidPassword(resource_group_id));
                }
                tx.commit().await?;
                return Ok(resource_group_id);
            }

            let inserted_resource_group_id = match sqlx::query(INSERT_QUERY)
                .bind(credentials.get_external_resource_group_id())
                .bind(credentials.get_password())
                .execute(&mut *tx)
                .await
            {
                Ok(result) => Some(ResourceGroupId::from(result.last_insert_id())),
                Err(error) if is_duplicate_entry(&error) => None,
                Err(error) => return Err(error.into()),
            };

            if let Some(resource_group_id) = inserted_resource_group_id {
                tx.commit().await?;
                return Ok(resource_group_id);
            }

            let (resource_group_id, stored_password) =
                sqlx::query_as::<_, (ResourceGroupId, Vec<u8>)>(SELECT_QUERY)
                    .bind(credentials.get_external_resource_group_id())
                    .fetch_one(&mut *tx)
                    .await?;
            if !bool::from(stored_password.ct_eq(credentials.get_password())) {
                return Err(DbError::InvalidPassword(resource_group_id));
            }
            tx.commit().await?;
            Ok(resource_group_id)
        })
        .await
    }

    async fn verify(
        &self,
        resource_group_id: ResourceGroupId,
        password: &[u8],
    ) -> Result<(), DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `password` FROM `{table}` WHERE `id` = ?;",
            table = RESOURCE_GROUPS_TABLE_NAME,
        );

        let Some(stored_password) = sqlx::query_scalar::<_, Vec<u8>>(QUERY)
            .bind(resource_group_id)
            .fetch_optional(&self.pool)
            .await?
        else {
            return Err(DbError::ResourceGroupNotFound(resource_group_id));
        };

        if stored_password.ct_eq(password).into() {
            Ok(())
        } else {
            Err(DbError::InvalidPassword(resource_group_id))
        }
    }

    async fn delete(&self, _resource_group_id: ResourceGroupId) -> Result<(), DbError> {
        todo!("not implemented")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, MySqlEnum)]
enum ExecutionManagerState {
    Alive,
    Dead,
}

#[async_trait]
impl ExecutionManagerLivenessManagement for MariaDbStorageConnector {
    async fn register_execution_manager(
        &self,
        ip_address: IpAddr,
        resource_group_credentials: Option<ExternalResourceGroupCredentials>,
    ) -> Result<(ExecutionManagerId, Option<ResourceGroupId>), DbError> {
        const INSERT_QUERY: &str = formatcp!(
            "INSERT INTO `{table}` (`ip_address`, `resource_group_id`) VALUES (?, ?);",
            table = EXECUTION_MANAGERS_TABLE_NAME,
        );
        const SELECT_RESOURCE_GROUP_QUERY: &str = formatcp!(
            "SELECT `id` FROM `{table}` WHERE `external_id` = ?;",
            table = RESOURCE_GROUPS_TABLE_NAME,
        );

        let resource_group_id = if let Some(credentials) = resource_group_credentials {
            let resource_group_id =
                sqlx::query_scalar::<_, ResourceGroupId>(SELECT_RESOURCE_GROUP_QUERY)
                    .bind(credentials.get_external_resource_group_id())
                    .fetch_optional(&self.pool)
                    .await?
                    .ok_or_else(|| {
                        DbError::ExternalResourceGroupNotFound(
                            credentials.get_external_resource_group_id().to_owned(),
                        )
                    })?;
            ResourceGroupManagement::verify(self, resource_group_id, credentials.get_password())
                .await?;
            Some(resource_group_id)
        } else {
            None
        };

        let execution_manager_id = sqlx::query(INSERT_QUERY)
            .bind(ip_address.to_string())
            .bind(resource_group_id)
            .execute(&self.pool)
            .await
            .map(|result| ExecutionManagerId::from(result.last_insert_id()))
            .map_err(DbError::from)?;
        Ok((execution_manager_id, resource_group_id))
    }

    async fn update_execution_manager_heartbeat(
        &self,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<(), DbError> {
        const SELECT_STATE_FOR_UPDATE_QUERY: &str = formatcp!(
            "SELECT `state` FROM `{table}` WHERE `id` = ? FOR UPDATE;",
            table = EXECUTION_MANAGERS_TABLE_NAME,
        );
        const UPDATE_QUERY: &str = formatcp!(
            "UPDATE `{table}` SET `last_heartbeat_at` = CURRENT_TIMESTAMP WHERE `id` = ? AND \
             `state` = '{alive_state}';",
            table = EXECUTION_MANAGERS_TABLE_NAME,
            alive_state = ExecutionManagerState::Alive.as_str(),
        );

        let mut tx = self.pool.begin().await?;

        let state = sqlx::query_scalar::<_, ExecutionManagerState>(SELECT_STATE_FOR_UPDATE_QUERY)
            .bind(execution_manager_id)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or(DbError::IllegalExecutionManagerId(execution_manager_id))?;

        if state == ExecutionManagerState::Dead {
            return Err(DbError::ExecutionManagerAlreadyDead(execution_manager_id));
        }

        sqlx::query(UPDATE_QUERY)
            .bind(execution_manager_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(())
    }

    async fn is_execution_manager_alive(
        &self,
        execution_manager_id: ExecutionManagerId,
    ) -> Result<bool, DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `state` FROM `{table}` WHERE `id` = ?;",
            table = EXECUTION_MANAGERS_TABLE_NAME,
        );

        let Some(state) = sqlx::query_scalar::<_, ExecutionManagerState>(QUERY)
            .bind(execution_manager_id)
            .fetch_optional(&self.pool)
            .await?
        else {
            return Err(DbError::IllegalExecutionManagerId(execution_manager_id));
        };

        match state {
            ExecutionManagerState::Alive => Ok(true),
            ExecutionManagerState::Dead => Ok(false),
        }
    }

    async fn get_dead_execution_managers(
        &self,
        stale_after_sec: u64,
    ) -> Result<Vec<ExecutionManagerId>, DbError> {
        run_read_committed_tx(self.pool.clone(), async move |connection| {
            get_dead_execution_managers(connection, stale_after_sec).await
        })
        .await
    }
}

#[async_trait]
impl SchedulerRegistrationManagement for MariaDbStorageConnector {
    async fn register_scheduler(&self, host: &str, port: u16) -> Result<SchedulerId, DbError> {
        const DELETE_QUERY: &str =
            formatcp!("DELETE FROM `{table}`;", table = SCHEDULERS_TABLE_NAME,);
        const INSERT_QUERY: &str = formatcp!(
            "INSERT INTO `{table}` (`host`, `port`) VALUES (?, ?);",
            table = SCHEDULERS_TABLE_NAME,
        );

        let mut tx = self.pool.begin().await?;
        sqlx::query(DELETE_QUERY).execute(&mut *tx).await?;
        let scheduler_id = sqlx::query(INSERT_QUERY)
            .bind(host)
            .bind(port)
            .execute(&mut *tx)
            .await?
            .last_insert_id();
        tx.commit().await?;
        Ok(SchedulerId::from(scheduler_id))
    }

    async fn get_schedulers(&self) -> Result<Vec<RegisteredScheduler>, DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `id`, `host`, `port` FROM `{table}` ORDER BY `id` ASC;",
            table = SCHEDULERS_TABLE_NAME,
        );

        let rows: Vec<SchedulerRowProjection> = sqlx::query_as(QUERY).fetch_all(&self.pool).await?;
        rows.into_iter()
            .map(SchedulerRowProjection::into_registered_scheduler)
            .collect()
    }

    async fn is_scheduler_registered(&self, scheduler_id: SchedulerId) -> Result<bool, DbError> {
        const QUERY: &str = formatcp!(
            "SELECT `id` FROM `{table}` WHERE `id` = ?;",
            table = SCHEDULERS_TABLE_NAME,
        );

        let registered_scheduler_id: Option<SchedulerId> = sqlx::query_scalar(QUERY)
            .bind(scheduler_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(registered_scheduler_id.is_some())
    }
}

impl SessionManagement for MariaDbStorageConnector {
    fn session_id(&self) -> SessionId {
        self.session_id
    }
}

impl DbStorage for MariaDbStorageConnector {}

/// `MySQL` error number for foreign key constraint violation.
const MYSQL_ER_FK_CONSTRAINT: u16 = 1452;

/// `MySQL` error number for duplicate entry.
const MYSQL_ER_DUP_ENTRY: u16 = 1062;

/// `MySQL` error number for a lock wait timeout.
const MYSQL_ER_LOCK_WAIT_TIMEOUT: u16 = 1205;

/// `MySQL` error number for a missing privilege on an operation.
const MYSQL_ER_SPECIFIC_ACCESS_DENIED: u16 = 1227;

/// `MySQL` error number for a missing privilege on a table.
const MYSQL_ER_TABLE_ACCESS_DENIED: u16 = 1142;

/// `MySQL` error number for a packet larger than `max_allowed_packet`.
const MYSQL_ER_NET_PACKET_TOO_LARGE: u16 = 1153;

/// `MySQL` error number for a value too long for its column.
const MYSQL_ER_DATA_TOO_LONG: u16 = 1406;

const RESOURCE_GROUPS_TABLE_NAME: &str = "resource_groups";
const JOBS_TABLE_NAME: &str = "jobs";
const EXECUTION_MANAGERS_TABLE_NAME: &str = "execution_managers";
const SCHEDULERS_TABLE_NAME: &str = "schedulers";
const SESSIONS_TABLE_NAME: &str = "sessions";
const STORAGE_FENCE_TABLE_NAME: &str = "storage_fence";

const JOB_PARENT_ID_COLUMN_NAME: &str = "parent_id";
const JOB_ROOT_ID_COLUMN_NAME: &str = "root_id";
const JOB_PARENT_INDEX_NAME: &str = "job_parent";

/// The upper bound of the payload bound to one statement.
const MAX_STATEMENT_PAYLOAD: u64 = 4 * 1024 * 1024;

/// The bytes of a packet that an `UPDATE` statement's text and its values other than one row's
/// outputs may take.
const MAX_STATEMENT_OVERHEAD: u64 = 64 * 1024;

/// The maximum number of IDs in one statement's `IN` list.
const IN_LIST_MAX_IDS: usize = 500;

/// The number of seconds that a database session may idle before the server closes it, which must
/// stay above the pool's idle timeout of 10 minutes.
const SESSION_WAIT_TIMEOUT_SEC: u64 = 900;

#[must_use]
const fn resource_groups_creation_query() -> &'static str {
    formatcp!(
        r"
CREATE TABLE IF NOT EXISTS `{RESOURCE_GROUPS_TABLE_NAME}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `external_id` VARCHAR(256) NOT NULL,
  `password` VARBINARY(2048) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `external_resource_group_id` (`external_id`)
);"
    )
}

#[must_use]
const fn jobs_creation_query() -> &'static str {
    formatcp!(
        r"
CREATE TABLE IF NOT EXISTS `{JOBS_TABLE_NAME}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `resource_group_id` BIGINT UNSIGNED NOT NULL,
  `state` {state_enum} NOT NULL DEFAULT {default_state},
  `compressed_serialized_task_graph` LONGBLOB NOT NULL,
  `compressed_serialized_job_inputs` LONGBLOB NOT NULL,
  `serialized_job_outputs` LONGBLOB,
  `error_message` LONGTEXT,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `ended_at` TIMESTAMP,
  `max_num_retries` INT UNSIGNED NOT NULL DEFAULT 0,
  `num_retries` INT UNSIGNED NOT NULL DEFAULT 0,
  `{JOB_PARENT_ID_COLUMN_NAME}` BIGINT UNSIGNED NULL DEFAULT NULL,
  `{JOB_ROOT_ID_COLUMN_NAME}` BIGINT UNSIGNED NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  INDEX `job_state` (`state`),
  INDEX `{JOB_PARENT_INDEX_NAME}` (`{JOB_PARENT_ID_COLUMN_NAME}`),
  CONSTRAINT `job_resource_group` FOREIGN KEY (`resource_group_id`)
    REFERENCES `{RESOURCE_GROUPS_TABLE_NAME}` (`id`)
    ON UPDATE RESTRICT ON DELETE RESTRICT
);",
        state_enum = JobState::as_mysql_enum_decl(),
        default_state = JobState::Ready.as_quoted_str(),
    )
}

#[must_use]
const fn execution_managers_creation_query() -> &'static str {
    formatcp!(
        r"
CREATE TABLE IF NOT EXISTS `{EXECUTION_MANAGERS_TABLE_NAME}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `ip_address` VARCHAR(45) NOT NULL,
  `resource_group_id` BIGINT UNSIGNED,
  `state` {state_enum} NOT NULL DEFAULT {default_state},
  `last_heartbeat_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `death_confirmed_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  INDEX `execution_manager_liveness` (`state`, `last_heartbeat_at`),
  CONSTRAINT `execution_manager_resource_group` FOREIGN KEY (`resource_group_id`)
    REFERENCES `{RESOURCE_GROUPS_TABLE_NAME}` (`id`)
    ON UPDATE RESTRICT ON DELETE RESTRICT
);",
        state_enum = ExecutionManagerState::as_mysql_enum_decl(),
        default_state = ExecutionManagerState::Alive.as_quoted_str(),
    )
}

#[must_use]
const fn schedulers_creation_query() -> &'static str {
    formatcp!(
        r"
CREATE TABLE IF NOT EXISTS `{SCHEDULERS_TABLE_NAME}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `host` VARCHAR(255) NOT NULL,
  `port` SMALLINT UNSIGNED NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
);"
    )
}

#[must_use]
const fn sessions_creation_query() -> &'static str {
    formatcp!(
        r"
CREATE TABLE IF NOT EXISTS `{SESSIONS_TABLE_NAME}` (
  `session_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`session_id`)
);"
    )
}

#[must_use]
const fn storage_fence_creation_query() -> &'static str {
    formatcp!(
        r"
CREATE TABLE IF NOT EXISTS `{STORAGE_FENCE_TABLE_NAME}` (
  `id` BIGINT UNSIGNED NOT NULL,
  `session_id` BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (`id`)
);"
    )
}

/// A raw row selected from the job table representing a recoverable job.
#[derive(sqlx::FromRow)]
struct RecoverableJobRowProjection {
    id: JobId,
    resource_group_id: ResourceGroupId,
    state: JobState,
    parent_id: Option<JobId>,
    root_id: Option<JobId>,
    compressed_serialized_task_graph: Vec<u8>,
    compressed_serialized_job_inputs: Vec<u8>,
    serialized_job_outputs: Option<Vec<u8>>,
}

impl RecoverableJobRowProjection {
    /// Converts the row projection into [`RecoverableJobContext`].
    ///
    /// # Returns
    ///
    /// The context of the recoverable job on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`SerializedTaskOutputs::deserialize_from_raw`]'s return values on failure.
    /// * Forwards [`ValidatedJobSubmission::create`]'s return values as
    ///   [`DbError::CorruptedDbState`] on failure.
    fn into_recoverable_job_context(self) -> Result<RecoverableJobContext, DbError> {
        let submission = ValidatedJobSubmission::create(
            self.compressed_serialized_task_graph,
            self.compressed_serialized_job_inputs,
        )
        .map_err(|e| DbError::CorruptedDbState(e.to_string()))?;
        let outputs = self
            .serialized_job_outputs
            .map(|outputs| {
                SerializedTaskOutputs::deserialize_from_raw(&outputs)
                    .map_err(|e| DbError::ValueDeserializationFailure(Box::new(e)))
            })
            .transpose()?;
        Ok(RecoverableJobContext {
            id: self.id,
            resource_group_id: self.resource_group_id,
            state: self.state,
            parent_id: self.parent_id,
            root_id: self.root_id,
            submission,
            outputs,
        })
    }
}

/// A raw row selected from the schedulers table.
#[derive(sqlx::FromRow)]
struct SchedulerRowProjection {
    id: SchedulerId,
    host: String,
    port: u16,
}

impl SchedulerRowProjection {
    /// Converts the row projection into [`RegisteredScheduler`].
    ///
    /// # Returns
    ///
    /// The registered scheduler on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`DbError::CorruptedDbState`] if the scheduler host is empty.
    fn into_registered_scheduler(self) -> Result<RegisteredScheduler, DbError> {
        let host = Host::new(self.host).map_err(|_| {
            DbError::CorruptedDbState(format!("scheduler `{}` has an empty host", self.id))
        })?;
        Ok(RegisteredScheduler {
            id: self.id,
            host,
            port: self.port,
        })
    }
}

/// One guarded `UPDATE` statement of a tree transition: every row moves from the same state to the
/// same state and gets the same error message.
#[derive(Debug)]
struct UpdateStatement {
    from: JobState,
    to: JobState,
    error_message: Option<String>,
    rows: Vec<UpdateRow>,
}

impl UpdateStatement {
    /// Groups `updates` into guarded statements.
    ///
    /// Updates are grouped by their `from` and `to` states, whether they carry outputs, and their
    /// error message. Groups that carry outputs come first. Each group is cut into statements of
    /// at most [`IN_LIST_MAX_IDS`] rows, and a group that carries outputs is also cut where a
    /// statement's outputs would exceed `max_payload`. Every statement holds at least one row.
    ///
    /// # Returns
    ///
    /// The statements, in the order they must run, on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`DbError::ValueSerializationFailure`] if any outputs fail to serialize.
    /// * [`DbError::OutputsTooLarge`] if any serialized outputs exceed `max_outputs_size`.
    fn group(
        updates: Vec<JobStateUpdate>,
        max_payload: u64,
        max_outputs_size: u64,
    ) -> Result<Vec<Self>, DbError> {
        let mut groups: Vec<Self> = Vec::new();
        for update in updates {
            let serialized_outputs = update
                .outputs
                .map(|outputs| {
                    SerializedTaskOutputs::serialize_with_size_hint(&outputs)
                        .map(|serialized| serialized.to_raw())
                        .map_err(|e| DbError::ValueSerializationFailure(Box::new(e)))
                })
                .transpose()?;
            if let Some(size) = serialized_outputs
                .as_ref()
                .map(|outputs| outputs.len() as u64)
                .filter(|size| *size > max_outputs_size)
            {
                return Err(DbError::OutputsTooLarge {
                    job_id: update.job_id,
                    size,
                    limit: max_outputs_size,
                });
            }
            let row = UpdateRow {
                job_id: update.job_id,
                serialized_outputs,
            };
            let has_outputs = row.serialized_outputs.is_some();
            if let Some(group) = groups.iter_mut().find(|group| {
                group.from == update.from
                    && group.to == update.to
                    && group.has_outputs() == has_outputs
                    && group.error_message == update.error_message
            }) {
                group.rows.push(row);
            } else {
                groups.push(Self {
                    from: update.from,
                    to: update.to,
                    error_message: update.error_message,
                    rows: vec![row],
                });
            }
        }
        groups.sort_by_key(|group| !group.has_outputs());

        let mut statements = Vec::new();
        for group in groups {
            let mut rows = group.rows.into_iter().peekable();
            while rows.peek().is_some() {
                let mut chunk = Vec::new();
                let mut payload: u64 = 0;
                while let Some(row) = rows.peek() {
                    let row_payload = row
                        .serialized_outputs
                        .as_ref()
                        .map_or(0, |outputs| outputs.len() as u64);
                    if !chunk.is_empty()
                        && (chunk.len() == IN_LIST_MAX_IDS || payload + row_payload > max_payload)
                    {
                        break;
                    }
                    payload += row_payload;
                    chunk.push(rows.next().expect("a peeked row should exist"));
                }
                statements.push(Self {
                    from: group.from,
                    to: group.to,
                    error_message: group.error_message.clone(),
                    rows: chunk,
                });
            }
        }
        Ok(statements)
    }

    /// Runs the statement on `connection` and checks that it matched every one of its rows.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`DbError::CorruptedDbState`] if fewer rows matched than the statement updates, that is,
    ///   if a row is missing or not in the statement's `from` state.
    /// * Forwards [`sqlx::query::Query::execute`]'s return values on failure.
    async fn execute(&self, connection: &mut MySqlConnection) -> Result<(), DbError> {
        let mut sql = format!("UPDATE `{JOBS_TABLE_NAME}` FORCE INDEX (PRIMARY) SET `state` = ?");
        if self.has_outputs() {
            sql.push_str(", `serialized_job_outputs` = CASE `id`");
            sql.push_str(&" WHEN ? THEN ?".repeat(self.rows.len()));
            sql.push_str(" END");
        }
        if self.error_message.is_some() {
            sql.push_str(", `error_message` = ?");
        }
        if self.to.is_terminal() {
            sql.push_str(", `ended_at` = CURRENT_TIMESTAMP");
        }
        sql.push_str(" WHERE `id` IN (");
        sql.push_str(&vec!["?"; self.rows.len()].join(","));
        sql.push_str(") AND `state` = ?;");

        let mut query = sqlx::query(&sql).bind(self.to);
        if self.has_outputs() {
            for row in &self.rows {
                query = query
                    .bind(row.job_id)
                    .bind(row.serialized_outputs.as_deref());
            }
        }
        if let Some(error_message) = &self.error_message {
            query = query.bind(error_message);
        }
        for row in &self.rows {
            query = query.bind(row.job_id);
        }
        let matched_rows = query
            .bind(self.from)
            .execute(connection)
            .await?
            .rows_affected();

        if matched_rows != self.rows.len() as u64 {
            let job_ids: Vec<JobId> = self.rows.iter().map(|row| row.job_id).collect();
            return Err(DbError::CorruptedDbState(format!(
                "expected jobs {job_ids:?} in state {from} to move to {to}, but only \
                 {matched_rows} of them matched",
                from = self.from,
                to = self.to,
            )));
        }
        Ok(())
    }

    /// # Returns
    ///
    /// Whether the statement's rows carry outputs.
    fn has_outputs(&self) -> bool {
        self.rows
            .first()
            .is_some_and(|row| row.serialized_outputs.is_some())
    }
}

/// A row of an [`UpdateStatement`].
#[derive(Debug)]
struct UpdateRow {
    job_id: JobId,
    serialized_outputs: Option<Vec<u8>>,
}

/// Cuts `inserts` into the chunks of one multi-row `INSERT` each: at most [`IN_LIST_MAX_IDS`] rows,
/// cut where a chunk's task graphs and job inputs would exceed `max_payload`. Every chunk holds at
/// least one row.
///
/// # Returns
///
/// The chunks, in the order of `inserts`.
fn chunk_child_inserts<'insert_lifetime, 'buffer_lifetime>(
    inserts: &'insert_lifetime [ChildJobInsert<'buffer_lifetime>],
    max_payload: u64,
) -> Vec<&'insert_lifetime [ChildJobInsert<'buffer_lifetime>]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut payload: u64 = 0;
    for (index, insert) in inserts.iter().enumerate() {
        let row_payload =
            (insert.compressed_task_graph.len() + insert.compressed_job_inputs.len()) as u64;
        if index > start
            && (index - start == IN_LIST_MAX_IDS || payload + row_payload > max_payload)
        {
            chunks.push(&inserts[start..index]);
            start = index;
            payload = 0;
        }
        payload += row_payload;
    }
    if start < inserts.len() {
        chunks.push(&inserts[start..]);
    }
    chunks
}

/// Inserts `chunk` as `Running` child jobs with one multi-row `INSERT … RETURNING`.
///
/// # Returns
///
/// The IDs of the inserted jobs, in the order of `chunk`, on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * [`DbError::CorruptedDbState`] if the database returns a different number of IDs than rows.
/// * Forwards [`sqlx::query::QueryScalar::fetch_all`]'s return values on failure.
async fn insert_children(
    connection: &mut MySqlConnection,
    chunk: &[ChildJobInsert<'_>],
) -> Result<Vec<JobId>, DbError> {
    let mut sql = format!(
        "INSERT INTO `{JOBS_TABLE_NAME}` (`resource_group_id`, `{JOB_PARENT_ID_COLUMN_NAME}`, \
         `{JOB_ROOT_ID_COLUMN_NAME}`, `state`, `compressed_serialized_task_graph`, \
         `compressed_serialized_job_inputs`) VALUES "
    );
    sql.push_str(&vec!["(?, ?, ?, ?, ?, ?)"; chunk.len()].join(", "));
    sql.push_str(" RETURNING `id`;");

    let mut query = sqlx::query_scalar::<_, JobId>(&sql);
    for insert in chunk {
        query = query
            .bind(insert.resource_group_id)
            .bind(insert.parent_id)
            .bind(insert.root_id)
            .bind(JobState::Running)
            .bind(insert.compressed_task_graph)
            .bind(insert.compressed_job_inputs);
    }
    let child_ids = query.fetch_all(connection).await?;
    if child_ids.len() != chunk.len() {
        return Err(DbError::CorruptedDbState(format!(
            "inserted {} child jobs, but the database returned {} IDs",
            chunk.len(),
            child_ids.len()
        )));
    }
    Ok(child_ids)
}

/// Upgrades an existing `jobs` table to the current schema.
///
/// The table's columns and indexes are read first, and only the missing parts are added, so an
/// up-to-date table is not altered. Every alteration is idempotent, so concurrent upgrades are
/// safe.
///
/// # Returns
///
/// The number of DDL statements issued on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`sqlx::query::QueryAs::fetch_all`]'s return values on failure.
/// * Forwards [`sqlx::query::QueryScalar::fetch_optional`]'s return values on failure.
/// * Forwards [`sqlx::Pool::acquire`]'s return values on failure.
/// * Forwards [`sqlx::query::Query::execute`]'s return values on failure.
/// * Forwards [`execute_ddl_with_lock_wait_retries`]'s return values on failure.
async fn upgrade_jobs_schema(pool: &MySqlPool) -> Result<usize, DbError> {
    const SET_LOCK_WAIT_TIMEOUT_QUERY: &str = "SET SESSION lock_wait_timeout = 5;";
    const SELECT_COLUMNS_QUERY: &str = formatcp!(
        "SELECT `COLUMN_NAME`, `COLUMN_TYPE` FROM `information_schema`.`COLUMNS` WHERE \
         `TABLE_SCHEMA` = DATABASE() AND `TABLE_NAME` = '{JOBS_TABLE_NAME}';"
    );
    const SELECT_PARENT_INDEX_QUERY: &str = formatcp!(
        "SELECT `INDEX_NAME` FROM `information_schema`.`STATISTICS` WHERE `TABLE_SCHEMA` = \
         DATABASE() AND `TABLE_NAME` = '{JOBS_TABLE_NAME}' AND `INDEX_NAME` = \
         '{JOB_PARENT_INDEX_NAME}' LIMIT 1;"
    );
    const ADD_PARENT_ID_CLAUSE: &str = formatcp!(
        "ADD COLUMN IF NOT EXISTS `{JOB_PARENT_ID_COLUMN_NAME}` BIGINT UNSIGNED NULL DEFAULT NULL"
    );
    const ADD_ROOT_ID_CLAUSE: &str = formatcp!(
        "ADD COLUMN IF NOT EXISTS `{JOB_ROOT_ID_COLUMN_NAME}` BIGINT UNSIGNED NULL DEFAULT NULL"
    );
    const ADD_PARENT_INDEX_CLAUSE: &str = formatcp!(
        "ADD INDEX IF NOT EXISTS `{JOB_PARENT_INDEX_NAME}` (`{JOB_PARENT_ID_COLUMN_NAME}`)"
    );
    const MODIFY_STATE_QUERY: &str = formatcp!(
        "ALTER TABLE `{JOBS_TABLE_NAME}` MODIFY `state` {state_enum} NOT NULL DEFAULT \
         {default_state};",
        state_enum = JobState::as_mysql_enum_decl(),
        default_state = JobState::Ready.as_quoted_str(),
    );

    let columns: Vec<(String, String)> =
        sqlx::query_as(SELECT_COLUMNS_QUERY).fetch_all(pool).await?;
    let has_column = |name: &str| columns.iter().any(|(column, _)| column == name);
    let has_parent_index = sqlx::query_scalar::<_, String>(SELECT_PARENT_INDEX_QUERY)
        .fetch_optional(pool)
        .await?
        .is_some();

    let missing_clauses: Vec<&str> = [
        (!has_column(JOB_PARENT_ID_COLUMN_NAME), ADD_PARENT_ID_CLAUSE),
        (!has_column(JOB_ROOT_ID_COLUMN_NAME), ADD_ROOT_ID_CLAUSE),
        (!has_parent_index, ADD_PARENT_INDEX_CLAUSE),
    ]
    .into_iter()
    .filter_map(|(is_missing, clause)| is_missing.then_some(clause))
    .collect();
    let state_has_blocking_on_children = columns.iter().any(|(column, column_type)| {
        column == "state" && column_type.contains(JobState::BlockingOnChildren.as_quoted_str())
    });

    let mut ddl_statements = Vec::new();
    if !missing_clauses.is_empty() {
        ddl_statements.push(format!(
            "ALTER TABLE `{JOBS_TABLE_NAME}` {};",
            missing_clauses.join(", ")
        ));
    }
    if !state_has_blocking_on_children {
        ddl_statements.push(MODIFY_STATE_QUERY.to_owned());
    }
    if ddl_statements.is_empty() {
        return Ok(0);
    }

    let mut connection = pool.acquire().await?.detach();
    let result: Result<(), DbError> = async {
        sqlx::query(SET_LOCK_WAIT_TIMEOUT_QUERY)
            .execute(&mut connection)
            .await?;
        for ddl_statement in &ddl_statements {
            execute_ddl_with_lock_wait_retries(&mut connection, ddl_statement).await?;
        }
        Ok(())
    }
    .await;
    close_detached_connection(connection).await;
    result?;

    tracing::info!(
        num_ddl_statements = ddl_statements.len(),
        "Upgraded the jobs table to the current schema."
    );
    Ok(ddl_statements.len())
}

/// Executes `ddl_statement` on `connection`, retrying it for as long as it times out waiting for
/// a metadata lock.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`sqlx::query::Query::execute`]'s return values on failure, other than a lock wait
///   timeout.
async fn execute_ddl_with_lock_wait_retries(
    connection: &mut MySqlConnection,
    ddl_statement: &str,
) -> Result<(), DbError> {
    let mut attempt: usize = 1;
    loop {
        match sqlx::query(ddl_statement).execute(&mut *connection).await {
            Ok(_) => return Ok(()),
            Err(e) if is_lock_wait_timeout(&e) => {
                tracing::warn!(
                    attempt,
                    "Timed out waiting for the metadata lock to upgrade the jobs table. Another \
                     connection holds the table in an open transaction. Retrying."
                );
                attempt += 1;
            }
            Err(e) => return Err(e.into()),
        }
    }
}

/// Closes a connection detached from the pool, logging a failure instead of returning it.
async fn close_detached_connection(connection: MySqlConnection) {
    if let Err(e) = connection.close().await {
        tracing::warn!(error = % e, "Failed to close a detached database connection.");
    }
}

/// Sets the fence row `fence_id` to `session_id`, so that every storage session that connected
/// earlier with the same fence ID fails its next write with [`DbError::Fenced`].
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`sqlx::Pool::acquire`]'s return values on failure.
/// * Forwards [`sqlx::query::Query::execute`]'s return values on failure, including a lock wait
///   timeout on the last attempt.
/// * Forwards [`sqlx::query::QueryScalar::fetch_one`]'s return values on failure.
async fn bump_fence(pool: &MySqlPool, fence_id: u64, session_id: SessionId) -> Result<(), DbError> {
    const MAX_ATTEMPTS: usize = 5;
    const BLOCKER_LOOKUP_INTERVAL: Duration = Duration::from_secs(1);
    const SET_LOCK_WAIT_TIMEOUT_QUERY: &str = "SET SESSION innodb_lock_wait_timeout = 5;";
    const SELECT_CONNECTION_ID_QUERY: &str = "SELECT CONNECTION_ID();";
    const BUMP_QUERY: &str = formatcp!(
        "UPDATE `{STORAGE_FENCE_TABLE_NAME}` SET `session_id` = GREATEST(`session_id`, ?) WHERE \
         `id` = ?;"
    );

    let mut connection = pool.acquire().await?.detach();
    let result: Result<(), DbError> = async {
        sqlx::query(SET_LOCK_WAIT_TIMEOUT_QUERY)
            .execute(&mut connection)
            .await?;
        let connection_id: u64 = sqlx::query_scalar(SELECT_CONNECTION_ID_QUERY)
            .fetch_one(&mut connection)
            .await?;

        let mut is_blocker_lookup_enabled = true;
        let mut attempt = 1;
        loop {
            let update = sqlx::query(BUMP_QUERY)
                .bind(session_id)
                .bind(fence_id)
                .execute(&mut connection);
            tokio::pin!(update);
            let mut blocker_lookup_ticker = tokio::time::interval_at(
                tokio::time::Instant::now() + BLOCKER_LOOKUP_INTERVAL,
                BLOCKER_LOOKUP_INTERVAL,
            );
            let result = loop {
                tokio::select! {
                    result = &mut update => break result,
                    _ = blocker_lookup_ticker.tick(), if is_blocker_lookup_enabled => {
                        match kill_fence_blockers(pool, connection_id).await {
                            Ok(()) => {}
                            Err(e) if is_access_denied(&e) => {
                                tracing::warn!(
                                    error = % e,
                                    "Cannot look up the connections that block the storage fence. \
                                     Relying on the lock wait timeout."
                                );
                                is_blocker_lookup_enabled = false;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = % e,
                                    "Failed to kill the connections that block the storage fence."
                                );
                            }
                        }
                    }
                }
            };

            match result {
                Ok(_) => return Ok(()),
                Err(e) if is_lock_wait_timeout(&e) && attempt < MAX_ATTEMPTS => {
                    tracing::warn!(
                        attempt,
                        "Timed out waiting for the storage fence. Retrying."
                    );
                    attempt += 1;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
    .await;
    close_detached_connection(connection).await;
    result
}

/// Kills every connection whose transaction blocks the connection `waiting_connection_id`.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`sqlx::query::QueryScalar::fetch_all`]'s return values on failure.
async fn kill_fence_blockers(
    pool: &MySqlPool,
    waiting_connection_id: u64,
) -> Result<(), sqlx::Error> {
    const SELECT_BLOCKERS_QUERY: &str =
        "SELECT DISTINCT `b`.`trx_mysql_thread_id` FROM `information_schema`.`INNODB_LOCK_WAITS` \
         `w` JOIN `information_schema`.`INNODB_TRX` `r` ON `r`.`trx_id` = `w`.`requesting_trx_id` \
         JOIN `information_schema`.`INNODB_TRX` `b` ON `b`.`trx_id` = `w`.`blocking_trx_id` JOIN \
         `information_schema`.`PROCESSLIST` `p` ON `p`.`ID` = `b`.`trx_mysql_thread_id` WHERE \
         `r`.`trx_mysql_thread_id` = ? AND `b`.`trx_mysql_thread_id` <> ? AND `p`.`DB` = \
         DATABASE() AND `b`.`trx_started` <= `r`.`trx_wait_started`;";

    let blocker_connection_ids: Vec<u64> = sqlx::query_scalar(SELECT_BLOCKERS_QUERY)
        .bind(waiting_connection_id)
        .bind(waiting_connection_id)
        .fetch_all(pool)
        .await?;
    for blocker_connection_id in blocker_connection_ids {
        match sqlx::raw_sql(&format!("KILL CONNECTION {blocker_connection_id};"))
            .execute(pool)
            .await
        {
            Ok(_) => {
                tracing::warn!(
                    blocker_connection_id,
                    "Killed a connection that blocked the storage fence."
                );
            }
            Err(e) => {
                tracing::warn!(
                    blocker_connection_id,
                    error = % e,
                    "Failed to kill a connection that blocked the storage fence."
                );
            }
        }
    }
    Ok(())
}

/// Runs `tx` on a freshly acquired pooled connection whose next transaction uses the `READ
/// COMMITTED` isolation level.
///
/// A `SET TRANSACTION ISOLATION LEVEL READ COMMITTED` statement is issued on the connection before
/// `tx` runs. Because it omits `SESSION`/`GLOBAL`, it applies only to the next transaction started
/// on that connection. `tx` is expected to begin exactly one transaction to consume the setting,
/// and to commit or roll it back itself.
///
/// If `tx` returns an error, the connection is detached from the pool and closed rather than being
/// released back into it. This ensures a failed attempt can never hand a later, unrelated borrower
/// a connection still carrying the pending isolation change (or a half-open transaction).
///
/// # Type Parameters
///
/// * `ReturnType` - The return type of `tx`.
/// * `TransactionType` - The type of `tx`, which is an async function that takes a mutable
///   reference to the connection.
///
/// # Returns
///
/// The value returned by `tx` on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`sqlx::Pool::acquire`]'s return values on failure.
/// * Forwards [`sqlx::query::Query::execute`]'s return values on failure.
/// * Forwards `tx`'s return values on failure.
async fn run_read_committed_tx<ReturnType, TransactionType>(
    pool: MySqlPool,
    tx: TransactionType,
) -> Result<ReturnType, DbError>
where
    for<'connection_lifetime> TransactionType:
        AsyncFnOnce(&'connection_lifetime mut sqlx::MySqlConnection) -> Result<ReturnType, DbError>,
{
    const SET_READ_COMMITTED: &str = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED";
    let mut conn = pool.acquire().await?;
    sqlx::query(SET_READ_COMMITTED).execute(&mut *conn).await?;
    let result = tx(&mut *conn).await;
    if result.is_err() {
        let _ = conn.detach().close().await;
    }
    result
}

/// Marks stale execution managers dead and returns their IDs, in a transaction run on `conn`.
///
/// # Note
///
/// It is assumed that `conn` must be set to run its next transaction at the `READ COMMITTED`
/// isolation level. Under the default `REPEATABLE REA`, the confirming `SELECT ... FOR UPDATE` can
/// observe a row changed by a concurrent
/// [`ExecutionManagerLivenessManagement::update_execution_manager_heartbeat`] since the
/// transaction's read view was established and failed with a "record has changed" error (error code
/// 1020).
///
/// # Returns
///
/// A vector of dead execution manager IDs (removed from the table) on success.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`sqlx::Connection::begin`]'s return values on failure.
/// * Forwards [`sqlx::query::QueryScalar::fetch_all`]'s return values on failure.
/// * Forwards [`sqlx::query::Query::execute`]'s return values on failure.
/// * Forwards [`sqlx::Transaction::commit`]'s return values on failure.
async fn get_dead_execution_managers(
    conn: &mut sqlx::MySqlConnection,
    stale_after_sec: u64,
) -> Result<Vec<ExecutionManagerId>, DbError> {
    const UPDATE_BATCH_SIZE: usize = 1000;

    const SELECT_CANDIDATES_QUERY: &str = formatcp!(
        "SELECT `id` FROM `{table}` WHERE `state` = '{alive_state}' AND `last_heartbeat_at` < \
         CURRENT_TIMESTAMP - INTERVAL ? SECOND ORDER BY `id`;",
        table = EXECUTION_MANAGERS_TABLE_NAME,
        alive_state = ExecutionManagerState::Alive.as_str(),
    );

    let mut tx = Connection::begin(conn).await?;
    let candidate_ids: Vec<ExecutionManagerId> = sqlx::query_scalar(SELECT_CANDIDATES_QUERY)
        .bind(stale_after_sec)
        .fetch_all(&mut *tx)
        .await?;

    let mut dead_ids: Vec<ExecutionManagerId> = Vec::with_capacity(candidate_ids.len());
    for candidate_batch in candidate_ids.chunks(UPDATE_BATCH_SIZE) {
        let placeholders = std::iter::repeat_n("?", candidate_batch.len())
            .collect::<Vec<_>>()
            .join(",");
        let select_for_update_stmt = format!(
            "SELECT `id` FROM `{table}` FORCE INDEX (PRIMARY) WHERE `id` IN ({placeholders}) AND \
             `state` = '{alive_state}' AND `last_heartbeat_at` < CURRENT_TIMESTAMP - INTERVAL ? \
             SECOND ORDER BY `id` FOR UPDATE;",
            table = EXECUTION_MANAGERS_TABLE_NAME,
            alive_state = ExecutionManagerState::Alive.as_str(),
        );
        let mut select_query = sqlx::query_scalar::<_, ExecutionManagerId>(&select_for_update_stmt);
        for execution_manager_id in candidate_batch {
            select_query = select_query.bind(execution_manager_id);
        }
        let confirmed_ids: Vec<ExecutionManagerId> = select_query
            .bind(stale_after_sec)
            .fetch_all(&mut *tx)
            .await?;

        if confirmed_ids.is_empty() {
            continue;
        }

        let placeholders = std::iter::repeat_n("?", confirmed_ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let update_stmt = format!(
            "UPDATE `{EXECUTION_MANAGERS_TABLE_NAME}` SET `state` = '{dead_state}', \
             `death_confirmed_at` = CURRENT_TIMESTAMP WHERE `id` IN ({placeholders});",
            dead_state = ExecutionManagerState::Dead.as_str(),
        );
        let mut update_query = sqlx::query(&update_stmt);
        for execution_manager_id in &confirmed_ids {
            update_query = update_query.bind(execution_manager_id);
        }
        update_query.execute(&mut *tx).await?;

        dead_ids.extend(confirmed_ids);
    }

    tx.commit().await?;
    Ok(dead_ids)
}

/// # Returns
///
/// Whether `error` is a duplicate-entry error reported by the database.
fn is_duplicate_entry(error: &sqlx::Error) -> bool {
    has_mysql_error_number(error, &[MYSQL_ER_DUP_ENTRY])
}

/// # Returns
///
/// Whether `error` is a lock wait timeout reported by the database.
fn is_lock_wait_timeout(error: &sqlx::Error) -> bool {
    has_mysql_error_number(error, &[MYSQL_ER_LOCK_WAIT_TIMEOUT])
}

/// # Returns
///
/// Whether `error` is a missing-privilege error reported by the database.
fn is_access_denied(error: &sqlx::Error) -> bool {
    has_mysql_error_number(
        error,
        &[
            MYSQL_ER_SPECIFIC_ACCESS_DENIED,
            MYSQL_ER_TABLE_ACCESS_DENIED,
        ],
    )
}

/// # Returns
///
/// Whether `error` is the database rejecting a statement's payload as too large.
fn is_payload_rejection(error: &sqlx::Error) -> bool {
    has_mysql_error_number(
        error,
        &[MYSQL_ER_NET_PACKET_TOO_LARGE, MYSQL_ER_DATA_TOO_LONG],
    )
}

/// # Returns
///
/// Whether `error` is a database error whose `MySQL` error number is one of `numbers`.
fn has_mysql_error_number(error: &sqlx::Error, numbers: &[u16]) -> bool {
    let sqlx::Error::Database(error) = error else {
        return false;
    };
    error
        .try_downcast_ref::<MySqlDatabaseError>()
        .is_some_and(|mysql_error| numbers.contains(&mysql_error.number()))
}

#[cfg(test)]
mod tests {
    use spider_core::job::JobState;
    use spider_core::types::id::JobId;
    use spider_core::types::id::ResourceGroupId;
    use spider_core::types::io::SerializedTaskOutputs;

    use super::IN_LIST_MAX_IDS;
    use super::UpdateStatement;
    use super::chunk_child_inserts;
    use crate::db::ChildJobInsert;
    use crate::db::DbError;
    use crate::db::JobStateUpdate;

    /// # Returns
    ///
    /// The `(from, to, number of rows)` of every statement that `updates` are planned into.
    fn plan_shape(
        updates: Vec<JobStateUpdate>,
        max_payload: u64,
    ) -> Vec<(JobState, JobState, usize)> {
        UpdateStatement::group(updates, max_payload, u64::MAX)
            .expect("planning should succeed")
            .into_iter()
            .map(|statement| (statement.from, statement.to, statement.rows.len()))
            .collect()
    }

    #[test]
    fn updates_are_grouped_by_edge_with_outputs_groups_first() {
        let updates = vec![
            JobStateUpdate::new(JobId::from(1), JobState::Running, JobState::Cancelled),
            JobStateUpdate {
                outputs: Some(vec![vec![1]]),
                ..JobStateUpdate::new(JobId::from(2), JobState::Running, JobState::Succeeded)
            },
            JobStateUpdate::new(JobId::from(3), JobState::Running, JobState::Cancelled),
            JobStateUpdate {
                error_message: Some("a".to_owned()),
                ..JobStateUpdate::new(JobId::from(4), JobState::Running, JobState::Failed)
            },
            JobStateUpdate {
                error_message: Some("b".to_owned()),
                ..JobStateUpdate::new(JobId::from(5), JobState::Running, JobState::Failed)
            },
        ];
        assert_eq!(
            plan_shape(updates, 1024),
            vec![
                (JobState::Running, JobState::Succeeded, 1),
                (JobState::Running, JobState::Cancelled, 2),
                (JobState::Running, JobState::Failed, 1),
                (JobState::Running, JobState::Failed, 1),
            ]
        );
    }

    #[test]
    fn a_group_is_cut_into_chunks_of_at_most_the_in_list_limit() {
        let updates = (0..1_200)
            .map(|id| JobStateUpdate::new(JobId::from(id), JobState::Running, JobState::Cancelled))
            .collect();
        assert_eq!(
            plan_shape(updates, 1024),
            vec![
                (JobState::Running, JobState::Cancelled, IN_LIST_MAX_IDS),
                (JobState::Running, JobState::Cancelled, IN_LIST_MAX_IDS),
                (JobState::Running, JobState::Cancelled, 200),
            ]
        );
    }

    #[test]
    fn an_outputs_group_is_cut_where_the_payload_would_exceed_the_limit() {
        let outputs = vec![vec![0u8; 1000]];
        let row_size = SerializedTaskOutputs::serialize_with_size_hint(&outputs)
            .expect("serialization should succeed")
            .to_raw()
            .len() as u64;
        let updates: Vec<_> = (0..5)
            .map(|id| JobStateUpdate {
                outputs: Some(outputs.clone()),
                ..JobStateUpdate::new(JobId::from(id), JobState::Running, JobState::Succeeded)
            })
            .collect();
        assert_eq!(
            plan_shape(updates.clone(), 2 * row_size),
            vec![
                (JobState::Running, JobState::Succeeded, 2),
                (JobState::Running, JobState::Succeeded, 2),
                (JobState::Running, JobState::Succeeded, 1),
            ]
        );
        assert_eq!(
            plan_shape(updates, row_size / 2),
            vec![(JobState::Running, JobState::Succeeded, 1); 5],
            "every statement should carry at least one row"
        );
    }

    #[test]
    fn outputs_over_the_limit_name_their_job() {
        let job_id = JobId::from(3);
        let outputs = vec![vec![1u8, 2, 3]];
        let size = SerializedTaskOutputs::serialize_with_size_hint(&outputs)
            .expect("serialization should succeed")
            .to_raw()
            .len() as u64;
        let updates = || {
            vec![
                JobStateUpdate::new(JobId::from(2), JobState::Running, JobState::Succeeded),
                JobStateUpdate {
                    outputs: Some(outputs.clone()),
                    ..JobStateUpdate::new(job_id, JobState::Running, JobState::Succeeded)
                },
            ]
        };
        let result = UpdateStatement::group(updates(), u64::MAX, size - 1);
        assert!(
            matches!(
                result,
                Err(DbError::OutputsTooLarge { job_id: culprit, size: culprit_size, limit })
                    if culprit == job_id && culprit_size == size && limit == size - 1
            ),
            "unexpected result {result:?}"
        );
        assert!(UpdateStatement::group(updates(), u64::MAX, size).is_ok());
    }

    #[test]
    fn child_inserts_are_cut_by_rows_and_by_payload() {
        let blob = [0u8; 100];
        let chunk_sizes = |num_inserts: usize, max_payload: u64| {
            let inserts: Vec<ChildJobInsert<'_>> = (0..num_inserts)
                .map(|_| ChildJobInsert {
                    parent_id: JobId::from(1),
                    root_id: JobId::from(1),
                    resource_group_id: ResourceGroupId::from(1),
                    compressed_task_graph: &blob[..60],
                    compressed_job_inputs: &blob[60..],
                })
                .collect();
            chunk_child_inserts(&inserts, max_payload)
                .iter()
                .map(|chunk| chunk.len())
                .collect::<Vec<_>>()
        };

        assert_eq!(
            chunk_sizes(1_200, u64::MAX),
            vec![IN_LIST_MAX_IDS, IN_LIST_MAX_IDS, 200]
        );
        assert_eq!(chunk_sizes(5, 250), vec![2, 2, 1]);
        assert_eq!(
            chunk_sizes(3, 50),
            vec![1, 1, 1],
            "every chunk should carry at least one row"
        );
        assert_eq!(chunk_sizes(0, 50), Vec::<usize>::new());
    }
}
