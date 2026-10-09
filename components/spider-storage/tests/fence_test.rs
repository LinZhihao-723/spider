use std::time::Duration;

use secrecy::SecretString;
use spider_core::job::JobState;
use spider_core::types::id::JobId;
use spider_storage::DatabaseConfig;
use spider_storage::DatabaseCredentials;
use spider_storage::db::DbError;
use spider_storage::db::ExternalJobOrchestration;
use spider_storage::db::InternalJobOrchestration;
use spider_storage::db::JobStateUpdate;
use spider_storage::db::MariaDbStorageConnector;
use spider_storage::db::SessionManagement;
use spider_storage::db::TreeTransition;
use sqlx::Acquire;
use sqlx::MySqlPool;
use sqlx::mysql::MySqlDatabaseError;

use super::mariadb_infra::ScratchDatabase;
use super::mariadb_infra::create_mariadb_config;
use super::mariadb_infra::create_raw_pool;
use super::mariadb_infra::create_test_resource_group;
use super::mariadb_infra::mariadb_root_password;
use super::task_graph_builder::build_flat_task_graph;
use super::task_graph_builder::create_validated_submission;

const MYSQL_ER_LOCK_WAIT_TIMEOUT: u16 = 1205;

const SELECT_FENCE_FOR_SHARE_QUERY: &str =
    "SELECT `session_id` FROM `storage_fence` WHERE `id` = ? LOCK IN SHARE MODE;";

/// Starts `job_id` through `storage` in one tree transition.
///
/// # Errors
///
/// Returns an error if:
///
/// * Forwards [`InternalJobOrchestration::apply_tree_transition`]'s return values on failure.
async fn start(storage: &MariaDbStorageConnector, job_id: JobId) -> Result<(), DbError> {
    storage
        .apply_tree_transition(TreeTransition {
            updates: vec![JobStateUpdate::new(
                job_id,
                JobState::Ready,
                JobState::Running,
            )],
            child_inserts: Vec::new(),
        })
        .await
        .map(|_| ())
}

/// # Returns
///
/// A [`MariaDbStorageConnector`] connected with `config`.
///
/// # Panics
///
/// Panics if the connection fails.
async fn connect(config: &DatabaseConfig) -> MariaDbStorageConnector {
    MariaDbStorageConnector::connect(config)
        .await
        .expect("connect should succeed")
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn newer_connector_fences_off_older_connector_with_same_fence_id() -> anyhow::Result<()> {
    let config = create_mariadb_config();
    let older = connect(&config).await;
    let newer = connect(&config).await;
    let rg_id = create_test_resource_group(&newer).await;
    let (graph, task_graph_input) = build_flat_task_graph(1, 16, false, false);
    let job_submission = create_validated_submission(graph, task_graph_input);

    let register_result = older.register(rg_id, &job_submission).await;
    let job_id = newer.register(rg_id, &job_submission).await?;
    let results = [
        register_result.map(|_| ()),
        start(&older, job_id).await,
        older.delete_expired_terminated_jobs(3600).await.map(|_| ()),
    ];

    for result in results {
        assert!(
            matches!(
                result,
                Err(DbError::Fenced { mine, current })
                    if mine == older.session_id() && current == newer.session_id()
            ),
            "expected Fenced, got {result:?}"
        );
    }
    start(&newer, job_id).await?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn connectors_with_different_fence_ids_are_independent() -> anyhow::Result<()> {
    let first = connect(&create_mariadb_config()).await;
    let second = connect(&create_mariadb_config()).await;
    let rg_id = create_test_resource_group(&first).await;
    let (graph, task_graph_input) = build_flat_task_graph(1, 16, false, false);
    let job_submission = create_validated_submission(graph, task_graph_input);

    let first_job_id = first.register(rg_id, &job_submission).await?;
    let second_job_id = second.register(rg_id, &job_submission).await?;
    start(&first, first_job_id).await?;
    start(&second, second_job_id).await?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn connect_waits_for_transactions_that_passed_the_fence() -> anyhow::Result<()> {
    let config = create_mariadb_config();
    let older = connect(&config).await;
    let rg_id = create_test_resource_group(&older).await;
    let (graph, task_graph_input) = build_flat_task_graph(1, 16, false, false);
    let job_id = older
        .register(rg_id, &create_validated_submission(graph, task_graph_input))
        .await?;
    let raw_pool = create_raw_pool(&config).await;
    let mut raw_connection = raw_pool.acquire().await?;
    let mut raw_tx = raw_connection.begin().await?;
    sqlx::query(SELECT_FENCE_FOR_SHARE_QUERY)
        .bind(config.fence_id)
        .execute(&mut *raw_tx)
        .await?;

    let connect_config = config.clone();
    let newer = tokio::spawn(async move { connect(&connect_config).await });
    tokio::time::timeout(
        Duration::from_secs(5),
        wait_for_queued_fence_bump(&raw_pool, config.fence_id),
    )
    .await
    .expect("connect should queue its fence bump behind the open transaction");
    assert!(
        !newer.is_finished(),
        "connect should wait for the open transaction"
    );
    let starting_older = older.clone();
    let older_start = tokio::spawn(async move { start(&starting_older, job_id).await });

    raw_tx.commit().await?;
    let newer = tokio::time::timeout(Duration::from_secs(5), newer).await??;
    let start_result = tokio::time::timeout(Duration::from_secs(5), older_start).await??;
    assert!(
        matches!(
            start_result,
            Err(DbError::Fenced { mine, current })
                if mine == older.session_id() && current == newer.session_id()
        ),
        "a transaction that starts while the fence bump is queued should be fenced, got \
         {start_result:?}"
    );
    Ok(())
}

/// Waits until an exclusive lock request on the fence row `fence_id` is queued, which a shared
/// lock request without waiting then conflicts with.
///
/// # Panics
///
/// Panics if a probe fails for any reason other than the lock conflict.
async fn wait_for_queued_fence_bump(pool: &MySqlPool, fence_id: u64) {
    loop {
        let mut probe_tx = pool.begin().await.expect("probe should begin");
        let probe_result = sqlx::query(
            "SELECT `session_id` FROM `storage_fence` WHERE `id` = ? LOCK IN SHARE MODE NOWAIT;",
        )
        .bind(fence_id)
        .execute(&mut *probe_tx)
        .await;
        probe_tx.rollback().await.expect("probe should roll back");
        match probe_result {
            Ok(_) => tokio::time::sleep(Duration::from_millis(20)).await,
            Err(sqlx::Error::Database(e))
                if e.try_downcast_ref::<MySqlDatabaseError>()
                    .is_some_and(|e| e.number() == MYSQL_ER_LOCK_WAIT_TIMEOUT) =>
            {
                return;
            }
            Err(e) => panic!("unexpected probe failure: {e}"),
        }
    }
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn connect_kills_an_orphaned_fence_holder_when_privileged() -> anyhow::Result<()> {
    let scratch = ScratchDatabase::create().await;
    let user_config = scratch.config();
    let root_config = DatabaseConfig {
        credentials: DatabaseCredentials {
            username: "root".to_owned(),
            password: SecretString::from(mariadb_root_password()),
        },
        ..user_config.clone()
    };
    let _older = connect(&user_config).await;
    let raw_pool = create_raw_pool(&user_config).await;
    let mut orphan_connection = raw_pool.acquire().await?;
    let mut orphan_tx = orphan_connection.begin().await?;
    sqlx::query(SELECT_FENCE_FOR_SHARE_QUERY)
        .bind(user_config.fence_id)
        .execute(&mut *orphan_tx)
        .await?;
    let unrelated_pool = create_raw_pool(&create_mariadb_config()).await;
    let mut unrelated_connection = unrelated_pool.acquire().await?;
    let mut unrelated_tx = unrelated_connection.begin().await?;
    sqlx::query("SELECT COUNT(*) FROM `sessions` LOCK IN SHARE MODE;")
        .execute(&mut *unrelated_tx)
        .await?;

    tokio::time::timeout(Duration::from_secs(4), connect(&root_config)).await?;

    let commit_result = orphan_tx.commit().await;
    assert!(
        commit_result.is_err(),
        "the orphaned transaction's connection should have been killed"
    );
    unrelated_tx.commit().await?;
    drop(orphan_connection);
    raw_pool.close().await;
    drop(unrelated_connection);
    unrelated_pool.close().await;
    scratch.drop_database().await;
    Ok(())
}
