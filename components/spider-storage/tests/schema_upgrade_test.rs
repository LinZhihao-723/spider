use std::time::Duration;

use spider_storage::db::MariaDbStorageConnector;
use spider_storage::db::SessionManagement;
use sqlx::Acquire;
use sqlx::MySqlPool;

use super::mariadb_infra::ScratchDatabase;
use super::mariadb_infra::create_raw_pool;

/// The `resource_groups` and `jobs` tables as created before job trees existed.
const PRE_JOB_TREE_SCHEMA: &str = r"
CREATE TABLE `resource_groups` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `external_id` VARCHAR(256) NOT NULL,
  `password` VARBINARY(2048) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `external_resource_group_id` (`external_id`)
);
CREATE TABLE `jobs` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `resource_group_id` BIGINT UNSIGNED NOT NULL,
  `state` ENUM('Ready','Running','CommitReady','CleanupReady','Succeeded','Failed','Cancelled')
    NOT NULL DEFAULT 'Ready',
  `compressed_serialized_task_graph` LONGBLOB NOT NULL,
  `compressed_serialized_job_inputs` LONGBLOB NOT NULL,
  `serialized_job_outputs` LONGBLOB,
  `error_message` LONGTEXT,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `ended_at` TIMESTAMP,
  `max_num_retries` INT UNSIGNED NOT NULL DEFAULT 0,
  `num_retries` INT UNSIGNED NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  INDEX `job_state` (`state`),
  CONSTRAINT `job_resource_group` FOREIGN KEY (`resource_group_id`)
    REFERENCES `resource_groups` (`id`)
    ON UPDATE RESTRICT ON DELETE RESTRICT
);
INSERT INTO `resource_groups` (`external_id`, `password`) VALUES ('pre-job-tree', 'password');
INSERT INTO `jobs` (`resource_group_id`, `state`, `compressed_serialized_task_graph`,
  `compressed_serialized_job_inputs`) VALUES (LAST_INSERT_ID(), 'Running', '', '');
";

/// The upgrade-relevant parts of the `jobs` table's schema and of the fence row.
#[derive(Debug, PartialEq, Eq)]
struct ObservedSchema {
    columns: Vec<String>,
    state_type: String,
    has_parent_index: bool,
    fence_session_id: Option<u64>,
    pre_existing_job_tree_ids: Option<(Option<u64>, Option<u64>)>,
}

impl ObservedSchema {
    /// Reads the schema of the database that `pool` is connected to.
    ///
    /// # Returns
    ///
    /// The observed schema.
    ///
    /// # Panics
    ///
    /// Panics if any of the reads fails.
    async fn read(pool: &MySqlPool, fence_id: u64) -> Self {
        let columns: Vec<String> = sqlx::query_scalar(
            "SELECT `COLUMN_NAME` FROM `information_schema`.`COLUMNS` WHERE `TABLE_SCHEMA` = \
             DATABASE() AND `TABLE_NAME` = 'jobs' AND `COLUMN_NAME` IN ('parent_id', 'root_id') \
             ORDER BY `COLUMN_NAME`;",
        )
        .fetch_all(pool)
        .await
        .expect("column read should succeed");
        let state_type: String = sqlx::query_scalar(
            "SELECT `COLUMN_TYPE` FROM `information_schema`.`COLUMNS` WHERE `TABLE_SCHEMA` = \
             DATABASE() AND `TABLE_NAME` = 'jobs' AND `COLUMN_NAME` = 'state';",
        )
        .fetch_one(pool)
        .await
        .expect("state type read should succeed");
        let parent_index_columns: Vec<String> = sqlx::query_scalar(
            "SELECT `COLUMN_NAME` FROM `information_schema`.`STATISTICS` WHERE `TABLE_SCHEMA` = \
             DATABASE() AND `TABLE_NAME` = 'jobs' AND `INDEX_NAME` = 'job_parent';",
        )
        .fetch_all(pool)
        .await
        .expect("index read should succeed");
        let fence_session_id: Option<u64> =
            sqlx::query_scalar("SELECT `session_id` FROM `storage_fence` WHERE `id` = ?;")
                .bind(fence_id)
                .fetch_optional(pool)
                .await
                .expect("fence read should succeed");
        let pre_existing_job_tree_ids: Option<(Option<u64>, Option<u64>)> = sqlx::query_as(
            "SELECT `parent_id`, `root_id` FROM `jobs` WHERE `compressed_serialized_task_graph` = \
             '';",
        )
        .fetch_optional(pool)
        .await
        .expect("job read should succeed");
        Self {
            columns,
            state_type,
            has_parent_index: parent_index_columns == ["parent_id"],
            fence_session_id,
            pre_existing_job_tree_ids,
        }
    }
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn connect_upgrades_pre_job_tree_schema_once() -> anyhow::Result<()> {
    let scratch = ScratchDatabase::create().await;
    let config = scratch.config();
    let raw_pool = create_raw_pool(&config).await;
    sqlx::raw_sql(PRE_JOB_TREE_SCHEMA)
        .execute(&raw_pool)
        .await?;

    let first = MariaDbStorageConnector::connect(&config).await?;
    let after_first = ObservedSchema::read(&raw_pool, config.fence_id).await;
    let second = MariaDbStorageConnector::connect(&config).await?;
    let after_second = ObservedSchema::read(&raw_pool, config.fence_id).await;

    assert_eq!(first.schema_ddl_statements_at_connect(), 2);
    assert_eq!(second.schema_ddl_statements_at_connect(), 0);
    let expected_schema = ObservedSchema {
        columns: vec!["parent_id".to_owned(), "root_id".to_owned()],
        state_type: "enum('Ready','Running','CommitReady','CleanupReady','Succeeded','Failed','\
                     Cancelled','BlockingOnChildren')"
            .to_owned(),
        has_parent_index: true,
        fence_session_id: Some(first.session_id()),
        pre_existing_job_tree_ids: Some((None, None)),
    };
    assert_eq!(after_first, expected_schema);
    assert_eq!(
        after_second,
        ObservedSchema {
            fence_session_id: Some(second.session_id()),
            ..expected_schema
        }
    );

    raw_pool.close().await;
    scratch.drop_database().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn connect_to_a_new_database_needs_no_upgrade() -> anyhow::Result<()> {
    let scratch = ScratchDatabase::create().await;
    let config = scratch.config();

    let connector = MariaDbStorageConnector::connect(&config).await?;

    assert_eq!(connector.schema_ddl_statements_at_connect(), 0);
    scratch.drop_database().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn upgrade_waits_for_an_open_transaction_without_blocking_the_table() -> anyhow::Result<()> {
    const SELECT_JOBS_QUERY: &str = "SELECT COUNT(*) FROM `jobs`;";

    let scratch = ScratchDatabase::create().await;
    let config = scratch.config();
    let raw_pool = create_raw_pool(&config).await;
    sqlx::raw_sql(PRE_JOB_TREE_SCHEMA)
        .execute(&raw_pool)
        .await?;
    let mut holder_connection = raw_pool.acquire().await?;
    let mut holder_tx = holder_connection.begin().await?;
    sqlx::query(SELECT_JOBS_QUERY)
        .execute(&mut *holder_tx)
        .await?;

    let connect_config = config.clone();
    let mut connector =
        tokio::spawn(async move { MariaDbStorageConnector::connect(&connect_config).await });
    assert!(
        tokio::time::timeout(Duration::from_secs(1), &mut connector)
            .await
            .is_err(),
        "the upgrade should wait for the open transaction"
    );
    tokio::time::timeout(
        Duration::from_secs(7),
        sqlx::query(SELECT_JOBS_QUERY).execute(&raw_pool),
    )
    .await
    .expect("a statement queued behind the upgrade should run once the upgrade's wait times out")?;
    assert!(
        !connector.is_finished(),
        "the upgrade should keep waiting while the transaction is open"
    );

    holder_tx.commit().await?;
    let connector = tokio::time::timeout(Duration::from_secs(7), connector).await???;

    assert_eq!(connector.schema_ddl_statements_at_connect(), 2);
    drop(holder_connection);
    raw_pool.close().await;
    scratch.drop_database().await;
    Ok(())
}
