use spider_core::job::JobState;
use spider_core::types::id::JobId;
use spider_core::types::io::TaskOutput;
use spider_storage::DatabaseConfig;
use spider_storage::db::ChildJobInsert;
use spider_storage::db::DbError;
use spider_storage::db::ExternalJobOrchestration;
use spider_storage::db::InternalJobOrchestration;
use spider_storage::db::JobStateUpdate;
use spider_storage::db::MariaDbStorageConnector;
use spider_storage::db::SessionManagement;
use spider_storage::db::TreeTransition;
use sqlx::MySqlPool;

use super::mariadb_infra::ScratchDatabase;
use super::mariadb_infra::create_raw_pool;
use super::mariadb_infra::create_test_resource_group;
use super::task_graph_builder::build_flat_task_graph;
use super::task_graph_builder::create_validated_submission;

/// A scratch database, a connector and a raw pool on it, and its configuration.
///
/// The tests leave non-terminal rows behind, which would be recovered by every runtime that
/// connects to a shared database, so each test uses its own database.
struct Fixture {
    scratch: ScratchDatabase,
    config: DatabaseConfig,
    storage: MariaDbStorageConnector,
    pool: MySqlPool,
}

impl Fixture {
    /// # Panics
    ///
    /// Panics if the connection fails.
    async fn create() -> Self {
        let scratch = ScratchDatabase::create().await;
        let config = scratch.config();
        let storage = MariaDbStorageConnector::connect(&config)
            .await
            .expect("connect should succeed");
        let pool = create_raw_pool(&config).await;
        Self {
            scratch,
            config,
            storage,
            pool,
        }
    }

    /// Closes the connections and drops the scratch database.
    async fn finish(self) {
        self.pool.close().await;
        drop(self.storage);
        self.scratch.drop_database().await;
    }

    /// Registers `num_jobs` one-task jobs.
    ///
    /// # Returns
    ///
    /// The IDs of the registered jobs, in `Ready`.
    ///
    /// # Panics
    ///
    /// Panics if a registration fails.
    async fn register_jobs(&self, num_jobs: usize) -> Vec<JobId> {
        let rg_id = create_test_resource_group(&self.storage).await;
        let (graph, task_graph_input) = build_flat_task_graph(1, 16, false, false);
        let job_submission = create_validated_submission(graph, task_graph_input);
        let mut job_ids = Vec::with_capacity(num_jobs);
        for _ in 0..num_jobs {
            job_ids.push(
                self.storage
                    .register(rg_id, &job_submission)
                    .await
                    .expect("register should succeed"),
            );
        }
        job_ids
    }

    /// # Returns
    ///
    /// The number of `job_ids` in `state`.
    ///
    /// # Panics
    ///
    /// Panics if a query fails.
    async fn count_in_state(&self, job_ids: &[JobId], state: JobState) -> usize {
        let mut count = 0;
        for job_id in job_ids {
            if self
                .storage
                .get_state(*job_id)
                .await
                .expect("get_state should succeed")
                == state
            {
                count += 1;
            }
        }
        count
    }

    /// # Returns
    ///
    /// Whether `job_id`'s row has an end timestamp.
    ///
    /// # Panics
    ///
    /// Panics if the query fails.
    async fn has_ended_at(&self, job_id: JobId) -> bool {
        sqlx::query_scalar::<_, bool>("SELECT `ended_at` IS NOT NULL FROM `jobs` WHERE `id` = ?;")
            .bind(job_id)
            .fetch_one(&self.pool)
            .await
            .expect("the end timestamp query should succeed")
    }
}

const fn transition(updates: Vec<JobStateUpdate>) -> TreeTransition<'static> {
    TreeTransition {
        updates,
        child_inserts: Vec::new(),
    }
}

fn updates(job_ids: &[JobId], from: JobState, to: JobState) -> Vec<JobStateUpdate> {
    job_ids
        .iter()
        .map(|job_id| JobStateUpdate::new(*job_id, from, to))
        .collect()
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn guard_miss_is_corruption_and_rolls_back_every_earlier_statement() -> anyhow::Result<()> {
    let fixture = Fixture::create().await;
    let job_ids = fixture.register_jobs(3).await;

    let mut tree_updates = updates(&job_ids[..2], JobState::Ready, JobState::Running);
    tree_updates.push(JobStateUpdate::new(
        job_ids[2],
        JobState::Running,
        JobState::Cancelled,
    ));
    let result = fixture
        .storage
        .apply_tree_transition(transition(tree_updates))
        .await;

    assert!(
        matches!(result, Err(DbError::CorruptedDbState(_))),
        "a guard miss should be corruption, got {result:?}"
    );
    assert_eq!(fixture.count_in_state(&job_ids, JobState::Ready).await, 3);
    fixture.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn chunks_of_one_group_commit_in_one_transaction() -> anyhow::Result<()> {
    let fixture = Fixture::create().await;
    let job_ids = fixture.register_jobs(1_201).await;
    let (group, culprit) = job_ids.split_at(1_200);

    let mut tree_updates = updates(group, JobState::Ready, JobState::Running);
    tree_updates.push(JobStateUpdate::new(
        culprit[0],
        JobState::Running,
        JobState::Cancelled,
    ));
    let result = fixture
        .storage
        .apply_tree_transition(transition(tree_updates))
        .await;
    assert!(
        matches!(result, Err(DbError::CorruptedDbState(_))),
        "the guard miss after three chunks should be corruption, got {result:?}"
    );
    assert_eq!(fixture.count_in_state(group, JobState::Ready).await, 1_200);

    let inserted = fixture
        .storage
        .apply_tree_transition(transition(updates(
            group,
            JobState::Ready,
            JobState::Running,
        )))
        .await?;
    assert_eq!(inserted, Vec::new());
    assert_eq!(
        fixture.count_in_state(group, JobState::Running).await,
        1_200
    );
    fixture.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn outputs_larger_than_one_statement_payload_are_all_written() -> anyhow::Result<()> {
    const OUTPUT_SIZE: usize = 1536 * 1024;

    let fixture = Fixture::create().await;
    let job_ids = fixture.register_jobs(3).await;
    fixture
        .storage
        .apply_tree_transition(transition(updates(
            &job_ids,
            JobState::Ready,
            JobState::Running,
        )))
        .await?;

    let outputs_of = |job_index: usize| -> Vec<TaskOutput> {
        vec![vec![
            u8::try_from(job_index)
                .expect("job index should fit in a byte");
            OUTPUT_SIZE
        ]]
    };
    let tree_updates = job_ids
        .iter()
        .enumerate()
        .map(|(job_index, job_id)| JobStateUpdate {
            outputs: Some(outputs_of(job_index)),
            ..JobStateUpdate::new(*job_id, JobState::Running, JobState::Succeeded)
        })
        .collect();
    fixture
        .storage
        .apply_tree_transition(transition(tree_updates))
        .await?;

    for (job_index, job_id) in job_ids.iter().enumerate() {
        assert_eq!(
            fixture.storage.get_outputs(*job_id).await?,
            outputs_of(job_index)
        );
    }
    fixture.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn a_fenced_off_connector_writes_nothing() -> anyhow::Result<()> {
    let fixture = Fixture::create().await;
    let job_ids = fixture.register_jobs(1).await;
    let newer = MariaDbStorageConnector::connect(&fixture.config).await?;

    let result = fixture
        .storage
        .apply_tree_transition(transition(updates(
            &job_ids,
            JobState::Ready,
            JobState::Running,
        )))
        .await;

    assert!(
        matches!(
            result,
            Err(DbError::Fenced { mine, current })
                if mine == fixture.storage.session_id() && current == newer.session_id()
        ),
        "expected Fenced, got {result:?}"
    );
    assert_eq!(fixture.count_in_state(&job_ids, JobState::Ready).await, 1);
    fixture.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn only_terminal_states_set_the_end_timestamp() -> anyhow::Result<()> {
    let fixture = Fixture::create().await;
    let job_ids = fixture.register_jobs(2).await;
    let (committing, failing) = (job_ids[0], job_ids[1]);

    fixture
        .storage
        .apply_tree_transition(transition(updates(
            &job_ids,
            JobState::Ready,
            JobState::Running,
        )))
        .await?;
    fixture
        .storage
        .apply_tree_transition(transition(vec![
            JobStateUpdate {
                outputs: Some(vec![vec![7]]),
                ..JobStateUpdate::new(committing, JobState::Running, JobState::CommitReady)
            },
            JobStateUpdate {
                error_message: Some("failed on purpose".to_owned()),
                ..JobStateUpdate::new(failing, JobState::Running, JobState::Failed)
            },
        ]))
        .await?;
    assert!(!fixture.has_ended_at(committing).await);
    assert!(fixture.has_ended_at(failing).await);
    assert_eq!(
        fixture.storage.get_error(failing).await?,
        "failed on purpose"
    );

    fixture
        .storage
        .apply_tree_transition(transition(updates(
            &[committing],
            JobState::CommitReady,
            JobState::Succeeded,
        )))
        .await?;
    assert!(fixture.has_ended_at(committing).await);
    assert_eq!(
        fixture.storage.get_outputs(committing).await?,
        vec![vec![7u8]]
    );
    fixture.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn child_inserts_return_their_ids_in_order_across_chunks() -> anyhow::Result<()> {
    const NUM_CHILDREN: usize = 1_200;

    let fixture = Fixture::create().await;
    let parent_id = fixture.register_jobs(1).await[0];
    fixture
        .storage
        .apply_tree_transition(transition(updates(
            &[parent_id],
            JobState::Ready,
            JobState::Running,
        )))
        .await?;
    let resource_group_id: spider_core::types::id::ResourceGroupId =
        sqlx::query_scalar("SELECT `resource_group_id` FROM `jobs` WHERE `id` = ?;")
            .bind(parent_id)
            .fetch_one(&fixture.pool)
            .await?;
    let graphs: Vec<Vec<u8>> = (0..NUM_CHILDREN)
        .map(|index| index.to_le_bytes().to_vec())
        .collect();

    let child_ids = fixture
        .storage
        .apply_tree_transition(TreeTransition {
            updates: updates(
                &[parent_id],
                JobState::Running,
                JobState::BlockingOnChildren,
            ),
            child_inserts: graphs
                .iter()
                .map(|graph| ChildJobInsert {
                    parent_id,
                    root_id: parent_id,
                    resource_group_id,
                    compressed_task_graph: graph,
                    compressed_job_inputs: &[],
                })
                .collect(),
        })
        .await?;

    let rows: Vec<(JobId, Vec<u8>)> = sqlx::query_as(
        "SELECT `id`, `compressed_serialized_task_graph` FROM `jobs` WHERE `parent_id` = ? AND \
         `root_id` = ? AND `state` = 'Running' ORDER BY `id`;",
    )
    .bind(parent_id)
    .bind(parent_id)
    .fetch_all(&fixture.pool)
    .await?;
    assert_eq!(rows.iter().map(|row| row.0).collect::<Vec<_>>(), child_ids);
    assert_eq!(
        rows.into_iter().map(|row| row.1).collect::<Vec<_>>(),
        graphs
    );
    assert_eq!(
        fixture
            .count_in_state(&[parent_id], JobState::BlockingOnChildren)
            .await,
        1
    );
    fixture.finish().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn only_a_child_insert_over_the_packet_size_is_a_payload_rejection() -> anyhow::Result<()> {
    let fixture = Fixture::create().await;
    let job_ids = fixture.register_jobs(2).await;
    let (parent_id, other_id) = (job_ids[0], job_ids[1]);
    fixture
        .storage
        .apply_tree_transition(transition(updates(
            &job_ids,
            JobState::Ready,
            JobState::Running,
        )))
        .await?;
    let resource_group_id: spider_core::types::id::ResourceGroupId =
        sqlx::query_scalar("SELECT `resource_group_id` FROM `jobs` WHERE `id` = ?;")
            .bind(parent_id)
            .fetch_one(&fixture.pool)
            .await?;
    let max_allowed_packet: u64 = sqlx::query_scalar("SELECT @@max_allowed_packet;")
        .fetch_one(&fixture.pool)
        .await?;
    let oversized = vec![7u8; usize::try_from(max_allowed_packet)? + 1];

    let result = fixture
        .storage
        .apply_tree_transition(TreeTransition {
            updates: updates(
                &[parent_id],
                JobState::Running,
                JobState::BlockingOnChildren,
            ),
            child_inserts: vec![ChildJobInsert {
                parent_id,
                root_id: parent_id,
                resource_group_id,
                compressed_task_graph: &oversized,
                compressed_job_inputs: &[],
            }],
        })
        .await;
    assert!(
        matches!(result, Err(DbError::PayloadRejected(_))),
        "an oversized child insert should be a payload rejection, got {result:?}"
    );
    assert_eq!(fixture.count_in_state(&job_ids, JobState::Running).await, 2);
    let num_children: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM `jobs` WHERE `parent_id` = ?;")
            .bind(parent_id)
            .fetch_one(&fixture.pool)
            .await?;
    assert_eq!(num_children, 0);

    let mut xorshift_state = 0x9e37_79b9_7f4a_7c15_u64;
    let incompressible: Vec<u8> = (0..oversized.len())
        .map(|_| {
            xorshift_state ^= xorshift_state << 13;
            xorshift_state ^= xorshift_state >> 7;
            xorshift_state ^= xorshift_state << 17;
            xorshift_state.to_le_bytes()[0]
        })
        .collect();
    let result = fixture
        .storage
        .apply_tree_transition(transition(vec![JobStateUpdate {
            outputs: Some(vec![incompressible]),
            ..JobStateUpdate::new(other_id, JobState::Running, JobState::Succeeded)
        }]))
        .await;
    assert!(
        matches!(
            result,
            Err(DbError::OutputsTooLarge { job_id, .. }) if job_id == other_id
        ),
        "oversized outputs should be refused before they are sent, got {result:?}"
    );
    assert_eq!(fixture.count_in_state(&job_ids, JobState::Running).await, 2);
    fixture.finish().await;
    Ok(())
}
