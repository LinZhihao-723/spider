use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::ExitStatus;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

use spider_proto_rust::storage::JobOrchestrationServiceClient;
use spider_proto_rust::storage::RegisterJobRequest;
use spider_storage::DatabaseConfig;
use sqlx::Acquire;

use super::mariadb_infra::ScratchDatabase;
use super::mariadb_infra::create_raw_pool;
use super::task_graph_builder::build_flat_task_graph;
use super::task_graph_builder::compress_job_inputs;
use super::task_graph_builder::compress_task_graph;

/// A storage server process that is killed when dropped.
struct StorageServerProcess {
    child: Child,
    port: u16,
    config_path: PathBuf,
}

impl StorageServerProcess {
    /// Starts the storage server binary on a free port with `db` as its database configuration,
    /// and waits until it accepts gRPC connections.
    ///
    /// The binary does not read `db.fence_id`, so it always uses the fence ID 0.
    ///
    /// # Returns
    ///
    /// The running process.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// * The configuration file cannot be written.
    /// * The process cannot be started.
    /// * The server does not accept connections within 30 seconds.
    async fn start(db: &DatabaseConfig) -> Self {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .expect("a free port should be available")
            .port();
        let config_path = std::env::temp_dir().join(format!(
            "spider-storage-{:016x}.yaml",
            rand::random::<u64>()
        ));
        std::fs::write(
            &config_path,
            format!(
                "host: 127.0.0.1\nport: {port}\nruntime:\n  db:\n    host: {host}\n    port: \
                 {db_port}\n    name: {name}\n    max_connections: {max_connections}\n",
                host = db.host,
                db_port = db.port,
                name = db.name,
                max_connections = db.max_connections,
            ),
        )
        .expect("config file should be written");
        let child = Command::new(env!("CARGO_BIN_EXE_spider-storage"))
            .arg("--config")
            .arg(&config_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("storage server should start");
        let process = Self {
            child,
            port,
            config_path,
        };

        let deadline = Instant::now() + Duration::from_secs(30);
        while process.client().await.is_err() {
            assert!(
                Instant::now() < deadline,
                "storage server should accept connections"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        process
    }

    /// # Returns
    ///
    /// A job orchestration client connected to the server on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`JobOrchestrationServiceClient::connect`]'s return values on failure.
    async fn client(
        &self,
    ) -> Result<JobOrchestrationServiceClient<tonic::transport::Channel>, tonic::transport::Error>
    {
        JobOrchestrationServiceClient::connect(format!("http://127.0.0.1:{}", self.port)).await
    }

    /// Sends `SIGTERM` to the process.
    ///
    /// # Panics
    ///
    /// Panics if the signal cannot be sent.
    fn terminate(&self) {
        let status = Command::new("kill")
            .arg("-TERM")
            .arg(self.child.id().to_string())
            .status()
            .expect("`kill` should run");
        assert!(status.success(), "`kill` failed with {status}");
    }

    /// Waits for the process to exit.
    ///
    /// # Returns
    ///
    /// The exit status, or `None` if the process is still running after `timeout`.
    ///
    /// # Panics
    ///
    /// Panics if the process status cannot be read.
    async fn wait_for_exit(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().expect("try_wait should succeed") {
                return Some(status);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

impl Drop for StorageServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.config_path);
    }
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn fenced_storage_server_exits_with_failure() -> anyhow::Result<()> {
    const EXIT_TIMEOUT: Duration = Duration::from_secs(11);

    let scratch = ScratchDatabase::create().await;
    let db = scratch.config();
    let mut fenced = StorageServerProcess::start(&db).await;
    let newer = StorageServerProcess::start(&db).await;

    let (graph, task_graph_input) = build_flat_task_graph(1, 16, false, false);
    let register_result = fenced
        .client()
        .await?
        .register_job(RegisterJobRequest {
            resource_group_id: 1,
            compressed_serialized_task_graph: compress_task_graph(&graph)?,
            compressed_serialized_task_graph_input: compress_job_inputs(&task_graph_input)?,
        })
        .await;
    let exit_status = fenced.wait_for_exit(EXIT_TIMEOUT).await;

    assert_eq!(
        register_result.err().map(|status| status.code()),
        Some(tonic::Code::Internal)
    );
    let exit_status = exit_status.expect("the fenced server should exit");
    assert!(
        !exit_status.success(),
        "unexpected exit status {exit_status}"
    );
    drop(fenced);
    drop(newer);
    scratch.drop_database().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MariaDB"]
async fn storage_server_exits_with_failure_when_a_request_outlives_the_shutdown_deadline()
-> anyhow::Result<()> {
    const MIN_EXIT_DELAY: Duration = Duration::from_secs(9);
    const EXIT_TIMEOUT: Duration = Duration::from_secs(13);

    let scratch = ScratchDatabase::create().await;
    let db = DatabaseConfig {
        fence_id: 0,
        ..scratch.config()
    };
    let mut server = StorageServerProcess::start(&db).await;
    let raw_pool = create_raw_pool(&db).await;
    let mut raw_connection = raw_pool.acquire().await?;
    let mut raw_tx = raw_connection.begin().await?;
    sqlx::query("SELECT `session_id` FROM `storage_fence` WHERE `id` = ? FOR UPDATE;")
        .bind(db.fence_id)
        .execute(&mut *raw_tx)
        .await?;

    let mut client = server.client().await?;
    let (graph, task_graph_input) = build_flat_task_graph(1, 16, false, false);
    let request = RegisterJobRequest {
        resource_group_id: 1,
        compressed_serialized_task_graph: compress_task_graph(&graph)?,
        compressed_serialized_task_graph_input: compress_job_inputs(&task_graph_input)?,
    };
    let mut blocked_request = tokio::spawn(async move { client.register_job(request).await });
    assert!(
        tokio::time::timeout(Duration::from_millis(500), &mut blocked_request)
            .await
            .is_err(),
        "the request should wait for the fence row lock"
    );

    let terminated_at = Instant::now();
    server.terminate();
    let exit_status = server.wait_for_exit(EXIT_TIMEOUT).await;
    let exit_delay = terminated_at.elapsed();

    let exit_status = exit_status.expect("the server should exit after the shutdown deadline");
    assert!(
        !exit_status.success(),
        "unexpected exit status {exit_status}"
    );
    assert!(
        exit_delay >= MIN_EXIT_DELAY,
        "the server should wait for the in-flight request until the deadline, exited after \
         {exit_delay:?}"
    );
    blocked_request.abort();
    raw_tx.rollback().await?;
    drop(raw_connection);
    raw_pool.close().await;
    drop(server);
    scratch.drop_database().await;
    Ok(())
}
