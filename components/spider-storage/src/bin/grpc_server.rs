//! Command-line entrypoint for the storage gRPC server.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use spider_proto_rust::storage::ExecutionManagerLivenessServiceServer;
use spider_proto_rust::storage::InboundQueueServiceServer;
use spider_proto_rust::storage::JobOrchestrationServiceServer;
use spider_proto_rust::storage::ResourceGroupManagementServiceServer;
use spider_proto_rust::storage::SchedulerRegistrationServiceServer;
use spider_proto_rust::storage::SessionManagementServiceServer;
use spider_proto_rust::storage::TaskInstanceManagementServiceServer;
use spider_storage::ServerConfig;
use spider_storage::db::MariaDbStorageConnector;
use spider_storage::grpc::GrpcServiceState;
use spider_storage::limits::MAX_TASK_INSTANCE_MESSAGE_SIZE;
use spider_storage::state::Runtime;
use spider_storage::state::ShutdownHandle;
use spider_storage::state::runtime::create_runtime;
use spider_storage::task_instance_pool::TaskInstancePoolHandle;
use spider_utils::config::YamlConfig;
use spider_utils::logging::set_up_logging;
use tokio::net::TcpListener;
use tokio::select;
use tokio::signal::unix::SignalKind;
use tokio::signal::unix::signal;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;

/// How long the gRPC server may take to shut down after the service is cancelled before the
/// process exits regardless.
const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(10);

/// Command-line arguments for the storage gRPC server.
#[derive(Debug, Parser)]
#[command(about = "Run the Spider storage gRPC server.")]
struct Cli {
    /// Path to the YAML server configuration file.
    #[arg(short, long, value_name = "PATH")]
    config: PathBuf,
}

fn main() -> ExitCode {
    let _log_guard = set_up_logging();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            tracing::error!(error = % e, "Failed to build the async runtime.");
            return ExitCode::FAILURE;
        }
    };
    let exit_code = runtime.block_on(run());
    runtime.shutdown_background();
    exit_code
}

/// Runs the storage gRPC server until it shuts down.
///
/// # Returns
///
/// [`ExitCode::FAILURE`] if the server failed to start, shut down on a fatal error, missed
/// [`SHUTDOWN_DEADLINE`], or failed to stop its runtime. [`ExitCode::SUCCESS`] otherwise.
async fn run() -> ExitCode {
    let cli = Cli::parse();
    let Ok(server_config) = ServerConfig::from_yaml_file(&cli.config).inspect_err(|e| {
        tracing::error!(error = % e, "Failed to load server configuration.");
    }) else {
        return ExitCode::FAILURE;
    };
    let listen_addr = SocketAddr::new(server_config.host, server_config.port);
    let Ok((runtime, cancellation_token)) = create_runtime(&server_config.runtime)
        .await
        .inspect_err(|e| {
            tracing::error!(error = % e, "Failed to create runtime.");
        })
    else {
        return ExitCode::FAILURE;
    };
    let service_state = runtime.get_service_state();
    let grpc_service = GrpcServiceState::new(service_state.clone());
    tracing::info!(listen_addr = % listen_addr, "Starting storage gRPC server.");

    let Ok(listener) = TcpListener::bind(listen_addr).await.inspect_err(
        |error| tracing::error!(error = % error, "Failed to bind storage listen address."),
    ) else {
        return ExitCode::FAILURE;
    };
    let incoming = TcpIncoming::from(listener).with_nodelay(Some(true));
    let Ok(mut sigterm) = signal(SignalKind::terminate()).inspect_err(
        |error| tracing::error!(error = % error, "Failed to register SIGTERM handler."),
    ) else {
        return ExitCode::FAILURE;
    };
    let server_cancellation_token = cancellation_token.clone();
    let server = tokio::spawn(
        Server::builder()
            .add_service(JobOrchestrationServiceServer::new(grpc_service.clone()))
            .add_service(
                TaskInstanceManagementServiceServer::new(grpc_service.clone())
                    .max_decoding_message_size(MAX_TASK_INSTANCE_MESSAGE_SIZE),
            )
            .add_service(InboundQueueServiceServer::new(grpc_service.clone()))
            .add_service(ResourceGroupManagementServiceServer::new(
                grpc_service.clone(),
            ))
            .add_service(ExecutionManagerLivenessServiceServer::new(
                grpc_service.clone(),
            ))
            .add_service(SchedulerRegistrationServiceServer::new(
                grpc_service.clone(),
            ))
            .add_service(SessionManagementServiceServer::new(grpc_service))
            .serve_with_incoming_shutdown(incoming, async move {
                select! {
                    () = server_cancellation_token.cancelled() => {
                        tracing::info!("Shutting down storage gRPC server.");
                    }
                    result = tokio::signal::ctrl_c() => {
                        if let Err(error) = result {
                            tracing::error!(error = % error, "Failed to listen for Ctrl-C.");
                        }
                        server_cancellation_token.cancel();
                    }
                    _ = sigterm.recv() => {
                        tracing::info!("Received SIGTERM. Shutting down storage gRPC server.");
                        server_cancellation_token.cancel();
                    }
                }
            }),
    );

    // Initialize the inbound queue by resending all ready tasks from the current cache. This must
    // be done after the server becomes serviceable. Otherwise, the inbound queue could fill up and
    // cause a deadlock.
    //
    // This operation may duplicate ready tasks for jobs submitted after the server starts but
    // before the resend starts. This is safe because the system is designed to tolerate
    // duplicate entries in the inbound queue.
    if let Err(e) = service_state.resend_ready_tasks().await {
        tracing::error!(error = % e, "Failed to initialize inbound queue.");
        service_state.shutdown().shut_down_fatally();
    }

    wait_for_shutdown(server, runtime, service_state.shutdown().clone()).await
}

/// Waits for the gRPC server to shut down, then stops the runtime.
///
/// Both waits together are bounded by [`SHUTDOWN_DEADLINE`] once `shutdown`'s token is cancelled.
///
/// # Returns
///
/// [`ExitCode::FAILURE`] if the server or the runtime missed the deadline, the server failed, the
/// runtime failed to stop, or the service shut down on a fatal error. [`ExitCode::SUCCESS`]
/// otherwise.
async fn wait_for_shutdown(
    mut server: JoinHandle<Result<(), tonic::transport::Error>>,
    runtime: Runtime<MariaDbStorageConnector, TaskInstancePoolHandle>,
    shutdown: ShutdownHandle,
) -> ExitCode {
    let deadline = async {
        shutdown.token().cancelled().await;
        tokio::time::sleep(SHUTDOWN_DEADLINE).await;
    };
    tokio::pin!(deadline);

    let server_result = select! {
        result = &mut server => result,
        () = &mut deadline => {
            tracing::error!("The gRPC server did not shut down before the deadline. Exiting.");
            return ExitCode::FAILURE;
        }
    };

    let mut exit_code = ExitCode::SUCCESS;
    match server_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::error!(error = % e, "gRPC server failure.");
            exit_code = ExitCode::FAILURE;
        }
        Err(e) => {
            tracing::error!(error = % e, "gRPC server panicked.");
            exit_code = ExitCode::FAILURE;
        }
    }

    let stop_result = select! {
        result = runtime.stop() => result,
        () = &mut deadline => {
            tracing::error!("The runtime did not stop before the deadline. Exiting.");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = stop_result {
        tracing::error!(error = % e, "gRPC runtime failure.");
        exit_code = ExitCode::FAILURE;
    }
    if shutdown.is_fatal() {
        tracing::error!("The storage service shut down on a fatal error.");
        exit_code = ExitCode::FAILURE;
    }
    exit_code
}
