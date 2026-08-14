//! The tonic service exposing the prototype scheduler to harness workers.
//!
//! The harness drives the prototype through the same gRPC surface the production scheduler
//! presents, so a measurement taken here includes every cost a real execution manager pays. The
//! service is a thin adapter: it neither buffers nor retries, and the only decision it makes is
//! whether a request is pinned to a resource group or general.

use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;

use crate::dispatch_queue::DispatchService;
use crate::error::HarnessError;
use crate::harness::fake_storage::FakeStorage;
use crate::proto::Assignment;
use crate::proto::NextTaskRequest;
use crate::proto::NextTaskResponse;
use crate::proto::PrototypeSchedulerService;
use crate::proto::PrototypeSchedulerServiceServer;
use crate::proto::Void;
use crate::proto::next_task_response;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskId;

/// A tonic server serving one [`DispatchService`] on an ephemeral local port.
#[derive(Debug)]
pub struct HarnessServer {
    endpoint: String,
    cancellation_token: CancellationToken,
    serve_task: JoinHandle<Result<(), tonic::transport::Error>>,
}

impl HarnessServer {
    /// Factory function.
    ///
    /// Binds an ephemeral port on the loopback interface and starts serving in the background, so
    /// that concurrently running integration tests never collide on a port.
    ///
    /// # Returns
    ///
    /// A newly created server already accepting connections on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`HarnessError::Bind`] if the loopback port could not be bound or its assigned address
    ///   could not be read back.
    pub async fn start(
        dispatch_service: DispatchService,
        storage: FakeStorage,
    ) -> Result<Self, HarnessError> {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .map_err(|error| HarnessError::Bind(error.to_string()))?;
        let local_addr = listener
            .local_addr()
            .map_err(|error| HarnessError::Bind(error.to_string()))?;

        let cancellation_token = CancellationToken::new();
        let shutdown_token = cancellation_token.clone();
        let serve_task = tokio::spawn(
            Server::builder()
                .add_service(PrototypeSchedulerServiceServer::new(Service {
                    dispatch_service,
                    storage,
                }))
                .serve_with_incoming_shutdown(TcpIncoming::from(listener), async move {
                    shutdown_token.cancelled().await;
                }),
        );

        Ok(Self {
            endpoint: format!("http://{local_addr}"),
            cancellation_token,
            serve_task,
        })
    }

    /// # Returns
    ///
    /// The URL a client connects to in order to reach this server.
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.endpoint.clone()
    }

    /// Asks the server to stop accepting connections and waits for it to finish serving.
    pub async fn shutdown(self) {
        self.cancellation_token.cancel();
        match self.serve_task.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    error = % error,
                    "Harness server stopped with a transport error."
                );
            }
            Err(error) => {
                tracing::warn!(
                    error = % error,
                    "Harness server task could not be joined."
                );
            }
        }
    }
}

/// The adapter translating `NextTask` requests into [`DispatchService`] calls.
struct Service {
    dispatch_service: DispatchService,
    storage: FakeStorage,
}

impl Service {
    /// Reports the assignment a worker carried back to storage, if it carried one.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`Status::invalid_argument`] if the completed assignment carries no task ID or one that
    ///   has no core representation.
    fn report_completed(&self, request: &NextTaskRequest) -> Result<(), Status> {
        let Some(completed) = request.completed else {
            return Ok(());
        };
        let wire_task_id = completed
            .task_id
            .ok_or_else(|| Status::invalid_argument("completed assignment has no task ID"))?;
        let task_id = TaskId::try_from(wire_task_id)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        self.storage
            .complete_task(JobId::from(completed.job_id), task_id);
        Ok(())
    }
}

#[tonic::async_trait]
impl PrototypeSchedulerService for Service {
    async fn next_task(
        &self,
        request: Request<NextTaskRequest>,
    ) -> Result<Response<NextTaskResponse>, Status> {
        let request = request.into_inner();
        self.report_completed(&request)?;

        let wait_time = Duration::from_millis(request.wait_time_ms);
        let assignment = if let Some(rg_id) = request.resource_group_id {
            self.dispatch_service
                .next_task_pinned(ResourceGroupId::from(rg_id), wait_time)
                .await
        } else {
            self.dispatch_service.next_task_general(wait_time).await
        };

        let result = assignment.map_or_else(
            || next_task_response::Result::NoTask(Void {}),
            |assignment| next_task_response::Result::Assignment(Assignment::from(assignment)),
        );
        Ok(Response::new(NextTaskResponse {
            result: Some(result),
        }))
    }
}
