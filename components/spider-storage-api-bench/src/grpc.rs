use std::{net::SocketAddr, str::FromStr, sync::Arc};

use async_trait::async_trait;
use tonic::{Request, Response, Status, transport::Server};

use crate::{
    api,
    api::{
        AddResourceGroupRequest,
        ApiError,
        ApiResult,
        CreateTaskInstanceRequest,
        EmptyResponse,
        EndMetricsSessionRequest,
        ErrorCode,
        ExecutionContextResponse,
        ExecutionManagerRequest,
        ExecutionManagerResponse,
        FailTaskInstanceRequest,
        GetSessionRequest,
        JobErrorResponse,
        JobIdRequest,
        JobIdResponse,
        JobOutputsResponse,
        JobStateResponse,
        PollReadyTasksRequest,
        ReadyTasksResponse,
        RegisterExecutionManagerRequest,
        RegisterJobRequest,
        ResourceGroupResponse,
        SessionResponse,
        StartMetricsSessionRequest,
        StartMetricsSessionResponse,
        SucceedTaskInstanceRequest,
        SucceedTerminationTaskInstanceRequest,
        TerminationTaskEntryDto,
        TerminationTasksResponse,
        VerifyResourceGroupRequest,
    },
    client::StorageApiClient,
    metrics::{
        RequestLatencySample,
        RequestLatencySummary,
        RequestSizeSummary,
        ServerMetricsSessionReport,
    },
    server::StorageApiService,
};

pub mod proto {
    #![allow(clippy::all, clippy::nursery, clippy::pedantic)]

    tonic::include_proto!("spider.storage_api_bench.v1");
}

pub(crate) async fn serve(bind: SocketAddr, service: Arc<StorageApiService>) -> anyhow::Result<()> {
    Server::builder()
        .add_service(proto::storage_api_server::StorageApiServer::new(
            GrpcStorageApiService { service },
        ))
        .serve_with_shutdown(bind, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

pub struct GrpcStorageApiService {
    service: Arc<StorageApiService>,
}

#[tonic::async_trait]
impl proto::storage_api_server::StorageApi for GrpcStorageApiService {
    async fn get_session(
        &self,
        request: Request<proto::GetSessionRequest>,
    ) -> Result<Response<proto::SessionResponse>, Status> {
        let response = self.service.get_session(request.into_inner().into());
        Ok(Response::new(response.into()))
    }

    async fn add_resource_group(
        &self,
        request: Request<proto::AddResourceGroupRequest>,
    ) -> Result<Response<proto::ResourceGroupResponse>, Status> {
        let response = self
            .service
            .add_resource_group(request.into_inner().into())
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn start_metrics_session(
        &self,
        request: Request<proto::StartMetricsSessionRequest>,
    ) -> Result<Response<proto::StartMetricsSessionResponse>, Status> {
        let response = self
            .service
            .start_metrics_session(request.into_inner().into());
        Ok(Response::new(response.into()))
    }

    async fn end_metrics_session(
        &self,
        request: Request<proto::EndMetricsSessionRequest>,
    ) -> Result<Response<proto::ServerMetricsSessionReport>, Status> {
        let response = self
            .service
            .end_metrics_session(request.into_inner().into())?;
        Ok(Response::new(response.try_into()?))
    }

    async fn verify_resource_group(
        &self,
        request: Request<proto::VerifyResourceGroupRequest>,
    ) -> Result<Response<proto::EmptyResponse>, Status> {
        self.service
            .verify_resource_group(request.into_inner().into())
            .await?;
        Ok(Response::new(proto::EmptyResponse {}))
    }

    async fn register_job(
        &self,
        request: Request<proto::RegisterJobRequest>,
    ) -> Result<Response<proto::JobIdResponse>, Status> {
        let response = self
            .service
            .register_job(request.into_inner().into())
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn start_job(
        &self,
        request: Request<proto::JobIdRequest>,
    ) -> Result<Response<proto::EmptyResponse>, Status> {
        self.service.start_job(request.into_inner().into()).await?;
        Ok(Response::new(proto::EmptyResponse {}))
    }

    async fn cancel_job(
        &self,
        request: Request<proto::JobIdRequest>,
    ) -> Result<Response<proto::JobStateResponse>, Status> {
        let response = self.service.cancel_job(request.into_inner().into()).await?;
        Ok(Response::new(response.into()))
    }

    async fn get_job_state(
        &self,
        request: Request<proto::JobIdRequest>,
    ) -> Result<Response<proto::JobStateResponse>, Status> {
        let response = self
            .service
            .get_job_state(request.into_inner().into())
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn get_job_outputs(
        &self,
        request: Request<proto::JobIdRequest>,
    ) -> Result<Response<proto::JobOutputsResponse>, Status> {
        let response = self
            .service
            .get_job_outputs(request.into_inner().into())
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn get_job_error(
        &self,
        request: Request<proto::JobIdRequest>,
    ) -> Result<Response<proto::JobErrorResponse>, Status> {
        let response = self
            .service
            .get_job_error(request.into_inner().into())
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn poll_ready_tasks(
        &self,
        request: Request<proto::PollReadyTasksRequest>,
    ) -> Result<Response<proto::ReadyTasksResponse>, Status> {
        let response = self
            .service
            .poll_ready_tasks(request.into_inner().try_into()?)
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn poll_commit_ready_tasks(
        &self,
        request: Request<proto::PollReadyTasksRequest>,
    ) -> Result<Response<proto::TerminationTasksResponse>, Status> {
        let response = self
            .service
            .poll_commit_ready_tasks(request.into_inner().try_into()?)
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn poll_cleanup_ready_tasks(
        &self,
        request: Request<proto::PollReadyTasksRequest>,
    ) -> Result<Response<proto::TerminationTasksResponse>, Status> {
        let response = self
            .service
            .poll_cleanup_ready_tasks(request.into_inner().try_into()?)
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn create_task_instance(
        &self,
        request: Request<proto::CreateTaskInstanceRequest>,
    ) -> Result<Response<proto::ExecutionContextResponse>, Status> {
        let response = self
            .service
            .create_task_instance(request.into_inner().try_into()?)
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn succeed_task_instance(
        &self,
        request: Request<proto::SucceedTaskInstanceRequest>,
    ) -> Result<Response<proto::JobStateResponse>, Status> {
        let response = self
            .service
            .succeed_task_instance(request.into_inner().try_into()?)
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn succeed_commit_task_instance(
        &self,
        request: Request<proto::SucceedTerminationTaskInstanceRequest>,
    ) -> Result<Response<proto::JobStateResponse>, Status> {
        let response = self
            .service
            .succeed_commit_task_instance(request.into_inner().into())
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn succeed_cleanup_task_instance(
        &self,
        request: Request<proto::SucceedTerminationTaskInstanceRequest>,
    ) -> Result<Response<proto::JobStateResponse>, Status> {
        let response = self
            .service
            .succeed_cleanup_task_instance(request.into_inner().into())
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn fail_task_instance(
        &self,
        request: Request<proto::FailTaskInstanceRequest>,
    ) -> Result<Response<proto::JobStateResponse>, Status> {
        let response = self
            .service
            .fail_task_instance(request.into_inner().try_into()?)
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn register_execution_manager(
        &self,
        request: Request<proto::RegisterExecutionManagerRequest>,
    ) -> Result<Response<proto::ExecutionManagerResponse>, Status> {
        let response = self
            .service
            .register_execution_manager(request.into_inner().try_into()?)
            .await?;
        Ok(Response::new(response.into()))
    }

    async fn update_execution_manager_heartbeat(
        &self,
        request: Request<proto::ExecutionManagerRequest>,
    ) -> Result<Response<proto::EmptyResponse>, Status> {
        self.service
            .update_execution_manager_heartbeat(request.into_inner().into())
            .await?;
        Ok(Response::new(proto::EmptyResponse {}))
    }
}

impl From<ApiError> for Status {
    fn from(error: ApiError) -> Self {
        let code = match error.code {
            ErrorCode::BadRequest => tonic::Code::InvalidArgument,
            ErrorCode::Conflict => tonic::Code::FailedPrecondition,
            ErrorCode::Internal => tonic::Code::Internal,
            ErrorCode::NotFound => tonic::Code::NotFound,
            ErrorCode::Unavailable => tonic::Code::Unavailable,
        };
        Self::new(code, error.message)
    }
}

impl From<Status> for ApiError {
    fn from(status: Status) -> Self {
        let code = match status.code() {
            tonic::Code::InvalidArgument => ErrorCode::BadRequest,
            tonic::Code::FailedPrecondition | tonic::Code::AlreadyExists => ErrorCode::Conflict,
            tonic::Code::NotFound => ErrorCode::NotFound,
            tonic::Code::Unavailable => ErrorCode::Unavailable,
            _ => ErrorCode::Internal,
        };
        Self {
            code,
            message: status.message().to_owned(),
            retryable: matches!(code, ErrorCode::Internal | ErrorCode::Unavailable),
        }
    }
}

#[derive(Clone)]
pub struct GrpcStorageApiClient {
    client: proto::storage_api_client::StorageApiClient<tonic::transport::Channel>,
}

impl GrpcStorageApiClient {
    /// Connects to a gRPC storage API server.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC channel cannot be established.
    pub async fn connect(target: String) -> ApiResult<Self> {
        let client = proto::storage_api_client::StorageApiClient::connect(target)
            .await
            .map_err(|e| ApiError::internal(format!("gRPC connect failed: {e}")))?;
        Ok(Self { client })
    }
}

#[async_trait]
impl StorageApiClient for GrpcStorageApiClient {
    async fn get_session(&self, request: GetSessionRequest) -> ApiResult<SessionResponse> {
        let mut client = self.client.clone();
        Ok(client
            .get_session(proto::GetSessionRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn start_metrics_session(
        &self,
        request: StartMetricsSessionRequest,
    ) -> ApiResult<StartMetricsSessionResponse> {
        let mut client = self.client.clone();
        Ok(client
            .start_metrics_session(proto::StartMetricsSessionRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn end_metrics_session(
        &self,
        request: EndMetricsSessionRequest,
    ) -> ApiResult<ServerMetricsSessionReport> {
        let mut client = self.client.clone();
        Ok(client
            .end_metrics_session(proto::EndMetricsSessionRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn add_resource_group(
        &self,
        request: AddResourceGroupRequest,
    ) -> ApiResult<ResourceGroupResponse> {
        let mut client = self.client.clone();
        Ok(client
            .add_resource_group(proto::AddResourceGroupRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn verify_resource_group(
        &self,
        request: VerifyResourceGroupRequest,
    ) -> ApiResult<EmptyResponse> {
        let mut client = self.client.clone();
        client
            .verify_resource_group(proto::VerifyResourceGroupRequest::from(request))
            .await?;
        drop(client);
        Ok(EmptyResponse {})
    }

    async fn register_job(&self, request: RegisterJobRequest) -> ApiResult<JobIdResponse> {
        let mut client = self.client.clone();
        Ok(client
            .register_job(proto::RegisterJobRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn start_job(&self, request: JobIdRequest) -> ApiResult<EmptyResponse> {
        let mut client = self.client.clone();
        client.start_job(proto::JobIdRequest::from(request)).await?;
        drop(client);
        Ok(EmptyResponse {})
    }

    async fn cancel_job(&self, request: JobIdRequest) -> ApiResult<JobStateResponse> {
        let mut client = self.client.clone();
        Ok(client
            .cancel_job(proto::JobIdRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn get_job_state(&self, request: JobIdRequest) -> ApiResult<JobStateResponse> {
        let mut client = self.client.clone();
        Ok(client
            .get_job_state(proto::JobIdRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn get_job_outputs(&self, request: JobIdRequest) -> ApiResult<JobOutputsResponse> {
        let mut client = self.client.clone();
        Ok(client
            .get_job_outputs(proto::JobIdRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn get_job_error(&self, request: JobIdRequest) -> ApiResult<JobErrorResponse> {
        let mut client = self.client.clone();
        Ok(client
            .get_job_error(proto::JobIdRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn poll_ready_tasks(
        &self,
        request: PollReadyTasksRequest,
    ) -> ApiResult<ReadyTasksResponse> {
        let mut client = self.client.clone();
        Ok(client
            .poll_ready_tasks(proto::PollReadyTasksRequest::try_from(request)?)
            .await?
            .into_inner()
            .into())
    }

    async fn poll_commit_ready_tasks(
        &self,
        request: PollReadyTasksRequest,
    ) -> ApiResult<TerminationTasksResponse> {
        let mut client = self.client.clone();
        Ok(client
            .poll_commit_ready_tasks(proto::PollReadyTasksRequest::try_from(request)?)
            .await?
            .into_inner()
            .into())
    }

    async fn poll_cleanup_ready_tasks(
        &self,
        request: PollReadyTasksRequest,
    ) -> ApiResult<TerminationTasksResponse> {
        let mut client = self.client.clone();
        Ok(client
            .poll_cleanup_ready_tasks(proto::PollReadyTasksRequest::try_from(request)?)
            .await?
            .into_inner()
            .into())
    }

    async fn create_task_instance(
        &self,
        request: CreateTaskInstanceRequest,
    ) -> ApiResult<ExecutionContextResponse> {
        let mut client = self.client.clone();
        Ok(client
            .create_task_instance(proto::CreateTaskInstanceRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn succeed_task_instance(
        &self,
        request: SucceedTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        let mut client = self.client.clone();
        Ok(client
            .succeed_task_instance(proto::SucceedTaskInstanceRequest::try_from(request)?)
            .await?
            .into_inner()
            .into())
    }

    async fn succeed_commit_task_instance(
        &self,
        request: SucceedTerminationTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        let mut client = self.client.clone();
        Ok(client
            .succeed_commit_task_instance(proto::SucceedTerminationTaskInstanceRequest::from(
                request,
            ))
            .await?
            .into_inner()
            .into())
    }

    async fn succeed_cleanup_task_instance(
        &self,
        request: SucceedTerminationTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        let mut client = self.client.clone();
        Ok(client
            .succeed_cleanup_task_instance(proto::SucceedTerminationTaskInstanceRequest::from(
                request,
            ))
            .await?
            .into_inner()
            .into())
    }

    async fn fail_task_instance(
        &self,
        request: FailTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        let mut client = self.client.clone();
        Ok(client
            .fail_task_instance(proto::FailTaskInstanceRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn register_execution_manager(
        &self,
        request: RegisterExecutionManagerRequest,
    ) -> ApiResult<ExecutionManagerResponse> {
        let mut client = self.client.clone();
        Ok(client
            .register_execution_manager(proto::RegisterExecutionManagerRequest::from(request))
            .await?
            .into_inner()
            .into())
    }

    async fn update_execution_manager_heartbeat(
        &self,
        request: ExecutionManagerRequest,
    ) -> ApiResult<EmptyResponse> {
        let mut client = self.client.clone();
        client
            .update_execution_manager_heartbeat(proto::ExecutionManagerRequest::from(request))
            .await?;
        drop(client);
        Ok(EmptyResponse {})
    }
}

impl From<proto::GetSessionRequest> for GetSessionRequest {
    fn from(_request: proto::GetSessionRequest) -> Self {
        Self {}
    }
}

impl From<GetSessionRequest> for proto::GetSessionRequest {
    fn from(_request: GetSessionRequest) -> Self {
        Self {}
    }
}

impl From<SessionResponse> for proto::SessionResponse {
    fn from(response: SessionResponse) -> Self {
        Self {
            session_id: response.session_id,
        }
    }
}

impl From<proto::SessionResponse> for SessionResponse {
    fn from(response: proto::SessionResponse) -> Self {
        Self {
            session_id: response.session_id,
        }
    }
}

impl From<proto::StartMetricsSessionRequest> for StartMetricsSessionRequest {
    fn from(request: proto::StartMetricsSessionRequest) -> Self {
        Self {
            label: if request.label.is_empty() {
                None
            } else {
                Some(request.label)
            },
        }
    }
}

impl From<StartMetricsSessionRequest> for proto::StartMetricsSessionRequest {
    fn from(request: StartMetricsSessionRequest) -> Self {
        Self {
            label: request.label.unwrap_or_default(),
        }
    }
}

impl From<StartMetricsSessionResponse> for proto::StartMetricsSessionResponse {
    fn from(response: StartMetricsSessionResponse) -> Self {
        Self {
            metrics_session_id: response.metrics_session_id,
        }
    }
}

impl From<proto::StartMetricsSessionResponse> for StartMetricsSessionResponse {
    fn from(response: proto::StartMetricsSessionResponse) -> Self {
        Self {
            metrics_session_id: response.metrics_session_id,
        }
    }
}

impl From<proto::EndMetricsSessionRequest> for EndMetricsSessionRequest {
    fn from(request: proto::EndMetricsSessionRequest) -> Self {
        Self {
            metrics_session_id: request.metrics_session_id,
        }
    }
}

impl From<EndMetricsSessionRequest> for proto::EndMetricsSessionRequest {
    fn from(request: EndMetricsSessionRequest) -> Self {
        Self {
            metrics_session_id: request.metrics_session_id,
        }
    }
}

impl TryFrom<ServerMetricsSessionReport> for proto::ServerMetricsSessionReport {
    type Error = Status;

    fn try_from(report: ServerMetricsSessionReport) -> Result<Self, Self::Error> {
        Ok(Self {
            metrics_session_id: report.metrics_session_id,
            label: report.label.unwrap_or_default(),
            elapsed_micros: u64::try_from(report.elapsed_micros)
                .map_err(|_| Status::internal("elapsed_micros overflows u64"))?,
            request_latency: report
                .request_latency
                .into_iter()
                .map(proto::RequestLatencySummary::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            low_count_request_latency: report
                .low_count_request_latency
                .into_iter()
                .map(proto::RequestLatencySample::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            request_sizes: report
                .request_sizes
                .into_iter()
                .map(proto::RequestSizeSummary::from)
                .collect(),
        })
    }
}

impl From<proto::ServerMetricsSessionReport> for ServerMetricsSessionReport {
    fn from(report: proto::ServerMetricsSessionReport) -> Self {
        Self {
            metrics_session_id: report.metrics_session_id,
            label: if report.label.is_empty() {
                None
            } else {
                Some(report.label)
            },
            elapsed_micros: u128::from(report.elapsed_micros),
            request_latency: report
                .request_latency
                .into_iter()
                .map(RequestLatencySummary::from)
                .collect(),
            low_count_request_latency: report
                .low_count_request_latency
                .into_iter()
                .map(RequestLatencySample::from)
                .collect(),
            request_sizes: report
                .request_sizes
                .into_iter()
                .map(RequestSizeSummary::from)
                .collect(),
        }
    }
}

impl From<RequestSizeSummary> for proto::RequestSizeSummary {
    fn from(summary: RequestSizeSummary) -> Self {
        Self {
            operation: summary.operation,
            count: u64::try_from(summary.count).unwrap_or(u64::MAX),
            avg_bytes: summary.avg_bytes,
            p50_bytes: summary.p50_bytes,
            p99_bytes: summary.p99_bytes,
            max_bytes: summary.max_bytes,
            total_bytes: summary.total_bytes,
        }
    }
}

impl From<proto::RequestSizeSummary> for RequestSizeSummary {
    fn from(summary: proto::RequestSizeSummary) -> Self {
        Self {
            operation: summary.operation,
            count: usize::try_from(summary.count).unwrap_or(usize::MAX),
            avg_bytes: summary.avg_bytes,
            p50_bytes: summary.p50_bytes,
            p99_bytes: summary.p99_bytes,
            max_bytes: summary.max_bytes,
            total_bytes: summary.total_bytes,
        }
    }
}

impl TryFrom<RequestLatencySample> for proto::RequestLatencySample {
    type Error = Status;

    fn try_from(sample: RequestLatencySample) -> Result<Self, Self::Error> {
        Ok(Self {
            category: sample.category,
            operation: sample.operation,
            latency_micros: u64::try_from(sample.latency_micros)
                .map_err(|_| Status::internal("latency_micros overflows u64"))?,
            succeeded: sample.succeeded,
        })
    }
}

impl From<proto::RequestLatencySample> for RequestLatencySample {
    fn from(sample: proto::RequestLatencySample) -> Self {
        Self {
            category: sample.category,
            operation: sample.operation,
            latency_micros: u128::from(sample.latency_micros),
            succeeded: sample.succeeded,
        }
    }
}

impl TryFrom<RequestLatencySummary> for proto::RequestLatencySummary {
    type Error = Status;

    fn try_from(summary: RequestLatencySummary) -> Result<Self, Self::Error> {
        Ok(Self {
            category: summary.category,
            operation: summary.operation,
            count: u64::try_from(summary.count)
                .map_err(|_| Status::internal("count overflows u64"))?,
            errors: u64::try_from(summary.errors)
                .map_err(|_| Status::internal("errors overflows u64"))?,
            avg_us: u64::try_from(summary.avg_us)
                .map_err(|_| Status::internal("avg_us overflows u64"))?,
            p50_us: u64::try_from(summary.p50_us)
                .map_err(|_| Status::internal("p50_us overflows u64"))?,
            p90_us: u64::try_from(summary.p90_us)
                .map_err(|_| Status::internal("p90_us overflows u64"))?,
            p99_us: u64::try_from(summary.p99_us)
                .map_err(|_| Status::internal("p99_us overflows u64"))?,
            max_us: u64::try_from(summary.max_us)
                .map_err(|_| Status::internal("max_us overflows u64"))?,
        })
    }
}

impl From<proto::RequestLatencySummary> for RequestLatencySummary {
    fn from(summary: proto::RequestLatencySummary) -> Self {
        Self {
            category: summary.category,
            operation: summary.operation,
            count: usize::try_from(summary.count)
                .expect("server request count should fit in usize"),
            errors: usize::try_from(summary.errors)
                .expect("server request errors should fit in usize"),
            avg_us: u128::from(summary.avg_us),
            p50_us: u128::from(summary.p50_us),
            p90_us: u128::from(summary.p90_us),
            p99_us: u128::from(summary.p99_us),
            max_us: u128::from(summary.max_us),
        }
    }
}

impl From<proto::AddResourceGroupRequest> for AddResourceGroupRequest {
    fn from(request: proto::AddResourceGroupRequest) -> Self {
        Self {
            external_id: request.external_id,
            password: request.password,
        }
    }
}

impl From<AddResourceGroupRequest> for proto::AddResourceGroupRequest {
    fn from(request: AddResourceGroupRequest) -> Self {
        Self {
            external_id: request.external_id,
            password: request.password,
        }
    }
}

impl From<proto::VerifyResourceGroupRequest> for VerifyResourceGroupRequest {
    fn from(request: proto::VerifyResourceGroupRequest) -> Self {
        Self {
            resource_group_id: request.resource_group_id,
            password: request.password,
        }
    }
}

impl From<VerifyResourceGroupRequest> for proto::VerifyResourceGroupRequest {
    fn from(request: VerifyResourceGroupRequest) -> Self {
        Self {
            resource_group_id: request.resource_group_id,
            password: request.password,
        }
    }
}

impl From<ResourceGroupResponse> for proto::ResourceGroupResponse {
    fn from(response: ResourceGroupResponse) -> Self {
        Self {
            resource_group_id: response.resource_group_id,
        }
    }
}

impl From<proto::ResourceGroupResponse> for ResourceGroupResponse {
    fn from(response: proto::ResourceGroupResponse) -> Self {
        Self {
            resource_group_id: response.resource_group_id,
        }
    }
}

impl From<proto::RegisterJobRequest> for RegisterJobRequest {
    fn from(request: proto::RegisterJobRequest) -> Self {
        Self {
            resource_group_id: request.resource_group_id,
            serialized_task_graph: request.serialized_task_graph,
            serialized_inputs: request.serialized_inputs,
        }
    }
}

impl From<RegisterJobRequest> for proto::RegisterJobRequest {
    fn from(request: RegisterJobRequest) -> Self {
        Self {
            resource_group_id: request.resource_group_id,
            serialized_task_graph: request.serialized_task_graph,
            serialized_inputs: request.serialized_inputs,
        }
    }
}

impl From<proto::JobIdRequest> for JobIdRequest {
    fn from(request: proto::JobIdRequest) -> Self {
        Self {
            job_id: request.job_id,
        }
    }
}

impl From<JobIdRequest> for proto::JobIdRequest {
    fn from(request: JobIdRequest) -> Self {
        Self {
            job_id: request.job_id,
        }
    }
}

impl From<JobIdResponse> for proto::JobIdResponse {
    fn from(response: JobIdResponse) -> Self {
        Self {
            job_id: response.job_id,
        }
    }
}

impl From<proto::JobIdResponse> for JobIdResponse {
    fn from(response: proto::JobIdResponse) -> Self {
        Self {
            job_id: response.job_id,
        }
    }
}

impl From<JobStateResponse> for proto::JobStateResponse {
    fn from(response: JobStateResponse) -> Self {
        Self {
            state: response.state,
        }
    }
}

impl From<proto::JobStateResponse> for JobStateResponse {
    fn from(response: proto::JobStateResponse) -> Self {
        Self {
            state: response.state,
        }
    }
}

impl From<JobOutputsResponse> for proto::JobOutputsResponse {
    fn from(response: JobOutputsResponse) -> Self {
        Self {
            outputs: response.outputs,
        }
    }
}

impl From<proto::JobOutputsResponse> for JobOutputsResponse {
    fn from(response: proto::JobOutputsResponse) -> Self {
        Self {
            outputs: response.outputs,
        }
    }
}

impl From<JobErrorResponse> for proto::JobErrorResponse {
    fn from(response: JobErrorResponse) -> Self {
        Self {
            error: response.error,
        }
    }
}

impl From<proto::JobErrorResponse> for JobErrorResponse {
    fn from(response: proto::JobErrorResponse) -> Self {
        Self {
            error: response.error,
        }
    }
}

impl TryFrom<proto::PollReadyTasksRequest> for PollReadyTasksRequest {
    type Error = Status;

    fn try_from(request: proto::PollReadyTasksRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            max_tasks: usize::try_from(request.max_tasks)
                .map_err(|_| Status::invalid_argument("max_tasks overflows usize"))?,
            wait_ms: request.wait_ms,
        })
    }
}

impl TryFrom<PollReadyTasksRequest> for proto::PollReadyTasksRequest {
    type Error = ApiError;

    fn try_from(request: PollReadyTasksRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            max_tasks: u64::try_from(request.max_tasks)
                .map_err(|_| ApiError::bad_request("max_tasks overflows u64"))?,
            wait_ms: request.wait_ms,
        })
    }
}

impl From<ReadyTasksResponse> for proto::ReadyTasksResponse {
    fn from(response: ReadyTasksResponse) -> Self {
        Self {
            tasks: response
                .tasks
                .into_iter()
                .map(|task| proto::ReadyTaskEntry {
                    resource_group_id: task.resource_group_id,
                    job_id: task.job_id,
                    task_index: task.task_index as u64,
                })
                .collect(),
        }
    }
}

impl From<proto::ReadyTasksResponse> for ReadyTasksResponse {
    fn from(response: proto::ReadyTasksResponse) -> Self {
        Self {
            tasks: response
                .tasks
                .into_iter()
                .map(|task| api::ReadyTaskEntryDto {
                    resource_group_id: task.resource_group_id,
                    job_id: task.job_id,
                    task_index: usize::try_from(task.task_index)
                        .expect("server task_index should fit in usize"),
                })
                .collect(),
        }
    }
}

impl From<TerminationTasksResponse> for proto::TerminationTasksResponse {
    fn from(response: TerminationTasksResponse) -> Self {
        Self {
            tasks: response
                .tasks
                .into_iter()
                .map(|task| proto::TerminationTaskEntry {
                    resource_group_id: task.resource_group_id,
                    job_id: task.job_id,
                })
                .collect(),
        }
    }
}

impl From<proto::TerminationTasksResponse> for TerminationTasksResponse {
    fn from(response: proto::TerminationTasksResponse) -> Self {
        Self {
            tasks: response
                .tasks
                .into_iter()
                .map(|task| TerminationTaskEntryDto {
                    resource_group_id: task.resource_group_id,
                    job_id: task.job_id,
                })
                .collect(),
        }
    }
}

impl From<api::TaskIdDto> for proto::TaskIdDto {
    fn from(task_id: api::TaskIdDto) -> Self {
        use proto::task_id_dto::Kind;
        let kind = match task_id {
            api::TaskIdDto::Index { task_index } => Kind::TaskIndex(task_index as u64),
            api::TaskIdDto::Commit => Kind::Commit(true),
            api::TaskIdDto::Cleanup => Kind::Cleanup(true),
        };
        Self { kind: Some(kind) }
    }
}

impl TryFrom<Option<proto::TaskIdDto>> for api::TaskIdDto {
    type Error = Status;

    fn try_from(task_id: Option<proto::TaskIdDto>) -> Result<Self, Self::Error> {
        use proto::task_id_dto::Kind;
        match task_id.and_then(|dto| dto.kind) {
            Some(Kind::TaskIndex(task_index)) => Ok(Self::Index {
                task_index: usize::try_from(task_index)
                    .map_err(|_| Status::invalid_argument("task_index overflows usize"))?,
            }),
            Some(Kind::Commit(true)) => Ok(Self::Commit),
            Some(Kind::Cleanup(true)) => Ok(Self::Cleanup),
            _ => Err(Status::invalid_argument("missing task_id")),
        }
    }
}

impl TryFrom<proto::CreateTaskInstanceRequest> for CreateTaskInstanceRequest {
    type Error = Status;

    fn try_from(request: proto::CreateTaskInstanceRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_id: request.task_id.try_into()?,
            execution_manager_id: request.execution_manager_id,
        })
    }
}

impl From<CreateTaskInstanceRequest> for proto::CreateTaskInstanceRequest {
    fn from(request: CreateTaskInstanceRequest) -> Self {
        Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_id: Some(request.task_id.into()),
            execution_manager_id: request.execution_manager_id,
        }
    }
}

impl From<ExecutionContextResponse> for proto::ExecutionContextResponse {
    fn from(response: ExecutionContextResponse) -> Self {
        Self {
            task_instance_id: response.task_instance_id,
            package: response.tdl_context.package,
            task_func: response.tdl_context.task_func,
            soft_timeout_ms: response.timeout_policy.soft_timeout_ms,
            hard_timeout_ms: response.timeout_policy.hard_timeout_ms,
            serialized_inputs: response.serialized_inputs,
        }
    }
}

impl From<proto::ExecutionContextResponse> for ExecutionContextResponse {
    fn from(response: proto::ExecutionContextResponse) -> Self {
        Self {
            task_instance_id: response.task_instance_id,
            tdl_context: spider_core::task::TdlContext {
                package: response.package,
                task_func: response.task_func,
            },
            timeout_policy: spider_core::task::TimeoutPolicy {
                soft_timeout_ms: response.soft_timeout_ms,
                hard_timeout_ms: response.hard_timeout_ms,
            },
            serialized_inputs: response.serialized_inputs,
        }
    }
}

impl TryFrom<proto::SucceedTaskInstanceRequest> for SucceedTaskInstanceRequest {
    type Error = Status;

    fn try_from(request: proto::SucceedTaskInstanceRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_instance_id: request.task_instance_id,
            task_index: usize::try_from(request.task_index)
                .map_err(|_| Status::invalid_argument("task_index overflows usize"))?,
            serialized_outputs: request.serialized_outputs,
        })
    }
}

impl TryFrom<SucceedTaskInstanceRequest> for proto::SucceedTaskInstanceRequest {
    type Error = ApiError;

    fn try_from(request: SucceedTaskInstanceRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_instance_id: request.task_instance_id,
            task_index: u64::try_from(request.task_index)
                .map_err(|_| ApiError::bad_request("task_index overflows u64"))?,
            serialized_outputs: request.serialized_outputs,
        })
    }
}

impl From<proto::SucceedTerminationTaskInstanceRequest> for SucceedTerminationTaskInstanceRequest {
    fn from(request: proto::SucceedTerminationTaskInstanceRequest) -> Self {
        Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_instance_id: request.task_instance_id,
        }
    }
}

impl From<SucceedTerminationTaskInstanceRequest> for proto::SucceedTerminationTaskInstanceRequest {
    fn from(request: SucceedTerminationTaskInstanceRequest) -> Self {
        Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_instance_id: request.task_instance_id,
        }
    }
}

impl TryFrom<proto::FailTaskInstanceRequest> for FailTaskInstanceRequest {
    type Error = Status;

    fn try_from(request: proto::FailTaskInstanceRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_instance_id: request.task_instance_id,
            task_id: request.task_id.try_into()?,
            error: request.error,
        })
    }
}

impl From<FailTaskInstanceRequest> for proto::FailTaskInstanceRequest {
    fn from(request: FailTaskInstanceRequest) -> Self {
        Self {
            session_id: request.session_id,
            job_id: request.job_id,
            task_instance_id: request.task_instance_id,
            task_id: Some(request.task_id.into()),
            error: request.error,
        }
    }
}

impl TryFrom<proto::RegisterExecutionManagerRequest> for RegisterExecutionManagerRequest {
    type Error = Status;

    fn try_from(request: proto::RegisterExecutionManagerRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            ip_address: std::net::IpAddr::from_str(&request.ip_address)
                .map_err(|e| Status::invalid_argument(format!("invalid IP address: {e}")))?,
        })
    }
}

impl From<RegisterExecutionManagerRequest> for proto::RegisterExecutionManagerRequest {
    fn from(request: RegisterExecutionManagerRequest) -> Self {
        Self {
            ip_address: request.ip_address.to_string(),
        }
    }
}

impl From<ExecutionManagerRequest> for proto::ExecutionManagerRequest {
    fn from(request: ExecutionManagerRequest) -> Self {
        Self {
            execution_manager_id: request.execution_manager_id,
        }
    }
}

impl From<proto::ExecutionManagerRequest> for ExecutionManagerRequest {
    fn from(request: proto::ExecutionManagerRequest) -> Self {
        Self {
            execution_manager_id: request.execution_manager_id,
        }
    }
}

impl From<ExecutionManagerResponse> for proto::ExecutionManagerResponse {
    fn from(response: ExecutionManagerResponse) -> Self {
        Self {
            execution_manager_id: response.execution_manager_id,
        }
    }
}

impl From<proto::ExecutionManagerResponse> for ExecutionManagerResponse {
    fn from(response: proto::ExecutionManagerResponse) -> Self {
        Self {
            execution_manager_id: response.execution_manager_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{api, grpc::proto};

    #[test]
    fn task_id_proto_roundtrips_index() {
        let original = api::TaskIdDto::Index { task_index: 42 };
        let proto = proto::TaskIdDto::from(original.clone());
        let decoded = api::TaskIdDto::try_from(Some(proto)).expect("task id should decode");
        assert_eq!(original, decoded);
    }
}
