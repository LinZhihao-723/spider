use std::{net::SocketAddr, sync::Arc};

use async_trait::async_trait;
use axum::{
    Json,
    Router,
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use reqwest::Url;
use serde::{Serialize, de::DeserializeOwned};
use tokio::net::TcpListener;

use crate::{
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
        TerminationTasksResponse,
        VerifyResourceGroupRequest,
    },
    client::StorageApiClient,
    metrics::ServerMetricsSessionReport,
    server::StorageApiService,
};

pub(crate) async fn serve(bind: SocketAddr, service: Arc<StorageApiService>) -> anyhow::Result<()> {
    let listener = TcpListener::bind(bind).await?;
    axum::serve(listener, router(service))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

pub(crate) fn router(service: Arc<StorageApiService>) -> Router {
    Router::new()
        .route("/resource-groups", post(add_resource_group))
        .route("/session", get(get_session))
        .route("/metrics-sessions", post(start_metrics_session))
        .route("/metrics-sessions/end", post(end_metrics_session))
        .route("/resource-groups/verify", post(verify_resource_group))
        .route("/jobs/register", post(register_job))
        .route("/jobs/:job_id/start", post(start_job))
        .route("/jobs/:job_id/cancel", post(cancel_job))
        .route("/jobs/:job_id/state", get(get_job_state))
        .route("/jobs/:job_id/outputs", get(get_job_outputs))
        .route("/jobs/:job_id/error", get(get_job_error))
        .route("/ready-tasks", post(poll_ready_tasks))
        .route("/ready-commit-tasks", post(poll_commit_ready_tasks))
        .route("/ready-cleanup-tasks", post(poll_cleanup_ready_tasks))
        .route("/task-instances/create", post(create_task_instance))
        .route("/task-instances/succeed", post(succeed_task_instance))
        .route(
            "/task-instances/commit/succeed",
            post(succeed_commit_task_instance),
        )
        .route(
            "/task-instances/cleanup/succeed",
            post(succeed_cleanup_task_instance),
        )
        .route("/task-instances/fail", post(fail_task_instance))
        .route("/execution-managers", post(register_execution_manager))
        .route(
            "/execution-managers/:execution_manager_id/heartbeat",
            post(update_execution_manager_heartbeat),
        )
        .layer(DefaultBodyLimit::max(REST_MAX_BODY_BYTES))
        .with_state(service)
}

/// Maximum REST request/response body in bytes. Generous enough to accept the JSON-encoded
/// `Vec<u8>` task inputs at large `payload_bytes` settings (one raw byte expands to up to ~4 JSON
/// characters).
const REST_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.code {
            ErrorCode::BadRequest => StatusCode::BAD_REQUEST,
            ErrorCode::Conflict => StatusCode::CONFLICT,
            ErrorCode::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorCode::NotFound => StatusCode::NOT_FOUND,
            ErrorCode::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        };
        (status, Json(self)).into_response()
    }
}

async fn get_session(
    State(service): State<Arc<StorageApiService>>,
) -> Result<Json<SessionResponse>, ApiError> {
    Ok(Json(service.get_session(GetSessionRequest {})))
}

async fn add_resource_group(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<AddResourceGroupRequest>,
) -> Result<Json<ResourceGroupResponse>, ApiError> {
    service.add_resource_group(request).await.map(Json)
}

async fn start_metrics_session(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<StartMetricsSessionRequest>,
) -> Result<Json<StartMetricsSessionResponse>, ApiError> {
    Ok(Json(service.start_metrics_session(request)))
}

async fn end_metrics_session(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<EndMetricsSessionRequest>,
) -> Result<Json<ServerMetricsSessionReport>, ApiError> {
    service.end_metrics_session(request).map(Json)
}

async fn verify_resource_group(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<VerifyResourceGroupRequest>,
) -> Result<Json<EmptyResponse>, ApiError> {
    service.verify_resource_group(request).await.map(Json)
}

async fn register_job(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<RegisterJobRequest>,
) -> Result<Json<JobIdResponse>, ApiError> {
    service.register_job(request).await.map(Json)
}

async fn start_job(
    State(service): State<Arc<StorageApiService>>,
    Path(job_id): Path<String>,
) -> Result<Json<EmptyResponse>, ApiError> {
    service.start_job(JobIdRequest { job_id }).await.map(Json)
}

async fn cancel_job(
    State(service): State<Arc<StorageApiService>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobStateResponse>, ApiError> {
    service.cancel_job(JobIdRequest { job_id }).await.map(Json)
}

async fn get_job_state(
    State(service): State<Arc<StorageApiService>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobStateResponse>, ApiError> {
    service
        .get_job_state(JobIdRequest { job_id })
        .await
        .map(Json)
}

async fn get_job_outputs(
    State(service): State<Arc<StorageApiService>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobOutputsResponse>, ApiError> {
    service
        .get_job_outputs(JobIdRequest { job_id })
        .await
        .map(Json)
}

async fn get_job_error(
    State(service): State<Arc<StorageApiService>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobErrorResponse>, ApiError> {
    service
        .get_job_error(JobIdRequest { job_id })
        .await
        .map(Json)
}

async fn poll_ready_tasks(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<PollReadyTasksRequest>,
) -> Result<Json<ReadyTasksResponse>, ApiError> {
    service.poll_ready_tasks(request).await.map(Json)
}

async fn poll_commit_ready_tasks(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<PollReadyTasksRequest>,
) -> Result<Json<TerminationTasksResponse>, ApiError> {
    service.poll_commit_ready_tasks(request).await.map(Json)
}

async fn poll_cleanup_ready_tasks(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<PollReadyTasksRequest>,
) -> Result<Json<TerminationTasksResponse>, ApiError> {
    service.poll_cleanup_ready_tasks(request).await.map(Json)
}

async fn create_task_instance(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<CreateTaskInstanceRequest>,
) -> Result<Json<ExecutionContextResponse>, ApiError> {
    service.create_task_instance(request).await.map(Json)
}

async fn succeed_task_instance(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<SucceedTaskInstanceRequest>,
) -> Result<Json<JobStateResponse>, ApiError> {
    service.succeed_task_instance(request).await.map(Json)
}

async fn succeed_commit_task_instance(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<SucceedTerminationTaskInstanceRequest>,
) -> Result<Json<JobStateResponse>, ApiError> {
    service
        .succeed_commit_task_instance(request)
        .await
        .map(Json)
}

async fn succeed_cleanup_task_instance(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<SucceedTerminationTaskInstanceRequest>,
) -> Result<Json<JobStateResponse>, ApiError> {
    service
        .succeed_cleanup_task_instance(request)
        .await
        .map(Json)
}

async fn fail_task_instance(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<FailTaskInstanceRequest>,
) -> Result<Json<JobStateResponse>, ApiError> {
    service.fail_task_instance(request).await.map(Json)
}

async fn register_execution_manager(
    State(service): State<Arc<StorageApiService>>,
    Json(request): Json<RegisterExecutionManagerRequest>,
) -> Result<Json<ExecutionManagerResponse>, ApiError> {
    service.register_execution_manager(request).await.map(Json)
}

async fn update_execution_manager_heartbeat(
    State(service): State<Arc<StorageApiService>>,
    Path(execution_manager_id): Path<String>,
) -> Result<Json<EmptyResponse>, ApiError> {
    service
        .update_execution_manager_heartbeat(ExecutionManagerRequest {
            execution_manager_id,
        })
        .await
        .map(Json)
}

#[derive(Clone)]
pub struct RestStorageApiClient {
    base_url: Url,
    client: reqwest::Client,
}

impl RestStorageApiClient {
    /// Creates a REST client for the given base URL.
    ///
    /// # Errors
    ///
    /// Returns an error if `base_url` is not a valid URL.
    pub fn new(base_url: &str) -> ApiResult<Self> {
        Ok(Self {
            base_url: Url::parse(base_url)
                .map_err(|e| ApiError::bad_request(format!("invalid REST target: {e}")))?,
            client: reqwest::Client::new(),
        })
    }

    async fn post<RequestType, ResponseType>(
        &self,
        path: &str,
        request: &RequestType,
    ) -> ApiResult<ResponseType>
    where
        RequestType: Serialize + Sync,
        ResponseType: DeserializeOwned, {
        let url = self
            .base_url
            .join(path)
            .map_err(|e| ApiError::bad_request(format!("invalid REST path `{path}`: {e}")))?;
        let response = self
            .client
            .post(url)
            .json(request)
            .send()
            .await
            .map_err(|e| ApiError::internal(format!("REST request failed: {e}")))?;
        decode_response(response).await
    }

    async fn post_empty<ResponseType>(&self, path: &str) -> ApiResult<ResponseType>
    where
        ResponseType: DeserializeOwned, {
        let url = self
            .base_url
            .join(path)
            .map_err(|e| ApiError::bad_request(format!("invalid REST path `{path}`: {e}")))?;
        let response = self
            .client
            .post(url)
            .send()
            .await
            .map_err(|e| ApiError::internal(format!("REST request failed: {e}")))?;
        decode_response(response).await
    }

    async fn get<ResponseType>(&self, path: &str) -> ApiResult<ResponseType>
    where
        ResponseType: DeserializeOwned, {
        let url = self
            .base_url
            .join(path)
            .map_err(|e| ApiError::bad_request(format!("invalid REST path `{path}`: {e}")))?;
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| ApiError::internal(format!("REST request failed: {e}")))?;
        decode_response(response).await
    }
}

async fn decode_response<ResponseType>(response: reqwest::Response) -> ApiResult<ResponseType>
where
    ResponseType: DeserializeOwned, {
    if response.status().is_success() {
        return response
            .json()
            .await
            .map_err(|e| ApiError::internal(format!("REST response decode failed: {e}")));
    }
    Err(response
        .json::<ApiError>()
        .await
        .unwrap_or_else(|e| ApiError::internal(format!("REST error decode failed: {e}"))))
}

#[async_trait]
impl StorageApiClient for RestStorageApiClient {
    async fn get_session(&self, _request: GetSessionRequest) -> ApiResult<SessionResponse> {
        self.get("session").await
    }

    async fn start_metrics_session(
        &self,
        request: StartMetricsSessionRequest,
    ) -> ApiResult<StartMetricsSessionResponse> {
        self.post("metrics-sessions", &request).await
    }

    async fn end_metrics_session(
        &self,
        request: EndMetricsSessionRequest,
    ) -> ApiResult<ServerMetricsSessionReport> {
        self.post("metrics-sessions/end", &request).await
    }

    async fn add_resource_group(
        &self,
        request: AddResourceGroupRequest,
    ) -> ApiResult<ResourceGroupResponse> {
        self.post("resource-groups", &request).await
    }

    async fn verify_resource_group(
        &self,
        request: VerifyResourceGroupRequest,
    ) -> ApiResult<EmptyResponse> {
        self.post("resource-groups/verify", &request).await
    }

    async fn register_job(&self, request: RegisterJobRequest) -> ApiResult<JobIdResponse> {
        self.post("jobs/register", &request).await
    }

    async fn start_job(&self, request: JobIdRequest) -> ApiResult<EmptyResponse> {
        self.post_empty(&format!("jobs/{}/start", request.job_id))
            .await
    }

    async fn cancel_job(&self, request: JobIdRequest) -> ApiResult<JobStateResponse> {
        self.post_empty(&format!("jobs/{}/cancel", request.job_id))
            .await
    }

    async fn get_job_state(&self, request: JobIdRequest) -> ApiResult<JobStateResponse> {
        self.get(&format!("jobs/{}/state", request.job_id)).await
    }

    async fn get_job_outputs(&self, request: JobIdRequest) -> ApiResult<JobOutputsResponse> {
        self.get(&format!("jobs/{}/outputs", request.job_id)).await
    }

    async fn get_job_error(&self, request: JobIdRequest) -> ApiResult<JobErrorResponse> {
        self.get(&format!("jobs/{}/error", request.job_id)).await
    }

    async fn poll_ready_tasks(
        &self,
        request: PollReadyTasksRequest,
    ) -> ApiResult<ReadyTasksResponse> {
        self.post("ready-tasks", &request).await
    }

    async fn poll_commit_ready_tasks(
        &self,
        request: PollReadyTasksRequest,
    ) -> ApiResult<TerminationTasksResponse> {
        self.post("ready-commit-tasks", &request).await
    }

    async fn poll_cleanup_ready_tasks(
        &self,
        request: PollReadyTasksRequest,
    ) -> ApiResult<TerminationTasksResponse> {
        self.post("ready-cleanup-tasks", &request).await
    }

    async fn create_task_instance(
        &self,
        request: CreateTaskInstanceRequest,
    ) -> ApiResult<ExecutionContextResponse> {
        self.post("task-instances/create", &request).await
    }

    async fn succeed_task_instance(
        &self,
        request: SucceedTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        self.post("task-instances/succeed", &request).await
    }

    async fn succeed_commit_task_instance(
        &self,
        request: SucceedTerminationTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        self.post("task-instances/commit/succeed", &request).await
    }

    async fn succeed_cleanup_task_instance(
        &self,
        request: SucceedTerminationTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        self.post("task-instances/cleanup/succeed", &request).await
    }

    async fn fail_task_instance(
        &self,
        request: FailTaskInstanceRequest,
    ) -> ApiResult<JobStateResponse> {
        self.post("task-instances/fail", &request).await
    }

    async fn register_execution_manager(
        &self,
        request: RegisterExecutionManagerRequest,
    ) -> ApiResult<ExecutionManagerResponse> {
        self.post("execution-managers", &request).await
    }

    async fn update_execution_manager_heartbeat(
        &self,
        request: ExecutionManagerRequest,
    ) -> ApiResult<EmptyResponse> {
        self.post_empty(&format!(
            "execution-managers/{}/heartbeat",
            request.execution_manager_id
        ))
        .await
    }
}

#[cfg(test)]
mod tests {
    use axum::{http::StatusCode, response::IntoResponse};

    use crate::api::{ApiError, ErrorCode};

    #[test]
    fn api_error_maps_bad_request_to_400() {
        let response = ApiError {
            code: ErrorCode::BadRequest,
            message: "invalid".to_owned(),
            retryable: false,
        }
        .into_response();
        assert_eq!(StatusCode::BAD_REQUEST, response.status());
    }
}
