//! The prototype scheduler's gRPC protocol and its conversions to the Spider core types.

use crate::error::HarnessError;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::TaskAssignment;
use crate::types::TaskAssignmentId;
use crate::types::TaskId as CoreTaskId;

#[allow(clippy::all, clippy::nursery, clippy::pedantic)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/prototype_scheduler.rs"));
}

pub use generated::prototype_scheduler_service_client::PrototypeSchedulerServiceClient;
pub use generated::prototype_scheduler_service_server::PrototypeSchedulerService;
pub use generated::prototype_scheduler_service_server::PrototypeSchedulerServiceServer;
pub use generated::*;

impl From<CoreTaskId> for TaskId {
    fn from(task_id: CoreTaskId) -> Self {
        let kind = match task_id {
            CoreTaskId::Index(task_index) => task_id::Kind::Index(
                u64::try_from(task_index).expect("task index does not fit in u64"),
            ),
            CoreTaskId::Commit => task_id::Kind::Commit(Void {}),
            CoreTaskId::Cleanup => task_id::Kind::Cleanup(Void {}),
        };
        Self { kind: Some(kind) }
    }
}

impl TryFrom<TaskId> for CoreTaskId {
    type Error = HarnessError;

    fn try_from(task_id: TaskId) -> Result<Self, Self::Error> {
        match task_id.kind {
            Some(task_id::Kind::Index(task_index)) => {
                Ok(Self::Index(usize::try_from(task_index).map_err(|_| {
                    HarnessError::InvalidMessage(format!("task index out of range: {task_index}"))
                })?))
            }
            Some(task_id::Kind::Commit(_)) => Ok(Self::Commit),
            Some(task_id::Kind::Cleanup(_)) => Ok(Self::Cleanup),
            None => Err(HarnessError::InvalidMessage(
                "task ID kind missing".to_owned(),
            )),
        }
    }
}

impl From<TaskAssignment> for Assignment {
    fn from(assignment: TaskAssignment) -> Self {
        Self {
            id: assignment.id.get(),
            resource_group_id: assignment.resource_group_id.get(),
            job_id: assignment.job_id.get(),
            task_id: Some(TaskId::from(assignment.task_id)),
            session_id: assignment.session_id,
        }
    }
}

impl TryFrom<Assignment> for TaskAssignment {
    type Error = HarnessError;

    fn try_from(assignment: Assignment) -> Result<Self, Self::Error> {
        let task_id = assignment
            .task_id
            .ok_or_else(|| HarnessError::InvalidMessage("task ID missing".to_owned()))?;
        Ok(Self {
            id: TaskAssignmentId::from(assignment.id),
            resource_group_id: ResourceGroupId::from(assignment.resource_group_id),
            job_id: JobId::from(assignment.job_id),
            task_id: CoreTaskId::try_from(task_id)?,
            session_id: assignment.session_id,
        })
    }
}
