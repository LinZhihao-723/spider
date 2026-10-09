//! The per-lane publisher: an [`InboundQueueSender`] whose sends never wait.
//!
//! Sends are staged in unbounded per-lane queues, and one task per lane forwards them into the
//! wrapped, bounded inbound queue. A full lane therefore delays only the delivery of its own ready
//! work, never the caller.

use async_trait::async_trait;
use spider_core::task::TaskIndex;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::cache::error::InternalError;
use crate::inbound_queue::InboundQueueSender;

/// An [`InboundQueueSender`] that stages every send and returns at once.
///
/// Staged entries are dropped when the service shuts down. They are hints: the next start of the
/// service resends the ready work of every cached job.
#[derive(Clone)]
pub struct StagingPublisher {
    task: mpsc::UnboundedSender<StagedSend<Vec<TaskIndex>>>,
    commit: mpsc::UnboundedSender<StagedSend<()>>,
    cleanup: mpsc::UnboundedSender<StagedSend<()>>,
}

impl StagingPublisher {
    /// Factory function.
    ///
    /// Spawns one forwarding task per lane into `tracker`. Each task stops when `token` is
    /// cancelled.
    ///
    /// # NOTE
    ///
    /// This function must be called within a tokio runtime.
    ///
    /// # Type Parameters
    ///
    /// * `InboundQueueSenderType` - The type of the wrapped inbound queue sender.
    ///
    /// # Returns
    ///
    /// A publisher that forwards into `inner`.
    pub fn spawn<InboundQueueSenderType: InboundQueueSender + 'static>(
        inner: InboundQueueSenderType,
        token: &CancellationToken,
        tracker: &TaskTracker,
    ) -> Self {
        let (task, task_receiver) = mpsc::unbounded_channel();
        let (commit, commit_receiver) = mpsc::unbounded_channel();
        let (cleanup, cleanup_receiver) = mpsc::unbounded_channel();

        let task_inner = inner.clone();
        tracker.spawn(forward(
            "task",
            task_receiver,
            token.clone(),
            move |staged: StagedSend<Vec<TaskIndex>>| {
                let inner = task_inner.clone();
                async move {
                    inner
                        .send_task_ready(staged.resource_group_id, staged.job_id, staged.task_kind)
                        .await
                }
            },
        ));
        let commit_inner = inner.clone();
        tracker.spawn(forward(
            "commit",
            commit_receiver,
            token.clone(),
            move |staged: StagedSend<()>| {
                let inner = commit_inner.clone();
                async move {
                    inner
                        .send_commit_ready(staged.resource_group_id, staged.job_id)
                        .await
                }
            },
        ));
        tracker.spawn(forward(
            "cleanup",
            cleanup_receiver,
            token.clone(),
            move |staged: StagedSend<()>| {
                let inner = inner.clone();
                async move {
                    inner
                        .send_cleanup_ready(staged.resource_group_id, staged.job_id)
                        .await
                }
            },
        ));

        Self {
            task,
            commit,
            cleanup,
        }
    }
}

#[async_trait]
impl InboundQueueSender for StagingPublisher {
    async fn send_task_ready(
        &self,
        resource_group_id: ResourceGroupId,
        job_id: JobId,
        task_indices: Vec<TaskIndex>,
    ) -> Result<(), InternalError> {
        // A send fails only after the service has stopped its lane tasks, when hints are dropped.
        let _ = self.task.send(StagedSend {
            resource_group_id,
            job_id,
            task_kind: task_indices,
        });
        Ok(())
    }

    async fn send_commit_ready(
        &self,
        resource_group_id: ResourceGroupId,
        job_id: JobId,
    ) -> Result<(), InternalError> {
        let _ = self.commit.send(StagedSend {
            resource_group_id,
            job_id,
            task_kind: (),
        });
        Ok(())
    }

    async fn send_cleanup_ready(
        &self,
        resource_group_id: ResourceGroupId,
        job_id: JobId,
    ) -> Result<(), InternalError> {
        let _ = self.cleanup.send(StagedSend {
            resource_group_id,
            job_id,
            task_kind: (),
        });
        Ok(())
    }
}

/// The arguments of one send staged on a lane.
///
/// # Type Parameters
///
/// * `TaskKind` - The lane's task specifier: the ready task indices for regular tasks, or `()` for
///   commit and cleanup tasks.
struct StagedSend<TaskKind> {
    resource_group_id: ResourceGroupId,
    job_id: JobId,
    task_kind: TaskKind,
}

/// Forwards every staged entry of one lane with `send`, until `token` is cancelled or the lane's
/// inbound queue is closed.
///
/// # Type Parameters
///
/// * `Entry` - The type of the lane's staged entries.
/// * `SendFuture` - The type of the future that `send` returns for one entry.
async fn forward<Entry: Send, SendFuture: Future<Output = Result<(), InternalError>> + Send>(
    lane: &'static str,
    mut receiver: mpsc::UnboundedReceiver<Entry>,
    token: CancellationToken,
    send: impl Fn(Entry) -> SendFuture + Send,
) {
    loop {
        let entry = tokio::select! {
            () = token.cancelled() => return,
            entry = receiver.recv() => entry,
        };
        let Some(entry) = entry else {
            return;
        };
        let result = tokio::select! {
            () = token.cancelled() => return,
            result = send(entry) => result,
        };
        if let Err(e) = result {
            tracing::warn!(
                lane,
                error = % e,
                "Inbound queue lane closed. Stopped publishing to it."
            );
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use spider_core::types::id::JobId;
    use spider_core::types::id::ResourceGroupId;
    use tokio_util::sync::CancellationToken;
    use tokio_util::task::TaskTracker;

    use super::StagingPublisher;
    use crate::inbound_queue::InboundQueueConfig;
    use crate::inbound_queue::InboundQueueSender;
    use crate::inbound_queue::create_inbound_queue;

    #[tokio::test]
    async fn staged_entries_reach_a_full_lane_once_it_drains() -> anyhow::Result<()> {
        let (sender, receiver) = create_inbound_queue(&InboundQueueConfig {
            task_capacity: 1,
            ..InboundQueueConfig::default()
        })?;
        let token = CancellationToken::new();
        let tracker = TaskTracker::new();
        let publisher = StagingPublisher::spawn(sender, &token, &tracker);
        let resource_group_id = ResourceGroupId::random();
        let job_id = JobId::random();

        publisher
            .send_task_ready(resource_group_id, job_id, vec![0, 1, 2])
            .await?;
        publisher
            .send_commit_ready(resource_group_id, job_id)
            .await?;

        let commits = receiver.recv_commits(1, Duration::from_secs(5)).await?;
        assert_eq!(commits.len(), 1);
        let mut task_indices = Vec::new();
        while task_indices.len() < 3 {
            let entries = receiver.recv_tasks(1, Duration::from_secs(5)).await?;
            task_indices.extend(entries.into_iter().map(|entry| entry.task_kind));
        }
        assert_eq!(task_indices, vec![0, 1, 2]);

        token.cancel();
        tracker.close();
        tokio::time::timeout(Duration::from_secs(5), tracker.wait()).await?;
        publisher
            .send_cleanup_ready(resource_group_id, job_id)
            .await?;
        Ok(())
    }
}
