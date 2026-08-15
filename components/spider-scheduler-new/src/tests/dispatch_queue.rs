//! Unit tests for the hint channel and the two paths an execution manager pulls assignments
//! through.

use std::sync::atomic::Ordering;
use std::time::Duration;

use super::DEFAULT_SESSION_ID;
use super::drain_reader;
use super::make_assignment;
use super::make_job_entry;
use super::make_unit;
use super::reader_of;
use crate::core::TaskAssignmentIdIssuer;
use crate::dispatch_queue::DispatchService;
use crate::dispatch_queue::GlobalDispatchQueue;
use crate::job_registry::JobRegistry;
use crate::resource_group::ResourceGroupTable;
use crate::scheduling_unit::RgSchedulingUnit;
use crate::session::SessionManager;
use crate::types::JobId;
use crate::types::ResourceGroupId;
use crate::types::SessionId;
use crate::types::TaskId;

/// The resource group every test in this module publishes into.
const RG_ID: ResourceGroupId = ResourceGroupId::from(1);

/// The job every test in this module draws its tasks from.
const JOB_ID: JobId = JobId::from(1);

/// The dispatch buffer capacity, large enough that no test reaches the admission threshold.
const DISPATCH_QUEUE_CAPACITY: usize = 16;

/// The free space passed to every assignment attempt, large enough never to bind.
const FREE_SPACE: usize = DISPATCH_QUEUE_CAPACITY;

/// The time a test lets a dispatch call block before it concludes nothing is coming.
const DISPATCH_WAIT: Duration = Duration::from_millis(50);

/// One resource group's publishing side, wired to the structures a test inspects.
struct PublisherFixture {
    unit: RgSchedulingUnit,
    rg_table: ResourceGroupTable,
    global_queue: GlobalDispatchQueue,
    id_issuer: TaskAssignmentIdIssuer,
}

impl PublisherFixture {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created fixture whose group holds one job of `num_tasks` buffered ready tasks.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`make_job_entry`]'s return values on failure.
    fn new(num_tasks: usize) -> anyhow::Result<Self> {
        let rg_table = ResourceGroupTable::new();
        let mut unit = make_unit(&rg_table, RG_ID, DEFAULT_SESSION_ID, 1);
        let mut registry = JobRegistry::new();
        unit.place_new_job(make_job_entry(&mut registry, JOB_ID, RG_ID, num_tasks)?);
        Ok(Self {
            unit,
            rg_table,
            global_queue: GlobalDispatchQueue::new(),
            id_issuer: TaskAssignmentIdIssuer::new(),
        })
    }

    /// Publishes one assignment for the group, as one turn of the core's decision loop would.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * Forwards [`RgSchedulingUnit::try_make_assignment`]'s return values on failure.
    fn publish(&mut self) -> anyhow::Result<()> {
        let mut jobs_to_retire = Vec::new();
        self.unit.try_make_assignment(
            FREE_SPACE,
            DEFAULT_SESSION_ID,
            &self.id_issuer,
            &self.global_queue,
            &mut jobs_to_retire,
        )?;
        Ok(())
    }

    /// # Returns
    ///
    /// The group's outstanding hint count.
    fn living_hint(&self) -> usize {
        reader_of(&self.rg_table, RG_ID, DEFAULT_SESSION_ID)
            .living_hint()
            .load(Ordering::Acquire)
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for the group.
    fn queue_len(&self) -> usize {
        reader_of(&self.rg_table, RG_ID, DEFAULT_SESSION_ID).len()
    }
}

#[tokio::test]
async fn a_hint_is_published_only_while_the_hint_count_trails_the_queue_size() -> anyhow::Result<()>
{
    let mut fixture = PublisherFixture::new(4)?;

    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 1);
    assert_eq!(fixture.living_hint(), 1);
    assert_eq!(fixture.global_queue.len(), 1);

    // A pinned pop empties the queue without spending the hint, so the hint now covers an
    // assignment that is no longer there.
    let reader = reader_of(&fixture.rg_table, RG_ID, DEFAULT_SESSION_ID);
    assert_eq!(drain_reader(&reader).await.len(), 1);
    assert_eq!(fixture.queue_len(), 0);
    assert_eq!(fixture.living_hint(), 1);

    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 1);
    assert_eq!(fixture.living_hint(), 1);
    assert_eq!(fixture.global_queue.len(), 1);

    // With the hint spent, the next assignment needs a hint of its own again.
    assert!(reader.consume_hint_and_try_recv().is_some());
    assert_eq!(fixture.living_hint(), 0);
    fixture.publish()?;
    assert_eq!(fixture.living_hint(), 1);
    assert_eq!(fixture.global_queue.len(), 2);
    Ok(())
}

#[tokio::test]
async fn a_pinned_pop_leaves_the_hint_count_untouched() -> anyhow::Result<()> {
    let mut fixture = PublisherFixture::new(4)?;
    fixture.publish()?;
    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 2);
    assert_eq!(fixture.living_hint(), 2);

    let reader = reader_of(&fixture.rg_table, RG_ID, DEFAULT_SESSION_ID);
    let assignments = drain_reader(&reader).await;
    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[0].task_id, TaskId::Index(0));
    assert_eq!(assignments[1].task_id, TaskId::Index(1));
    assert_eq!(fixture.queue_len(), 0);
    assert_eq!(fixture.living_hint(), 2);
    assert_eq!(fixture.global_queue.len(), 2);
    Ok(())
}

#[tokio::test]
async fn a_general_pop_consumes_one_hint() -> anyhow::Result<()> {
    let mut fixture = PublisherFixture::new(4)?;
    fixture.publish()?;
    fixture.publish()?;

    let hinted = fixture
        .global_queue
        .recv(DISPATCH_WAIT)
        .await
        .expect("a published hint is receivable");
    assert_eq!(hinted.rg_id(), RG_ID);

    let assignment = hinted
        .consume_hint_and_try_recv()
        .expect("the hinted group holds an assignment");
    assert_eq!(assignment.task_id, TaskId::Index(0));
    assert_eq!(fixture.living_hint(), 1);
    assert_eq!(fixture.queue_len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_stale_hint_on_an_empty_group_consumes_the_hint_and_yields_nothing() -> anyhow::Result<()>
{
    let mut fixture = PublisherFixture::new(4)?;
    fixture.publish()?;

    let reader = reader_of(&fixture.rg_table, RG_ID, DEFAULT_SESSION_ID);
    assert_eq!(drain_reader(&reader).await.len(), 1);
    assert_eq!(fixture.queue_len(), 0);
    assert_eq!(fixture.living_hint(), 1);

    let hinted = fixture
        .global_queue
        .recv(DISPATCH_WAIT)
        .await
        .expect("a published hint is receivable");
    assert_eq!(hinted.consume_hint_and_try_recv(), None);
    assert_eq!(fixture.living_hint(), 0);
    Ok(())
}

#[tokio::test]
async fn next_task_pinned_drops_an_assignment_published_in_a_stale_session() -> anyhow::Result<()> {
    const STALE_SESSION_ID: SessionId = DEFAULT_SESSION_ID;
    const CURRENT_SESSION_ID: SessionId = DEFAULT_SESSION_ID + 1;

    let (service, rg_table, _global_queue) = make_dispatch_service(CURRENT_SESSION_ID);
    let endpoints = rg_table.get_or_create(RG_ID, CURRENT_SESSION_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), STALE_SESSION_ID);
    assert!(endpoints.sender.try_send(assignment).is_ok());

    assert_eq!(service.next_task_pinned(RG_ID, DISPATCH_WAIT).await, None);
    assert_eq!(endpoints.reader.len(), 0);
    Ok(())
}

#[tokio::test]
async fn next_task_general_drops_an_assignment_published_in_a_stale_session() -> anyhow::Result<()>
{
    const STALE_SESSION_ID: SessionId = DEFAULT_SESSION_ID;
    const CURRENT_SESSION_ID: SessionId = DEFAULT_SESSION_ID + 1;

    let (service, rg_table, global_queue) = make_dispatch_service(CURRENT_SESSION_ID);
    let endpoints = rg_table.get_or_create(RG_ID, CURRENT_SESSION_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), STALE_SESSION_ID);
    assert!(endpoints.sender.try_send(assignment).is_ok());
    endpoints
        .reader
        .living_hint()
        .fetch_add(1, Ordering::Release);
    assert!(global_queue.try_send(endpoints.reader.clone()));

    assert_eq!(service.next_task_general(DISPATCH_WAIT).await, None);
    assert_eq!(endpoints.reader.len(), 0);
    assert_eq!(endpoints.reader.living_hint().load(Ordering::Acquire), 0);
    Ok(())
}

#[tokio::test]
async fn next_task_general_discards_a_stale_hint_without_touching_the_hint_count()
-> anyhow::Result<()> {
    const STALE_SESSION_ID: SessionId = DEFAULT_SESSION_ID;
    const CURRENT_SESSION_ID: SessionId = DEFAULT_SESSION_ID + 1;

    let (service, rg_table, global_queue) = make_dispatch_service(CURRENT_SESSION_ID);
    let endpoints = rg_table.get_or_create(RG_ID, STALE_SESSION_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), STALE_SESSION_ID);
    assert!(endpoints.sender.try_send(assignment).is_ok());
    endpoints
        .reader
        .living_hint()
        .fetch_add(1, Ordering::Release);
    assert!(global_queue.try_send(endpoints.reader.clone()));

    assert_eq!(service.next_task_general(DISPATCH_WAIT).await, None);
    assert_eq!(endpoints.reader.living_hint().load(Ordering::Acquire), 1);
    Ok(())
}

/// # Returns
///
/// A tuple containing:
///
/// * A dispatch service serving `session_id`.
/// * The resource group table it dispatches from.
/// * The hint channel it is steered by.
fn make_dispatch_service(
    session_id: SessionId,
) -> (DispatchService, ResourceGroupTable, GlobalDispatchQueue) {
    let rg_table = ResourceGroupTable::new();
    let global_queue = GlobalDispatchQueue::new();
    let service = DispatchService::new(
        rg_table.clone(),
        global_queue.clone(),
        SessionManager::new(session_id),
    );
    (service, rg_table, global_queue)
}
