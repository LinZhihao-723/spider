//! Unit tests for the broadcast queue and the two paths an execution manager pulls assignments
//! through.

use std::time::Duration;

use spider_core::session::SessionTracker;

use super::DEFAULT_SESSION_ID;
use super::drain_reader;
use super::make_assignment;
use super::make_job_entry;
use super::make_unit;
use super::reader_of;
use super::writer_of;
use crate::core::TaskAssignmentIdIssuer;
use crate::dispatch_queue::DispatchOutcome;
use crate::dispatch_queue::DispatchQueueRegistry;
use crate::dispatch_queue::DispatchService;
use crate::job_registry::JobRegistry;
use crate::scheduling_unit::RgSchedulingUnit;
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

/// How long a test that exercises the waited class leaves the queue empty before publishing into
/// it, short enough that the dispatch call is still waiting and long enough that it has stopped
/// polling and started waiting.
const PUBLISH_DELAY: Duration = Duration::from_millis(5);

/// One resource group's publishing side, wired to the structures a test inspects.
struct PublisherFixture {
    unit: RgSchedulingUnit,
    job_registry: JobRegistry,
    dispatch_queue_registry: DispatchQueueRegistry,
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
        let dispatch_queue_registry =
            DispatchQueueRegistry::new(SessionTracker::new(DEFAULT_SESSION_ID));
        let mut unit = make_unit(&dispatch_queue_registry, RG_ID, 1);
        let mut job_registry = JobRegistry::new();
        unit.place_new_job(make_job_entry(&mut job_registry, JOB_ID, num_tasks)?);
        Ok(Self {
            unit,
            job_registry,
            dispatch_queue_registry,
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
            &mut self.job_registry,
            &mut jobs_to_retire,
        )?;
        Ok(())
    }

    /// # Returns
    ///
    /// The number of assignments currently queued for the group.
    fn queue_len(&self) -> usize {
        writer_of(&self.dispatch_queue_registry, RG_ID).queue_len()
    }

    /// # Returns
    ///
    /// The number of hints waiting in the broadcast queue.
    fn num_outstanding_hints(&self) -> usize {
        self.dispatch_queue_registry.num_outstanding_hints()
    }
}

#[tokio::test]
async fn a_hint_is_published_only_while_the_hint_count_trails_the_queue_size() -> anyhow::Result<()>
{
    let mut fixture = PublisherFixture::new(4)?;

    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 1);
    assert_eq!(fixture.num_outstanding_hints(), 1);

    // A pinned pop empties the queue without spending the hint, so the hint now covers an
    // assignment that is no longer there.
    let reader = reader_of(&fixture.dispatch_queue_registry, RG_ID);
    assert_eq!(drain_reader(&reader).await.len(), 1);
    assert_eq!(fixture.queue_len(), 0);

    // The outstanding hint still covers the refilled queue, so this assignment publishes none of
    // its own and the broadcast queue stays as long as it was.
    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 1);
    assert_eq!(fixture.num_outstanding_hints(), 1);

    // With the hint spent -- along with the assignment it finally found -- the next assignment
    // needs a hint of its own again.
    let hint = fixture
        .dispatch_queue_registry
        .try_next_hint()
        .expect("a published hint is receivable");
    assert!(hint.consume_and_try_recv().is_some());
    assert_eq!(fixture.queue_len(), 0);
    assert_eq!(fixture.num_outstanding_hints(), 0);
    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 1);
    assert_eq!(fixture.num_outstanding_hints(), 1);
    Ok(())
}

#[tokio::test]
async fn a_pinned_pop_leaves_the_hint_count_untouched() -> anyhow::Result<()> {
    let mut fixture = PublisherFixture::new(6)?;
    fixture.publish()?;
    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 2);
    assert_eq!(fixture.num_outstanding_hints(), 2);

    let reader = reader_of(&fixture.dispatch_queue_registry, RG_ID);
    let assignments = drain_reader(&reader).await;
    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[0].task_id, TaskId::Index(0));
    assert_eq!(assignments[1].task_id, TaskId::Index(1));
    assert_eq!(fixture.queue_len(), 0);
    assert_eq!(fixture.num_outstanding_hints(), 2);

    // Both hints survived the pops, and refilling the queue counts them back out: the first two
    // assignments are still covered, and only the third -- the one that puts the queue ahead of
    // the count -- publishes a hint.
    fixture.publish()?;
    assert_eq!(fixture.num_outstanding_hints(), 2);
    fixture.publish()?;
    assert_eq!(fixture.num_outstanding_hints(), 2);
    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 3);
    assert_eq!(fixture.num_outstanding_hints(), 3);
    Ok(())
}

#[tokio::test]
async fn a_general_pop_consumes_one_hint() -> anyhow::Result<()> {
    let mut fixture = PublisherFixture::new(4)?;
    fixture.publish()?;
    fixture.publish()?;

    let hint = fixture
        .dispatch_queue_registry
        .next_hint(DISPATCH_WAIT)
        .await
        .expect("a published hint is receivable");
    assert_eq!(hint.rg_id(), RG_ID);

    let assignment = hint
        .consume_and_try_recv()
        .expect("the hinted group holds an assignment");
    assert_eq!(assignment.task_id, TaskId::Index(0));
    assert_eq!(fixture.queue_len(), 1);

    // One of the two hints is gone, so the count no longer covers the refilled queue and the next
    // assignment publishes a hint -- which it would not have done had the pop left the count at
    // two.
    fixture.publish()?;
    assert_eq!(fixture.queue_len(), 2);
    assert_eq!(fixture.num_outstanding_hints(), 2);
    Ok(())
}

#[tokio::test]
async fn a_stale_hint_on_an_empty_group_consumes_the_hint_and_yields_nothing() -> anyhow::Result<()>
{
    let mut fixture = PublisherFixture::new(4)?;
    fixture.publish()?;

    let reader = reader_of(&fixture.dispatch_queue_registry, RG_ID);
    assert_eq!(drain_reader(&reader).await.len(), 1);
    assert_eq!(fixture.queue_len(), 0);

    let hint = fixture
        .dispatch_queue_registry
        .next_hint(DISPATCH_WAIT)
        .await
        .expect("a published hint is receivable");
    assert_eq!(hint.consume_and_try_recv(), None);
    assert_eq!(fixture.num_outstanding_hints(), 0);

    // The stale hint was spent even though it yielded nothing, so the group is uncovered again and
    // the next assignment publishes a fresh hint.
    fixture.publish()?;
    assert_eq!(fixture.num_outstanding_hints(), 1);
    Ok(())
}

#[tokio::test]
async fn clearing_the_registry_discards_the_hints_of_the_session_left_behind() -> anyhow::Result<()>
{
    let mut fixture = PublisherFixture::new(4)?;
    fixture.publish()?;
    fixture.publish()?;
    assert_eq!(fixture.num_outstanding_hints(), 2);

    fixture.dispatch_queue_registry.clear();
    assert_eq!(fixture.dispatch_queue_registry.len(), 0);
    assert_eq!(fixture.num_outstanding_hints(), 0);

    // The queue the discarded hints named is gone with them, so a general execution manager is
    // steered by nothing the flushed session published.
    let service = DispatchService::new(fixture.dispatch_queue_registry.clone());
    assert_eq!(service.next_task_general(DISPATCH_WAIT).await, None);
    Ok(())
}

#[tokio::test]
async fn next_task_pinned_drops_an_assignment_published_in_a_stale_session() -> anyhow::Result<()> {
    const STALE_SESSION_ID: SessionId = DEFAULT_SESSION_ID;
    const CURRENT_SESSION_ID: SessionId = DEFAULT_SESSION_ID + 1;

    let (service, registry) = make_dispatch_service(SessionTracker::new(CURRENT_SESSION_ID));
    let writer = writer_of(&registry, RG_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), STALE_SESSION_ID);
    assert!(writer.try_send(assignment).is_ok());

    assert_eq!(service.next_task_pinned(RG_ID, DISPATCH_WAIT).await, None);
    assert_eq!(writer.queue_len(), 0);
    Ok(())
}

#[tokio::test]
async fn next_task_general_drops_an_assignment_published_in_a_stale_session() -> anyhow::Result<()>
{
    const STALE_SESSION_ID: SessionId = DEFAULT_SESSION_ID;
    const CURRENT_SESSION_ID: SessionId = DEFAULT_SESSION_ID + 1;

    let (service, registry) = make_dispatch_service(SessionTracker::new(CURRENT_SESSION_ID));
    let writer = writer_of(&registry, RG_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), STALE_SESSION_ID);
    assert!(writer.try_send(assignment).is_ok());
    assert_eq!(registry.num_outstanding_hints(), 1);

    assert_eq!(service.next_task_general(DISPATCH_WAIT).await, None);
    assert_eq!(writer.queue_len(), 0);

    // Dropping the assignment spent the hint that led to it, so a fresh assignment makes a hint of
    // its own again.
    let fresh = make_assignment(RG_ID, JOB_ID, TaskId::Index(1), CURRENT_SESSION_ID);
    assert!(writer.try_send(fresh).is_ok());
    assert_eq!(registry.num_outstanding_hints(), 1);
    Ok(())
}

#[tokio::test]
async fn next_task_general_discards_a_stale_hint_without_touching_the_hint_count()
-> anyhow::Result<()> {
    const STALE_SESSION_ID: SessionId = DEFAULT_SESSION_ID;
    const CURRENT_SESSION_ID: SessionId = DEFAULT_SESSION_ID + 1;

    let session_tracker = SessionTracker::new(STALE_SESSION_ID);
    let (service, registry) = make_dispatch_service(session_tracker.clone());
    let writer = writer_of(&registry, RG_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), STALE_SESSION_ID);
    assert!(writer.try_send(assignment).is_ok());

    // The group and the hint covering its assignment both belong to the session left behind here.
    assert!(session_tracker.try_advance(CURRENT_SESSION_ID));
    assert_eq!(service.next_task_general(DISPATCH_WAIT).await, None);

    // The hint was discarded without a decrement, so it still covers the assignment left in the
    // queue.
    assert_eq!(writer.queue_len(), 1);
    assert_eq!(reader_of(&registry, RG_ID).living_hint(), 1);
    Ok(())
}

#[tokio::test]
async fn next_task_pinned_classifies_an_already_queued_assignment_as_immediate()
-> anyhow::Result<()> {
    let (service, registry) = make_dispatch_service(SessionTracker::new(DEFAULT_SESSION_ID));
    let writer = writer_of(&registry, RG_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), DEFAULT_SESSION_ID);
    assert!(writer.try_send(assignment).is_ok());

    assert_eq!(
        service
            .next_task_pinned_classified(RG_ID, DISPATCH_WAIT)
            .await,
        DispatchOutcome::Immediate(assignment)
    );
    Ok(())
}

#[tokio::test]
async fn next_task_pinned_classifies_an_assignment_published_after_the_request_as_waited()
-> anyhow::Result<()> {
    let (service, registry) = make_dispatch_service(SessionTracker::new(DEFAULT_SESSION_ID));
    let writer = writer_of(&registry, RG_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), DEFAULT_SESSION_ID);
    let publisher = tokio::spawn(async move {
        tokio::time::sleep(PUBLISH_DELAY).await;
        writer.try_send(assignment).is_ok()
    });

    let outcome = service
        .next_task_pinned_classified(RG_ID, DISPATCH_WAIT)
        .await;

    assert!(publisher.await.expect("the publishing task finishes"));
    assert_eq!(outcome, DispatchOutcome::Waited(assignment));
    Ok(())
}

#[tokio::test]
async fn next_task_general_classifies_an_already_hinted_assignment_as_immediate()
-> anyhow::Result<()> {
    let (service, registry) = make_dispatch_service(SessionTracker::new(DEFAULT_SESSION_ID));
    let writer = writer_of(&registry, RG_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), DEFAULT_SESSION_ID);
    assert!(writer.try_send(assignment).is_ok());

    assert_eq!(
        service.next_task_general_classified(DISPATCH_WAIT).await,
        DispatchOutcome::Immediate(assignment)
    );
    Ok(())
}

#[tokio::test]
async fn next_task_general_classifies_a_hint_published_after_the_request_as_waited()
-> anyhow::Result<()> {
    let (service, registry) = make_dispatch_service(SessionTracker::new(DEFAULT_SESSION_ID));
    let writer = writer_of(&registry, RG_ID);
    let assignment = make_assignment(RG_ID, JOB_ID, TaskId::Index(0), DEFAULT_SESSION_ID);
    let publisher = tokio::spawn(async move {
        tokio::time::sleep(PUBLISH_DELAY).await;
        writer.try_send(assignment).is_ok()
    });

    let outcome = service.next_task_general_classified(DISPATCH_WAIT).await;

    assert!(publisher.await.expect("the publishing task finishes"));
    assert_eq!(outcome, DispatchOutcome::Waited(assignment));
    Ok(())
}

/// # Returns
///
/// A tuple containing:
///
/// * A dispatch service serving the session `session_tracker` holds.
/// * The registry it dispatches from.
fn make_dispatch_service(
    session_tracker: SessionTracker,
) -> (DispatchService, DispatchQueueRegistry) {
    let registry = DispatchQueueRegistry::new(session_tracker);
    let service = DispatchService::new(registry.clone());
    (service, registry)
}
