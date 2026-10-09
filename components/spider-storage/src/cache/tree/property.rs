//! Property tests of the planner and the repair planner over random trees.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::seq::SliceRandom;
use spider_core::job::JobState;
use spider_core::types::id::JobId;
use spider_core::types::id::ResourceGroupId;

use super::*;

const MAX_MODEL_DEPTH: u32 = 5;
const MAX_FAN_OUT: usize = 4;
const MAX_ROUNDS_PER_TREE: usize = 400;
const LARGE_OUTPUT_SIZE: usize = 6 * 1024 * 1024;
const REGISTRATION_PROBABILITIES: [f64; 3] = [0.3, 0.6, 0.95];
const FAILURE_PROBABILITIES: [f64; 2] = [0.15, 0.01];
const CANCEL_PROBABILITIES: [f64; 2] = [0.02, 0.001];
const BURST_PROBABILITY: f64 = 0.005;
const ALL_STATES: [JobState; 8] = [
    JobState::Ready,
    JobState::Running,
    JobState::BlockingOnChildren,
    JobState::CommitReady,
    JobState::CleanupReady,
    JobState::Succeeded,
    JobState::Failed,
    JobState::Cancelled,
];

/// Returns the formatted message as an error unless `condition` holds.
macro_rules! ensure {
    ($condition:expr, $($message:tt)+) => {
        if !$condition {
            return Err(format!($($message)+));
        }
    };
}

/// How the model raises the events of its jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Raising {
    /// Every round, each job may raise an event for its state again.
    Repeated,

    /// Each job raises the event of each of its states once, as the real producers do.
    Once,
}

/// Which events arrive in a round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arrivals {
    /// The events that the rows raise, and copies of earlier events.
    All,

    /// Only the events that the rows raise.
    Raised,

    /// No event.
    Closed,
}

/// A job row of the in-memory database model.
#[derive(Debug, Clone)]
struct Row {
    parent: Option<JobId>,
    depth: u32,
    state: JobState,
    has_commit: bool,
    has_cleanup: bool,
    outputs: Option<Vec<TaskOutput>>,
    error: Option<String>,
}

/// An in-memory model of one tree's rows, plus the structure the committer would keep for it.
struct Model {
    rows: BTreeMap<u64, Row>,
    structure: TreeStructure,
    next_id: u64,
    registration_probability: f64,
    failure_probability: f64,
    cancel_probability: f64,

    /// Whether the terminal rows are exempt from the ID order of I1 and from "success is earned",
    /// because they were corrupted before a repair, which cannot change terminal rows.
    lenient_terminal_rows: bool,
}

impl Model {
    fn new(rng: &mut StdRng) -> Self {
        let root = Row {
            parent: None,
            depth: 0,
            state: JobState::Ready,
            has_commit: rng.random_bool(0.5),
            has_cleanup: rng.random_bool(0.5),
            outputs: None,
            error: None,
        };
        let structure = TreeStructure::new_root(JobId::from(1), Self::node_of(&root));
        Self {
            rows: BTreeMap::from([(1, root)]),
            structure,
            next_id: 2,
            registration_probability: *REGISTRATION_PROBABILITIES
                .choose(rng)
                .expect("there are probabilities"),
            failure_probability: *FAILURE_PROBABILITIES
                .choose(rng)
                .expect("there are probabilities"),
            cancel_probability: *CANCEL_PROBABILITIES
                .choose(rng)
                .expect("there are probabilities"),
            lenient_terminal_rows: false,
        }
    }

    /// Factory function.
    ///
    /// Builds the model of a repaired tree from its rows and the structure that recovery rebuilt
    /// from them. The terminal rows are checked leniently if they break an invariant.
    fn from_repaired(
        rows: &BTreeMap<u64, RepairRow>,
        structure: TreeStructure,
        rng: &mut StdRng,
    ) -> Self {
        let rows: BTreeMap<u64, Row> = rows
            .iter()
            .map(|(id, row)| {
                let depth = structure
                    .node(JobId::from(*id))
                    .map_or(0, |node| node.depth);
                let model_row = Row {
                    parent: row.parent.map(JobId::from),
                    depth,
                    state: row.state,
                    has_commit: row.has_commit,
                    has_cleanup: row.has_cleanup,
                    outputs: None,
                    error: None,
                };
                (*id, model_row)
            })
            .collect();
        let next_id = rows.keys().max().copied().unwrap_or_default() + 1;
        let mut model = Self {
            rows,
            structure,
            next_id,
            registration_probability: *REGISTRATION_PROBABILITIES
                .choose(rng)
                .expect("there are probabilities"),
            failure_probability: *FAILURE_PROBABILITIES
                .choose(rng)
                .expect("there are probabilities"),
            cancel_probability: *CANCEL_PROBABILITIES
                .choose(rng)
                .expect("there are probabilities"),
            lenient_terminal_rows: false,
        };
        model.lenient_terminal_rows = model.violation(true).is_err();
        model
    }

    fn node_of(row: &Row) -> TreeNode {
        TreeNode {
            parent: row.parent,
            depth: row.depth,
            resource_group_id: ResourceGroupId::from(1),
            has_commit: row.has_commit,
            has_cleanup: row.has_cleanup,
            active_children: 0,
        }
    }

    fn row(&self, job_id: JobId) -> &Row {
        self.rows.get(&job_id.get()).expect("the row should exist")
    }

    /// # Returns
    ///
    /// The ancestors of `job_id` in the rows, nearest first.
    fn ancestors(&self, job_id: JobId) -> Vec<JobId> {
        std::iter::successors(self.row(job_id).parent, |ancestor_id| {
            self.row(*ancestor_id).parent
        })
        .collect()
    }

    /// # Returns
    ///
    /// The events that the current rows raise this round, each sometimes followed by a failure of
    /// another task instance of its job and lane. With [`Raising::Once`], a job raises nothing
    /// while it is in the state that `raised` records for it, and `raised` records the state of
    /// each job that raises. Job states never repeat, so each state raises once.
    fn raise_events(
        &self,
        raising: Raising,
        raised: &mut HashMap<u64, JobState>,
        rng: &mut StdRng,
    ) -> Vec<TreeEvent> {
        let mut events = Vec::new();
        for (id, row) in &self.rows {
            if row.state.is_terminal()
                || row.state == JobState::BlockingOnChildren
                || !rng.random_bool(0.6)
                || (raising == Raising::Once && raised.insert(*id, row.state) == Some(row.state))
            {
                continue;
            }
            let job_id = JobId::from(*id);
            let kind = match row.state {
                JobState::Ready => TreeEventKind::Start,
                JobState::Running if !rng.random_bool(self.failure_probability) => {
                    TreeEventKind::RegularTasksDone {
                        registers: row.depth + 1 < MAX_MODEL_DEPTH
                            && rng.random_bool(self.registration_probability),
                    }
                }
                JobState::Running => TreeEventKind::TerminalFailure {
                    lane: Lane::Regular,
                    message: format!("job {id} failed"),
                },
                JobState::CommitReady if !rng.random_bool(self.failure_probability) => {
                    TreeEventKind::CommitDone
                }
                JobState::CommitReady => TreeEventKind::TerminalFailure {
                    lane: Lane::Commit,
                    message: format!("commit of {id} failed"),
                },
                JobState::CleanupReady if !rng.random_bool(self.failure_probability) => {
                    TreeEventKind::CleanupDone
                }
                _ => TreeEventKind::TerminalFailure {
                    lane: Lane::Cleanup,
                    message: format!("cleanup of {id} failed"),
                },
            };
            events.push(TreeEvent { job_id, kind });
            let lane = match row.state {
                JobState::Running => Some(Lane::Regular),
                JobState::CommitReady => Some(Lane::Commit),
                JobState::CleanupReady => Some(Lane::Cleanup),
                _ => None,
            };
            if let Some(lane) = lane
                && rng.random_bool(self.failure_probability)
            {
                events.push(TreeEvent {
                    job_id,
                    kind: TreeEventKind::TerminalFailure {
                        lane,
                        message: format!("another task instance of job {id} failed"),
                    },
                });
            }
        }
        if rng.random_bool(self.cancel_probability) {
            events.push(TreeEvent {
                job_id: JobId::from(1),
                kind: TreeEventKind::CancelRoot,
            });
        }
        events
    }

    /// # Returns
    ///
    /// The input the committer would collect for `event` from the current rows.
    fn collect_input(&self, event: &TreeEvent, rng: &mut StdRng) -> PlannerInput {
        if self.structure.node(event.job_id).is_none() {
            return PlannerInput::Empty;
        }
        let state = self.row(event.job_id).state;
        match (&event.kind, state) {
            (TreeEventKind::Start, JobState::Ready) => PlannerInput::Start {
                ready_task_indices: vec![0],
            },
            (TreeEventKind::RegularTasksDone { registers }, JobState::Running) => {
                let children = registers.then(|| {
                    let roll: f64 = rng.random();
                    if roll < 0.05 {
                        ChildSetSummary {
                            count: 2,
                            bytes: MAX_JOB_CHILD_BYTES + 1,
                        }
                    } else {
                        let bytes = if roll < 0.1 {
                            MAX_JOB_CHILD_BYTES
                        } else if roll < 0.2 {
                            rng.random_range(MAX_JOB_CHILD_BYTES / 2..MAX_JOB_CHILD_BYTES)
                        } else {
                            rng.random_range(1..4096)
                        };
                        ChildSetSummary {
                            count: rng.random_range(1..=MAX_FAN_OUT),
                            bytes,
                        }
                    }
                });
                let output_size = if rng.random_bool(0.05) {
                    LARGE_OUTPUT_SIZE
                } else {
                    1
                };
                let byte = u8::try_from(event.job_id.get() % 256).expect("fits");
                PlannerInput::RegularTasksDone {
                    outputs: vec![vec![byte; output_size]],
                    children,
                }
            }
            _ => PlannerInput::Empty,
        }
    }

    /// # Returns
    ///
    /// The states the committer would collect for `events`.
    fn collect_states(&self, events: &[&TreeEvent]) -> HashMap<JobId, JobState> {
        self.structure
            .jobs_to_collect(events.iter().copied())
            .into_iter()
            .map(|job_id| (job_id, self.row(job_id).state))
            .collect()
    }

    fn structure_states(&self) -> HashMap<JobId, JobState> {
        self.structure
            .job_ids()
            .map(|job_id| (job_id, self.row(job_id).state))
            .collect()
    }

    /// Writes `plan` into the rows and the structure, checking every guard.
    fn apply(&mut self, plan: &Plan, rng: &mut StdRng) {
        let mut seen = HashSet::new();
        for update in &plan.updates {
            assert!(
                seen.insert(update.job_id),
                "job {} is updated twice in one round",
                update.job_id
            );
            let row = self
                .rows
                .get_mut(&update.job_id.get())
                .expect("an updated row should exist");
            assert_eq!(
                row.state, update.from,
                "guard miss on job {}",
                update.job_id
            );
            assert!(
                JobState::is_valid_transition(update.from, update.to),
                "invalid edge {} -> {} on job {}",
                update.from,
                update.to,
                update.job_id
            );
            row.state = update.to;
            if update.outputs.is_some() {
                row.outputs.clone_from(&update.outputs);
            }
            if update.error_message.is_some() {
                row.error.clone_from(&update.error_message);
            }
        }
        for (job_id, active_children) in &plan.counters {
            self.structure
                .set_active_children(*job_id, *active_children);
        }
        if let Some(registration) = plan.registration {
            for _ in 0..registration.children.count {
                let row = Row {
                    parent: Some(registration.parent_id),
                    depth: registration.child_depth,
                    state: JobState::Running,
                    has_commit: rng.random_bool(0.4),
                    has_cleanup: rng.random_bool(0.4),
                    outputs: None,
                    error: None,
                };
                self.structure
                    .insert(JobId::from(self.next_id), Self::node_of(&row));
                self.rows.insert(self.next_id, row);
                self.next_id += 1;
            }
        }
        for job_id in &plan.terminated {
            self.structure.remove(*job_id);
        }
    }

    /// Checks I1 to I5 on the rows. With `sees_terminal_rows`, the children of a blocking job
    /// include its terminal ones, and "success is earned" is checked too. Without, the check
    /// sees what the structure sees, as [`check_invariants`] does.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * The first broken invariant found, as text.
    fn violation(&self, sees_terminal_rows: bool) -> Result<(), String> {
        self.check_children(sees_terminal_rows)?;
        self.check_rows()
    }

    /// Checks I2, and "success is earned" if `sees_terminal_rows` and the terminal rows are not
    /// checked leniently.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * The first broken invariant found, as text.
    fn check_children(&self, sees_terminal_rows: bool) -> Result<(), String> {
        let mut children: HashMap<u64, Vec<&Row>> = HashMap::new();
        for row in self.rows.values() {
            if let Some(parent_id) = row.parent
                && (sees_terminal_rows || !row.state.is_terminal())
            {
                children.entry(parent_id.get()).or_default().push(row);
            }
        }
        let is_active = |state: JobState| {
            matches!(
                state,
                JobState::Running | JobState::BlockingOnChildren | JobState::CommitReady
            )
        };
        for (id, row) in &self.rows {
            let child_states: Vec<JobState> = children
                .get(id)
                .into_iter()
                .flatten()
                .map(|child| child.state)
                .collect();
            if row.state == JobState::BlockingOnChildren {
                ensure!(
                    child_states
                        .iter()
                        .all(|state| *state == JobState::Succeeded || is_active(*state)),
                    "I2: children of blocking job {id} are {child_states:?}"
                );
                let num_active = child_states
                    .iter()
                    .filter(|state| is_active(**state))
                    .count();
                ensure!(num_active > 0, "I2: blocking job {id} has no active child");
                let Some(node) = self.structure.node(JobId::from(*id)) else {
                    return Err(format!("I5: blocking job {id} is not in the tree"));
                };
                ensure!(
                    node.active_children as usize == num_active,
                    "I2: counter of job {id} is {}, not {num_active}",
                    node.active_children
                );
            }
            if sees_terminal_rows
                && !self.lenient_terminal_rows
                && matches!(row.state, JobState::CommitReady | JobState::Succeeded)
            {
                ensure!(
                    child_states
                        .iter()
                        .all(|state| *state == JobState::Succeeded),
                    "success of job {id} is not earned: children are {child_states:?}"
                );
            }
        }
        Ok(())
    }

    /// Checks I1, I3, I4 and I5.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * The first broken invariant found, as text.
    fn check_rows(&self) -> Result<(), String> {
        let root = self.row(JobId::from(1));
        for (id, row) in &self.rows {
            ensure!(
                (*id == 1) == row.parent.is_none(),
                "I1: job {id} parent link"
            );
            ensure!(
                row.state != JobState::Ready || *id == 1,
                "I1: non-root job {id} is Ready"
            );
            if let Some(parent_id) = row.parent
                && !(self.lenient_terminal_rows && row.state.is_terminal())
            {
                ensure!(parent_id.get() < *id, "I1: job {id} id order");
            }
            if matches!(row.state, JobState::Running | JobState::BlockingOnChildren) {
                let mut ancestor = row.parent;
                while let Some(ancestor_id) = ancestor {
                    let ancestor_row = self.row(ancestor_id);
                    ensure!(
                        ancestor_row.state == JobState::BlockingOnChildren,
                        "I3: ancestor {ancestor_id} of active job {id} is {}",
                        ancestor_row.state
                    );
                    ancestor = ancestor_row.parent;
                }
            }
            if row.state == JobState::CommitReady
                && let Some(parent_id) = row.parent
            {
                let parent_state = self.row(parent_id).state;
                ensure!(
                    parent_state == JobState::BlockingOnChildren
                        || parent_state == JobState::CleanupReady
                        || parent_state.is_terminal(),
                    "I3: CommitReady job {id} under {parent_state} parent"
                );
            }
            if root.state == JobState::CleanupReady || root.state.is_terminal() {
                ensure!(
                    !matches!(
                        row.state,
                        JobState::Ready | JobState::Running | JobState::BlockingOnChildren
                    ),
                    "I4: job {id} is {} in a closed tree",
                    row.state
                );
            }
            let node = self.structure.node(JobId::from(*id));
            ensure!(
                node.is_some() != row.state.is_terminal(),
                "I5: job {id} ({}) in the structure",
                row.state
            );
            if let Some(node) = node {
                ensure!(
                    (node.parent, node.depth, node.has_commit, node.has_cleanup)
                        == (row.parent, row.depth, row.has_commit, row.has_cleanup),
                    "I5: node of job {id}"
                );
            }
        }
        Ok(())
    }

    /// Checks every invariant on the rows, checks that [`check_invariants`] agrees, then
    /// perturbs one job state or counter of the structure and checks that
    /// [`check_invariants`] finds a violation exactly when the model does on what the
    /// structure sees. The perturbation is undone afterwards.
    ///
    /// # Panics
    ///
    /// Panics if a check fails.
    fn check(&mut self, rng: &mut StdRng, coverage: &mut Coverage) {
        if let Err(violation) = self.violation(true) {
            panic!("the model breaks an invariant: {violation}");
        }
        check_invariants(&self.structure, &self.structure_states())
            .expect("check_invariants should agree with the model");

        let Some(job_id) = self.structure.sorted_job_ids().choose(rng).copied() else {
            return;
        };
        if rng.random_bool(0.5) {
            let row = self
                .rows
                .get_mut(&job_id.get())
                .expect("the row should exist");
            let original = row.state;
            row.state = *ALL_STATES.choose(rng).expect("there are states");
            self.compare_verdicts(coverage);
            self.rows
                .get_mut(&job_id.get())
                .expect("the row should exist")
                .state = original;
        } else {
            let original = self
                .structure
                .node(job_id)
                .expect("a listed job should be in the tree")
                .active_children;
            let max_counter = u32::try_from(MAX_FAN_OUT).expect("fits") + 1;
            self.structure
                .set_active_children(job_id, rng.random_range(0..=max_counter));
            self.compare_verdicts(coverage);
            self.structure.set_active_children(job_id, original);
        }
    }

    /// Checks that [`check_invariants`] and the model, on what the structure sees, agree on
    /// whether the tree breaks an invariant.
    ///
    /// # Panics
    ///
    /// Panics if they disagree.
    fn compare_verdicts(&self, coverage: &mut Coverage) {
        let model_verdict = self.violation(false);
        let checker_verdict = check_invariants(&self.structure, &self.structure_states());
        assert_eq!(
            model_verdict.is_err(),
            checker_verdict.is_err(),
            "the model says {model_verdict:?}, check_invariants says {checker_verdict:?}"
        );
        if model_verdict.is_err() {
            coverage.broken_perturbations += 1;
        } else {
            coverage.harmless_perturbations += 1;
        }
    }

    /// # Returns
    ///
    /// The state that `event`, which acts on its job's `state`, moves the job to, given the
    /// event's collected `input`, per the design's event table.
    fn expected_applied_state(
        &self,
        event: &TreeEvent,
        state: JobState,
        input: &PlannerInput,
    ) -> JobState {
        let row = self.row(event.job_id);
        match &event.kind {
            TreeEventKind::Start => JobState::Running,
            TreeEventKind::RegularTasksDone { .. } => {
                let PlannerInput::RegularTasksDone { children, .. } = input else {
                    panic!("a running job's regular-tasks-done event should carry its outputs");
                };
                match children {
                    Some(children)
                        if children.count > MAX_CHILDREN_PER_JOB
                            || children.bytes > MAX_JOB_CHILD_BYTES =>
                    {
                        JobState::Failed
                    }
                    Some(children) if children.count > 0 => JobState::BlockingOnChildren,
                    _ if row.has_commit => JobState::CommitReady,
                    _ => JobState::Succeeded,
                }
            }
            TreeEventKind::CommitDone => JobState::Succeeded,
            TreeEventKind::CleanupDone => JobState::Cancelled,
            TreeEventKind::TerminalFailure { .. } => JobState::Failed,
            TreeEventKind::CancelRoot if state == JobState::Ready || !row.has_cleanup => {
                JobState::Cancelled
            }
            TreeEventKind::CancelRoot => JobState::CleanupReady,
        }
    }

    /// Checks every planned outcome of a round exactly against the design's event table, as
    /// [`Self::check_outcome`] does, and the round's registration, as
    /// [`Self::check_registration`] does.
    ///
    /// # Returns
    ///
    /// The failures that the round's applied events planned, to check after the apply.
    ///
    /// # Panics
    ///
    /// Panics if a check fails.
    fn check_outcomes(
        &self,
        collected: &HashMap<JobId, JobState>,
        events: &[TreeEvent],
        inputs: &[PlannerInput],
        plan: &Plan,
    ) -> Vec<ExpectedFailure> {
        let order = planning_order(events, plan);
        let failures = (1..=order.len())
            .filter_map(|num_planned| {
                self.check_outcome(collected, events, inputs, plan, &order[..num_planned])
            })
            .collect();
        if let Some(registration) = &plan.registration {
            self.check_registration(registration, inputs);
        }
        failures
    }

    /// Checks the outcome of the last event of `planned`, the events of a round in the order that
    /// the planner planned them. The event is replanned with only the events that the planner
    /// planned before it, which gives the state that it found its job in. Its outcome must be
    /// [`TreeOutcome::Untracked`] if its job left the tree, else applied with the state the table
    /// gives if the event acts on the state it found, else stale with the error the table gives.
    /// An applied event changes its job from the state the job had at the start of the round.
    ///
    /// # Returns
    ///
    /// The failure that the event planned, if it applied a failure.
    ///
    /// # Panics
    ///
    /// Panics if a check fails.
    fn check_outcome(
        &self,
        collected: &HashMap<JobId, JobState>,
        events: &[TreeEvent],
        inputs: &[PlannerInput],
        plan: &Plan,
        planned: &[usize],
    ) -> Option<ExpectedFailure> {
        let (index, earlier) = planned.split_last().expect("at least one event is planned");
        let event = &events[*index];
        let outcome = &plan.outcomes[index];
        let job_id = event.job_id;
        if self.structure.node(job_id).is_none() {
            assert!(
                matches!(outcome, TreeOutcome::Untracked),
                "event {index} of a job that left the tree is answered {outcome:?}"
            );
            return None;
        }

        let mut earlier = earlier.to_vec();
        earlier.sort_unstable();
        let mut earlier_inputs: Vec<PlannerInput> =
            earlier.iter().map(|index| inputs[*index].clone()).collect();
        let mut earlier_events: Vec<PlannerEvent<'_>> = earlier
            .iter()
            .zip(earlier_inputs.iter_mut())
            .map(|(index, input)| PlannerEvent {
                event: &events[*index],
                input,
            })
            .collect();
        let earlier_plan = plan_events(&self.structure, collected, &mut earlier_events);
        assert_eq!(
            earlier_plan.deferred,
            Vec::<usize>::new(),
            "the events planned before event {index} are planned again without it"
        );
        for (earlier_index, index_in_round) in earlier.iter().enumerate() {
            assert_eq!(
                format!("{:?}", earlier_plan.outcomes[&earlier_index]),
                format!("{:?}", plan.outcomes[index_in_round]),
                "event {index_in_round} is planned the same without the later events"
            );
        }
        let found = |job_id: JobId| {
            earlier_plan
                .updates
                .iter()
                .find(|update| update.job_id == job_id)
                .map_or_else(|| self.row(job_id).state, |update| update.to)
        };
        let found_state = found(job_id);

        let expected = if acts_on(&event.kind, found_state) {
            TreeOutcome::Applied(self.expected_applied_state(event, found_state, &inputs[*index]))
        } else {
            TreeOutcome::Stale(expected_stale_error(&event.kind, found_state))
        };
        assert_eq!(
            format!("{outcome:?}"),
            format!("{expected:?}"),
            "outcome of event {index} ({:?} of job {job_id}, found {found_state})",
            event.kind
        );
        let TreeOutcome::Applied(state) = outcome else {
            return None;
        };
        let update = plan
            .updates
            .iter()
            .find(|update| update.job_id == job_id)
            .expect("an applied event updates its job");
        assert_eq!(
            (update.from, update.to),
            (self.row(job_id).state, *state),
            "update of applied event {index}"
        );
        if *state != JobState::Failed {
            return None;
        }
        let (message, lane) = match &event.kind {
            TreeEventKind::TerminalFailure { lane, message } => (message.clone(), *lane),
            TreeEventKind::RegularTasksDone { .. } => {
                ("child-job limit exceeded".to_owned(), Lane::Regular)
            }
            kind => panic!("a {kind:?} event cannot fail its job"),
        };
        let propagates = lane != Lane::Cleanup
            && self
                .row(job_id)
                .parent
                .is_some_and(|parent_id| found(parent_id) == JobState::BlockingOnChildren);
        Some(ExpectedFailure {
            job_id,
            message,
            chain: if propagates {
                self.ancestors(job_id)
            } else {
                Vec::new()
            },
        })
    }

    /// Checks that `registration` registers the children of its event's input, within the limits,
    /// one level below its parent, which has no child yet.
    ///
    /// # Panics
    ///
    /// Panics if a check fails.
    fn check_registration(&self, registration: &PlannedRegistration, inputs: &[PlannerInput]) {
        let PlannerInput::RegularTasksDone {
            children: Some(children),
            ..
        } = &inputs[registration.event_index]
        else {
            panic!("a registration's event should carry its children");
        };
        let parent = self.row(registration.parent_id);
        assert_eq!(
            (
                registration.children,
                registration.child_depth,
                registration.root_id
            ),
            (*children, parent.depth + 1, JobId::from(1)),
            "registration of job {}",
            registration.parent_id
        );
        assert!(
            children.count <= MAX_CHILDREN_PER_JOB && children.bytes <= MAX_JOB_CHILD_BYTES,
            "a registration within the limits"
        );
        assert!(
            self.rows
                .values()
                .all(|row| row.parent != Some(registration.parent_id)),
            "job {} registers children twice",
            registration.parent_id
        );
    }
}

/// How often the random trees reached the cases the planner must handle.
#[derive(Debug, Default)]
struct Coverage {
    max_rows: usize,
    grouped_rounds: usize,
    deferred_events: usize,
    registrations: usize,
    absent_events: usize,
    applied_cancels: usize,
    propagated_failures: usize,
    limit_failures: usize,
    skipped_commits: usize,
    unblocked_parents: usize,
    crowded_cancels: usize,
    budget_deferrals: usize,
    reserved_registrations: usize,
    full_rounds: usize,
    broken_perturbations: usize,
    harmless_perturbations: usize,
    unfinished_trees: usize,
    backlogged_trees: usize,
    outcomes: BTreeMap<(&'static str, &'static str), usize>,
}

impl Coverage {
    /// Counts what the round of `events`, whose inputs were `inputs`, covered with `plan`,
    /// before the plan is applied.
    fn count(&mut self, model: &Model, events: &[TreeEvent], inputs: &[PlannerInput], plan: &Plan) {
        self.max_rows = self.max_rows.max(model.rows.len());
        self.grouped_rounds += usize::from(plan.outcomes.len() > 1);
        self.deferred_events += plan.deferred.len();
        self.registrations += usize::from(plan.registration.is_some());
        self.full_rounds += usize::from(events.len() == MAX_EVENTS_PER_ROUND);
        let budget_deferrals = plan
            .deferred
            .iter()
            .filter(|index| {
                let payload_bytes = inputs[**index].payload_bytes();
                payload_bytes > 0 && plan.payload_bytes + payload_bytes > MAX_ROUND_PAYLOAD
            })
            .count();
        self.budget_deferrals += budget_deferrals;
        self.reserved_registrations += usize::from(
            budget_deferrals > 0
                && plan.registration.is_some_and(|registration| {
                    registration.children.bytes > MAX_ROUND_PAYLOAD / 2
                }),
        );
        for (index, outcome) in &plan.outcomes {
            let event = &events[*index];
            let outcome_label = match outcome {
                TreeOutcome::Applied(_) => "Applied",
                TreeOutcome::Stale(_) => "Stale",
                TreeOutcome::Untracked => "Untracked",
            };
            *self
                .outcomes
                .entry((kind_label(&event.kind), outcome_label))
                .or_default() += 1;
            match (outcome, &event.kind) {
                (TreeOutcome::Untracked, _) => self.absent_events += 1,
                (TreeOutcome::Applied(_), TreeEventKind::CancelRoot) => {
                    self.applied_cancels += 1;
                    self.crowded_cancels += usize::from(plan.outcomes.len() > 1);
                }
                (TreeOutcome::Applied(JobState::Succeeded), TreeEventKind::CommitDone)
                    if model.row(event.job_id).parent.is_some_and(|parent_id| {
                        model.row(parent_id).state != JobState::BlockingOnChildren
                    }) =>
                {
                    self.skipped_commits += 1;
                }
                _ => {}
            }
        }
        for update in &plan.updates {
            let message = update.error_message.as_deref().unwrap_or_default();
            self.propagated_failures += usize::from(message.starts_with("descendant job"));
            self.limit_failures += usize::from(message == "child-job limit exceeded");
            self.unblocked_parents += usize::from(
                update.from == JobState::BlockingOnChildren
                    && matches!(update.to, JobState::CommitReady | JobState::Succeeded),
            );
        }
    }

    /// # Panics
    ///
    /// Panics if an event kind never reached one of `outcome_labels`.
    fn assert_outcomes_reached(&self, outcome_labels: &[&'static str]) {
        for kind in [
            "Start",
            "RegularTasksDone",
            "RegularTasksDone(registers)",
            "CommitDone",
            "CleanupDone",
            "TerminalFailure(Regular)",
            "TerminalFailure(Commit)",
            "TerminalFailure(Cleanup)",
            "CancelRoot",
        ] {
            for outcome in outcome_labels {
                assert!(
                    self.outcomes
                        .get(&(kind, *outcome))
                        .copied()
                        .unwrap_or_default()
                        > 0,
                    "no {kind} event was {outcome}"
                );
            }
        }
    }
}

/// A failure that an applied event planned, to check once the plan is applied.
struct ExpectedFailure {
    job_id: JobId,
    message: String,

    /// The ancestors that fail with the job: all of them if the event found its parent
    /// `BlockingOnChildren` and it did not fail its cleanup, else none.
    chain: Vec<JobId>,
}

impl ExpectedFailure {
    /// Checks the failed job's error and its failed chain in the rows of `model`.
    ///
    /// # Panics
    ///
    /// Panics if a row of the failure is not `Failed` with the expected error.
    fn check(&self, model: &Model) {
        let row = model.row(self.job_id);
        assert_eq!(
            (row.state, row.error.as_deref()),
            (JobState::Failed, Some(self.message.as_str())),
            "failed job {}",
            self.job_id
        );
        let chain_message = format!("descendant job {} failed: {}", self.job_id, self.message);
        for ancestor_id in &self.chain {
            let ancestor = model.row(*ancestor_id);
            assert_eq!(
                (ancestor.state, ancestor.error.as_deref()),
                (JobState::Failed, Some(chain_message.as_str())),
                "ancestor {ancestor_id} of failed job {}",
                self.job_id
            );
        }
    }
}

/// A simulated committer of one tree: events arrive in its buffer, and each round plans a prefix
/// of the buffer against the model and applies the plan.
struct Simulation {
    model: Model,
    raising: Raising,
    buffer: VecDeque<(TreeEvent, Option<PlannerInput>)>,
    history: Vec<TreeEvent>,
    raised: HashMap<u64, JobState>,
}

impl Simulation {
    fn new(model: Model, raising: Raising) -> Self {
        Self {
            model,
            raising,
            buffer: VecDeque::new(),
            history: Vec::new(),
            raised: HashMap::new(),
        }
    }

    /// Runs rounds until the tree is quiescent or the round limit is reached.
    ///
    /// # Returns
    ///
    /// The number of rounds run, and whether the tree became quiescent.
    fn run(&mut self, rng: &mut StdRng, coverage: &mut Coverage) -> (usize, bool) {
        for round in 0..MAX_ROUNDS_PER_TREE {
            self.round(rng, coverage, Arrivals::All);
            if self.model.structure.is_quiescent() {
                return (round + 1, true);
            }
        }
        (MAX_ROUNDS_PER_TREE, false)
    }

    /// Runs rounds until the tree is quiescent, as [`Self::run`] does, then, if the tree is not
    /// quiescent yet, stops the copies of earlier events and runs rounds until the backlog in the
    /// buffer and [`MAX_ROUNDS_PER_TREE`] more rounds are planned. Each round plans the oldest
    /// event, so with every event raised once, a tree that is not quiescent by then lost an event.
    ///
    /// # Returns
    ///
    /// The number of rounds run.
    ///
    /// # Panics
    ///
    /// Panics if the tree is not quiescent at the end, or if a check fails.
    fn run_to_quiescence(&mut self, rng: &mut StdRng, coverage: &mut Coverage) -> usize {
        let (mut rounds, is_quiescent) = self.run(rng, coverage);
        if is_quiescent {
            return rounds;
        }
        coverage.backlogged_trees += 1;
        let max_rounds = rounds + self.buffer.len() + MAX_ROUNDS_PER_TREE;
        while !self.model.structure.is_quiescent() {
            assert!(
                rounds < max_rounds,
                "a tree whose jobs raise each event once is quiescent once its backlog is planned"
            );
            self.round(rng, coverage, Arrivals::Raised);
            rounds += 1;
        }
        rounds
    }

    /// Plans the events left in the buffer of a quiescent tree, with no new arrivals.
    ///
    /// # Panics
    ///
    /// Panics if the tree is not quiescent, or if an event is answered other than
    /// [`TreeOutcome::Untracked`].
    fn drain(&mut self, rng: &mut StdRng, coverage: &mut Coverage) {
        assert!(self.model.structure.is_quiescent());
        while let Some(plan) = self.round(rng, coverage, Arrivals::Closed) {
            assert!(
                plan.outcomes
                    .values()
                    .all(|outcome| matches!(outcome, TreeOutcome::Untracked)),
                "an event of a quiescent tree is answered {:?}",
                plan.outcomes
            );
        }
    }

    /// Runs one round: `arrivals` arrive, a random prefix of the buffer, of at most
    /// [`MAX_EVENTS_PER_ROUND`] events as the committer takes, is planned, checked and applied, and
    /// the deferred events go back to the front of the buffer. After a burst of more than
    /// [`MAX_EVENTS_PER_ROUND`] copies of earlier events, the prefix is a full round.
    ///
    /// # Returns
    ///
    /// The plan of the round, or `None` if the buffer was empty.
    ///
    /// # Panics
    ///
    /// Panics if a check fails.
    fn round(
        &mut self,
        rng: &mut StdRng,
        coverage: &mut Coverage,
        arrivals: Arrivals,
    ) -> Option<Plan> {
        let mut bursts = false;
        if arrivals != Arrivals::Closed {
            let mut raised = self.model.raise_events(self.raising, &mut self.raised, rng);
            self.history.extend(raised.iter().cloned());
            bursts = arrivals == Arrivals::All
                && !self.history.is_empty()
                && rng.random_bool(BURST_PROBABILITY);
            let num_reissued = match arrivals {
                Arrivals::All if bursts => MAX_EVENTS_PER_ROUND + rng.random_range(1..64),
                Arrivals::All => rng.random_range(0..3),
                Arrivals::Raised | Arrivals::Closed => 0,
            };
            for _ in 0..num_reissued {
                if let Some(old) = self.history.choose(rng) {
                    raised.push(old.clone());
                }
            }
            raised.shuffle(rng);
            self.buffer
                .extend(raised.into_iter().map(|event| (event, None)));
        }
        if self.buffer.is_empty() {
            return None;
        }

        let num_taken = if bursts {
            MAX_EVENTS_PER_ROUND
        } else {
            rng.random_range(1..=self.buffer.len().min(MAX_EVENTS_PER_ROUND))
        };
        let mut taken: Vec<(TreeEvent, Option<PlannerInput>)> =
            self.buffer.drain(..num_taken).collect();
        for (event, input) in &mut taken {
            if input.is_none() {
                *input = Some(self.model.collect_input(event, rng));
            }
        }
        let (events, inputs): (Vec<TreeEvent>, Vec<Option<PlannerInput>>) =
            taken.into_iter().unzip();
        let mut inputs: Vec<PlannerInput> = inputs
            .into_iter()
            .map(|input| input.expect("every input is collected"))
            .collect();
        let inputs_before = inputs.clone();
        let collected = self
            .model
            .collect_states(&events.iter().collect::<Vec<_>>());
        let mut planner_events: Vec<PlannerEvent<'_>> = events
            .iter()
            .zip(inputs.iter_mut())
            .map(|(event, input)| PlannerEvent { event, input })
            .collect();
        let plan = plan_events(&self.model.structure, &collected, &mut planner_events);
        drop(planner_events);

        check_plan(&self.model, &events, &inputs_before, &inputs, &plan);
        let failures = self
            .model
            .check_outcomes(&collected, &events, &inputs_before, &plan);
        coverage.count(&self.model, &events, &inputs_before, &plan);
        self.model.apply(&plan, rng);
        for (index, outcome) in &plan.outcomes {
            if let TreeOutcome::Applied(state) = outcome {
                let job_id = events[*index].job_id;
                assert_eq!(
                    self.model.row(job_id).state,
                    *state,
                    "outcome of event {index}"
                );
                if let PlannerInput::RegularTasksDone { outputs, .. } = &inputs_before[*index]
                    && *state != JobState::Failed
                {
                    assert_eq!(
                        self.model.row(job_id).outputs.as_ref(),
                        Some(outputs),
                        "event {index} stores its outputs"
                    );
                }
            }
        }
        for failure in &failures {
            failure.check(&self.model);
        }
        self.model.check(rng, coverage);

        for index in plan.deferred.iter().rev() {
            self.buffer
                .push_front((events[*index].clone(), Some(inputs[*index].clone())));
        }
        Some(plan)
    }
}

/// A job row of a tree generated for the repair planner.
#[derive(Debug, Clone, Copy)]
struct RepairRow {
    parent: Option<u64>,
    state: JobState,
    has_commit: bool,
    has_cleanup: bool,
}

/// How often the injected trees reached each repair.
#[derive(Debug, Default)]
struct RepairCoverage {
    consistent_trees: usize,
    orphan_aborts: usize,
    unblocked_parents: usize,
    failed_blocked_parents: usize,
    tree_failures: usize,
    continued_trees: usize,
    lenient_trees: usize,
    unmodelled_trees: usize,
    rebuilt_depth_mismatches: usize,
}

/// Prints the seed of a failing tree.
struct SeedReporter(u64);

impl Drop for SeedReporter {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "The planner property test failed for seed {}; rerun it with \
                 JOB_TREE_PROPERTY_SEED={}.",
                self.0, self.0
            );
        }
    }
}

/// # Returns
///
/// The label of an event kind, for the coverage counters.
const fn kind_label(kind: &TreeEventKind) -> &'static str {
    match kind {
        TreeEventKind::Start => "Start",
        TreeEventKind::RegularTasksDone { registers: false } => "RegularTasksDone",
        TreeEventKind::RegularTasksDone { registers: true } => "RegularTasksDone(registers)",
        TreeEventKind::CommitDone => "CommitDone",
        TreeEventKind::CleanupDone => "CleanupDone",
        TreeEventKind::TerminalFailure {
            lane: Lane::Regular,
            ..
        } => "TerminalFailure(Regular)",
        TreeEventKind::TerminalFailure {
            lane: Lane::Commit, ..
        } => "TerminalFailure(Commit)",
        TreeEventKind::TerminalFailure {
            lane: Lane::Cleanup,
            ..
        } => "TerminalFailure(Cleanup)",
        TreeEventKind::CancelRoot => "CancelRoot",
    }
}

/// # Returns
///
/// Whether an event of `kind` acts on a job in `state`, per the design's event table.
const fn acts_on(kind: &TreeEventKind, state: JobState) -> bool {
    match kind {
        TreeEventKind::Start => matches!(state, JobState::Ready),
        TreeEventKind::RegularTasksDone { .. }
        | TreeEventKind::TerminalFailure {
            lane: Lane::Regular,
            ..
        } => matches!(state, JobState::Running),
        TreeEventKind::CommitDone
        | TreeEventKind::TerminalFailure {
            lane: Lane::Commit, ..
        } => matches!(state, JobState::CommitReady),
        TreeEventKind::CleanupDone
        | TreeEventKind::TerminalFailure {
            lane: Lane::Cleanup,
            ..
        } => matches!(state, JobState::CleanupReady),
        TreeEventKind::CancelRoot => !matches!(
            state,
            JobState::CleanupReady | JobState::Succeeded | JobState::Failed | JobState::Cancelled
        ),
    }
}

/// # Returns
///
/// The stale-state error that a caller must see for an event of `kind` that finds its job in
/// `state`, per the design's event table.
const fn expected_stale_error(kind: &TreeEventKind, state: JobState) -> StaleStateError {
    let lane = match kind {
        TreeEventKind::Start => return StaleStateError::JobAlreadyStarted,
        TreeEventKind::CancelRoot => {
            return match state {
                JobState::CleanupReady => StaleStateError::JobCancellationAlreadyRequested,
                JobState::Cancelled => StaleStateError::JobAlreadyCancelled,
                state => StaleStateError::JobAlreadyTerminated(state),
            };
        }
        TreeEventKind::RegularTasksDone { .. } => Lane::Regular,
        TreeEventKind::CommitDone => Lane::Commit,
        TreeEventKind::CleanupDone => Lane::Cleanup,
        TreeEventKind::TerminalFailure { lane, .. } => *lane,
    };
    if matches!(
        state,
        JobState::Succeeded | JobState::Failed | JobState::Cancelled
    ) {
        return StaleStateError::JobAlreadyTerminated(state);
    }
    match lane {
        Lane::Regular => StaleStateError::JobNoLongerRunning,
        Lane::Commit => StaleStateError::JobNoLongerCommitReady,
        Lane::Cleanup => StaleStateError::JobNoLongerCleanupReady,
    }
}

/// # Returns
///
/// The indices of the planned events of a round in the order that the planner planned them: the
/// first `CancelRoot`, then the others in their order, then the registering event.
///
/// # Panics
///
/// Panics if more than one registering event was planned, or if the first `CancelRoot` was not.
fn planning_order(events: &[TreeEvent], plan: &Plan) -> Vec<usize> {
    let cancel_index = events
        .iter()
        .position(|event| event.kind == TreeEventKind::CancelRoot);
    let mut order: Vec<usize> = cancel_index.into_iter().collect();
    let mut registering = Vec::new();
    for index in plan.outcomes.keys() {
        if Some(*index) == cancel_index {
            continue;
        }
        if events[*index].kind.registers() {
            registering.push(*index);
        } else {
            order.push(*index);
        }
    }
    assert!(
        registering.len() <= 1,
        "at most one registering event is planned per round"
    );
    order.extend(registering);
    order
}

/// Checks the properties of one planned round of `events` over the model before the plan is
/// applied: every event is answered or deferred, with its input taken or kept accordingly.
fn check_plan(
    model: &Model,
    events: &[TreeEvent],
    inputs_before: &[PlannerInput],
    inputs_after: &[PlannerInput],
    plan: &Plan,
) {
    let mut answered: Vec<usize> = plan.outcomes.keys().copied().collect();
    answered.extend(&plan.deferred);
    answered.sort_unstable();
    assert_eq!(
        answered,
        (0..events.len()).collect::<Vec<_>>(),
        "every event is planned or deferred, once"
    );
    assert!(
        plan.outcomes.len() <= MAX_EVENTS_PER_ROUND,
        "a round plans at most {MAX_EVENTS_PER_ROUND} events"
    );
    for index in &plan.deferred {
        assert_eq!(
            inputs_after[*index], inputs_before[*index],
            "deferred event {index} keeps its input"
        );
    }
    for index in plan.outcomes.keys() {
        assert_eq!(
            inputs_after[*index],
            PlannerInput::Empty,
            "planned event {index} takes its input"
        );
    }
    let cancel_index = events
        .iter()
        .position(|event| event.kind == TreeEventKind::CancelRoot);
    if let Some(cancel_index) = cancel_index {
        assert!(
            plan.outcomes.contains_key(&cancel_index),
            "the first cancel is planned"
        );
    }
    if let Some(oldest_index) = (0..events.len()).find(|index| Some(*index) != cancel_index) {
        assert!(
            plan.outcomes.contains_key(&oldest_index),
            "the oldest event other than the first cancel is planned"
        );
    }
    let applied = plan
        .outcomes
        .values()
        .filter(|outcome| matches!(outcome, TreeOutcome::Applied(_)))
        .count();
    assert_eq!(
        applied == 0,
        plan.is_empty(),
        "a plan writes exactly when an event applies"
    );
    if plan.is_empty() {
        assert_eq!(plan.publications, Vec::new());
        assert_eq!(plan.registration, None);
    }
    for (index, outcome) in &plan.outcomes {
        let event = &events[*index];
        let in_tree = model.structure.node(event.job_id).is_some();
        assert_eq!(
            matches!(outcome, TreeOutcome::Untracked),
            !in_tree,
            "event {index} is absent exactly when its job left the tree"
        );
    }
    if let Some(registration) = plan.registration {
        let event = &events[registration.event_index];
        assert!(event.kind.registers());
        assert_eq!(event.job_id, registration.parent_id);
        assert!(matches!(
            plan.outcomes.get(&registration.event_index),
            Some(TreeOutcome::Applied(JobState::BlockingOnChildren))
        ));
    }
    let mut payload_carriers = HashSet::new();
    let mut payload_bytes = 0;
    for update in &plan.updates {
        let outputs_bytes = update.outputs.as_deref().map_or(0, outputs_bytes);
        if outputs_bytes > 0 {
            payload_carriers.insert(update.job_id);
            payload_bytes += outputs_bytes;
        }
    }
    if let Some(registration) = plan.registration {
        payload_carriers.insert(registration.parent_id);
        payload_bytes += registration.children.bytes;
    }
    assert_eq!(plan.payload_bytes, payload_bytes);
    assert!(
        plan.payload_bytes <= MAX_ROUND_PAYLOAD || payload_carriers.len() == 1,
        "a round over the payload budget carries a single event's payload"
    );
    check_plan_effects(model, plan);
}

/// Checks that the publications, the terminated jobs and the apply order of `plan` follow from
/// its updates.
fn check_plan_effects(model: &Model, plan: &Plan) {
    let mut expected_publications = Vec::new();
    let mut terminated = Vec::new();
    for update in &plan.updates {
        match update.to {
            JobState::CommitReady => {
                expected_publications.push(Publication::Commit(update.job_id));
            }
            JobState::CleanupReady => {
                expected_publications.push(Publication::Cleanup(update.job_id));
            }
            JobState::Running => expected_publications.push(Publication::Ready {
                job_id: update.job_id,
                task_indices: vec![0],
            }),
            to if to.is_terminal() => terminated.push(update.job_id),
            _ => {}
        }
    }
    let mut publications = plan.publications.clone();
    publications.sort_by_key(|publication| publication.job_id().get());
    expected_publications.sort_by_key(|publication| publication.job_id().get());
    assert_eq!(publications, expected_publications);
    assert_eq!(plan.terminated, terminated);

    let mut apply_jobs: Vec<u64> = plan
        .apply_order
        .iter()
        .map(|transition| transition.job_id.get())
        .collect();
    let rank = |job_id: JobId| {
        let depth = model.row(job_id).depth;
        (job_id == JobId::from(1), std::cmp::Reverse(depth))
    };
    assert!(
        plan.apply_order
            .windows(2)
            .all(|pair| rank(pair[0].job_id) <= rank(pair[1].job_id)),
        "the apply order is deepest first and root last"
    );
    apply_jobs.sort_unstable();
    let mut update_jobs: Vec<u64> = plan
        .updates
        .iter()
        .map(|update| update.job_id.get())
        .collect();
    update_jobs.sort_unstable();
    assert_eq!(apply_jobs, update_jobs);
}

/// Runs one random tree with `raising` until it is quiescent or the round limit is reached. With
/// [`Raising::Once`], the tree must become quiescent as [`Simulation::run_to_quiescence`] requires,
/// and the events left in the buffer are then drained.
///
/// # Returns
///
/// The number of rounds planned before the tree became quiescent or the limit was reached.
///
/// # Panics
///
/// Panics if a check fails.
fn run_tree(seed: u64, raising: Raising, coverage: &mut Coverage) -> usize {
    let _reporter = SeedReporter(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let model = Model::new(&mut rng);
    let mut simulation = Simulation::new(model, raising);
    if raising == Raising::Once {
        let rounds = simulation.run_to_quiescence(&mut rng, coverage);
        simulation.drain(&mut rng, coverage);
        return rounds;
    }
    let (rounds, is_quiescent) = simulation.run(&mut rng, coverage);
    coverage.unfinished_trees += usize::from(!is_quiescent);
    rounds
}

/// Adds children under the generated job `parent_id`, consistently with its state, and their
/// subtrees, numbering every new job after the last one.
fn add_generated_children(
    rows: &mut BTreeMap<u64, RepairRow>,
    parent_id: u64,
    depth: u32,
    rng: &mut StdRng,
) {
    if depth >= MAX_MODEL_DEPTH {
        return;
    }
    let child_states: &[JobState] = match rows[&parent_id].state {
        JobState::BlockingOnChildren => &[
            JobState::Running,
            JobState::BlockingOnChildren,
            JobState::CommitReady,
            JobState::Succeeded,
        ],
        JobState::CommitReady | JobState::Succeeded => &[JobState::Succeeded],
        JobState::CleanupReady | JobState::Failed | JobState::Cancelled => &[
            JobState::Succeeded,
            JobState::Failed,
            JobState::Cancelled,
            JobState::CleanupReady,
            JobState::CommitReady,
        ],
        JobState::Ready | JobState::Running => return,
    };
    let is_blocking = rows[&parent_id].state == JobState::BlockingOnChildren;
    let num_children = rng.random_range(usize::from(is_blocking)..=MAX_FAN_OUT);
    let mut states: Vec<JobState> = (0..num_children)
        .map(|_| {
            let state = *child_states.choose(rng).expect("there are states");
            if state == JobState::BlockingOnChildren && depth + 1 == MAX_MODEL_DEPTH {
                JobState::Running
            } else {
                state
            }
        })
        .collect();
    if is_blocking && states.iter().all(|state| *state == JobState::Succeeded) {
        states[0] = JobState::Running;
    }
    for state in states {
        let id = rows.len() as u64 + 1;
        rows.insert(
            id,
            RepairRow {
                parent: Some(parent_id),
                state,
                has_commit: rng.random_bool(0.4),
                has_cleanup: rng.random_bool(0.4),
            },
        );
        add_generated_children(rows, id, depth + 1, rng);
    }
}

/// # Returns
///
/// What recovery loads from `rows`: the non-terminal rows in ascending ID order, the structure
/// rebuilt from them, their states, and the `BlockingOnChildren` jobs with a `Failed` or
/// `Cancelled` child.
fn load(
    rows: &BTreeMap<u64, RepairRow>,
) -> (TreeStructure, HashMap<JobId, JobState>, HashSet<JobId>) {
    let loaded: Vec<RecoveredRow> = rows
        .iter()
        .filter(|(_, row)| !row.state.is_terminal())
        .map(|(id, row)| RecoveredRow {
            id: JobId::from(*id),
            parent_id: row.parent.map(JobId::from),
            state: row.state,
            resource_group_id: ResourceGroupId::from(1),
            has_commit: row.has_commit,
            has_cleanup: row.has_cleanup,
        })
        .collect();
    let failed_child = rows
        .values()
        .filter(|row| matches!(row.state, JobState::Failed | JobState::Cancelled))
        .filter_map(|row| row.parent)
        .filter(|parent_id| {
            rows.get(parent_id)
                .is_some_and(|parent| parent.state == JobState::BlockingOnChildren)
        })
        .map(JobId::from)
        .collect();
    (
        TreeStructure::from_rows(JobId::from(1), &loaded),
        loaded.iter().map(|row| (row.id, row.state)).collect(),
        failed_child,
    )
}

/// Writes the repair `plan` into `rows`, checking its guards against the loaded `states` and
/// its edges, and counts which repair it is.
///
/// # Panics
///
/// Panics if a check fails.
fn apply_repair(
    rows: &mut BTreeMap<u64, RepairRow>,
    states: &HashMap<JobId, JobState>,
    plan: &Plan,
    coverage: &mut RepairCoverage,
) {
    let mut repaired_job_ids = HashSet::new();
    for update in &plan.updates {
        assert!(
            repaired_job_ids.insert(update.job_id),
            "job {} is repaired twice",
            update.job_id
        );
        assert_eq!(
            states.get(&update.job_id),
            Some(&update.from),
            "guard of job {}",
            update.job_id
        );
        assert!(
            JobState::is_valid_transition(update.from, update.to),
            "invalid edge {} -> {} on job {}",
            update.from,
            update.to,
            update.job_id
        );
        rows.get_mut(&update.job_id.get())
            .expect("a repaired row exists")
            .state = update.to;
    }
    let messages: Vec<&str> = plan
        .updates
        .iter()
        .filter_map(|update| update.error_message.as_deref())
        .collect();
    if messages
        .iter()
        .any(|message| message.starts_with("tree invariant violation"))
    {
        coverage.tree_failures += 1;
    } else if messages
        .iter()
        .any(|message| message.starts_with("tree repair"))
    {
        coverage.failed_blocked_parents += 1;
    } else {
        coverage.unblocked_parents += usize::from(plan.updates.iter().any(|update| {
            update.from == JobState::BlockingOnChildren
                && matches!(update.to, JobState::CommitReady | JobState::Succeeded)
        }));
        coverage.orphan_aborts += usize::from(
            plan.updates
                .iter()
                .any(|update| matches!(update.to, JobState::CleanupReady | JobState::Cancelled)),
        );
    }
}

/// Checks the depths that [`TreeStructure::from_rows`] rebuilt from the loaded rows of `rows`:
/// a job whose ancestors were all loaded has its depth in the rows, and any other job, whose depth
/// the rebuild cannot know, can no longer register children.
///
/// # Panics
///
/// Panics if a check fails.
fn check_rebuilt_depths(
    rows: &BTreeMap<u64, RepairRow>,
    structure: &TreeStructure,
    coverage: &mut RepairCoverage,
) {
    for job_id in structure.sorted_job_ids() {
        let mut depth = 0;
        let mut ancestors_loaded = true;
        let mut ancestor = rows[&job_id.get()].parent;
        while let Some(ancestor_id) = ancestor {
            depth += 1;
            let ancestor_row = &rows[&ancestor_id];
            ancestors_loaded &= !ancestor_row.state.is_terminal();
            ancestor = ancestor_row.parent;
            if depth > rows.len() {
                ancestors_loaded = false;
                break;
            }
        }
        let rebuilt_depth = structure
            .node(job_id)
            .expect("a listed job should be in the tree")
            .depth as usize;
        if ancestors_loaded {
            assert_eq!(rebuilt_depth, depth, "rebuilt depth of job {job_id}");
        } else if rebuilt_depth != depth {
            coverage.rebuilt_depth_mismatches += 1;
            assert!(
                !matches!(
                    rows[&job_id.get()].state,
                    JobState::Ready | JobState::Running | JobState::BlockingOnChildren
                ),
                "job {job_id}, which can still register children, has a wrong rebuilt depth"
            );
        }
    }
}

/// Generates a consistent tree, checks that it needs no repair, injects up to three
/// corruptions, repairs it, checks the repaired rows, then plans events whose jobs raise each
/// event once against the repaired tree until it is quiescent.
///
/// # Panics
///
/// Panics if a check fails.
fn repair_tree(seed: u64, coverage: &mut RepairCoverage) {
    let _reporter = SeedReporter(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let root_state = *[
        JobState::Ready,
        JobState::Running,
        JobState::BlockingOnChildren,
        JobState::BlockingOnChildren,
        JobState::BlockingOnChildren,
        JobState::CommitReady,
        JobState::CleanupReady,
        JobState::Succeeded,
        JobState::Failed,
        JobState::Cancelled,
    ]
    .choose(&mut rng)
    .expect("there are states");
    let mut rows = BTreeMap::from([(
        1,
        RepairRow {
            parent: None,
            state: root_state,
            has_commit: rng.random_bool(0.4),
            has_cleanup: rng.random_bool(0.4),
        },
    )]);
    add_generated_children(&mut rows, 1, 0, &mut rng);
    let (structure, states, failed_child) = load(&rows);
    assert!(
        plan_repair(&structure, &states, &failed_child).is_empty(),
        "a consistent tree needs no repair: {rows:?}"
    );

    let num_rows = rows.len() as u64;
    let num_corruptions = rng.random_range(0..=3);
    for _ in 0..num_corruptions {
        let id = rng.random_range(1..=num_rows);
        let row = rows.get_mut(&id).expect("the row exists");
        match rng.random_range(0..10) {
            0 if id != 1 => row.parent = Some(rng.random_range(1..=num_rows)),
            1 => row.has_cleanup = !row.has_cleanup,
            _ => row.state = *ALL_STATES.choose(&mut rng).expect("there are states"),
        }
    }
    let (structure, states, failed_child) = load(&rows);
    if structure.is_quiescent() {
        return;
    }
    let plan = plan_repair(&structure, &states, &failed_child);
    if num_corruptions == 0 {
        assert!(plan.is_empty());
        coverage.consistent_trees += 1;
    }

    apply_repair(&mut rows, &states, &plan, coverage);

    let (structure, states, failed_child) = load(&rows);
    if let Err(violation) = check_invariants(&structure, &states) {
        panic!("the repaired tree breaks an invariant: {violation}: {rows:?}");
    }
    for (id, row) in &rows {
        if row.state == JobState::BlockingOnChildren {
            assert!(
                rows.values()
                    .filter(|child| child.parent == Some(*id))
                    .all(|child| !matches!(
                        child.state,
                        JobState::Failed | JobState::Cancelled | JobState::CleanupReady
                    )),
                "I2: repaired blocking job {id} has a failed child: {rows:?}"
            );
        }
    }
    assert!(
        plan_repair(&structure, &states, &failed_child).is_empty(),
        "a repaired tree needs no second repair: {rows:?}"
    );
    check_rebuilt_depths(&rows, &structure, coverage);
    if structure.is_quiescent() {
        return;
    }
    if rows.values().any(|child| {
        child.parent.is_some_and(|parent_id| {
            matches!(rows[&parent_id].state, JobState::Ready | JobState::Running)
        })
    }) {
        coverage.unmodelled_trees += 1;
        return;
    }

    let model = Model::from_repaired(&rows, structure, &mut rng);
    coverage.continued_trees += 1;
    coverage.lenient_trees += usize::from(model.lenient_terminal_rows);
    let mut simulation = Simulation::new(model, Raising::Once);
    let mut continuation_coverage = Coverage::default();
    simulation.model.check(&mut rng, &mut continuation_coverage);
    simulation.run_to_quiescence(&mut rng, &mut continuation_coverage);
    simulation.drain(&mut rng, &mut continuation_coverage);
}

/// # Returns
///
/// The number of random cases that a property test runs: `JOB_TREE_PROPERTY_ROUNDS` if it is set,
/// otherwise 3,000 in debug builds and 20,000 in others.
fn num_property_cases() -> u64 {
    std::env::var("JOB_TREE_PROPERTY_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(if cfg!(debug_assertions) {
            3_000
        } else {
            20_000
        })
}

/// # Returns
///
/// The seed of the only case that a property test reruns, from `JOB_TREE_PROPERTY_SEED`, if it is
/// set.
fn property_seed() -> Option<u64> {
    std::env::var("JOB_TREE_PROPERTY_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Runs random trees with `raising` until [`num_property_cases`] rounds are planned, or only the
/// tree of [`property_seed`] if it is set.
///
/// # Returns
///
/// The coverage of the trees, or `None` if only one seed was rerun.
///
/// # Panics
///
/// Panics if a check fails.
fn run_trees(raising: Raising) -> Option<Coverage> {
    if let Some(seed) = property_seed() {
        run_tree(seed, raising, &mut Coverage::default());
        return None;
    }
    let num_rounds =
        usize::try_from(num_property_cases()).expect("the number of rounds should fit a usize");
    let base_seed: u64 = rand::random();
    let mut coverage = Coverage::default();
    let mut rounds = 0;
    let mut num_trees = 0u64;
    while rounds < num_rounds {
        rounds += run_tree(base_seed.wrapping_add(num_trees), raising, &mut coverage);
        num_trees += 1;
    }
    println!(
        "Planned {rounds} rounds over {num_trees} random trees raising events {raising:?} (base \
         seed {base_seed}): {coverage:?}."
    );
    Some(coverage)
}

#[test]
fn repaired_trees_keep_the_invariants() {
    let num_trees = num_property_cases();
    if let Some(seed) = property_seed() {
        repair_tree(seed, &mut RepairCoverage::default());
        return;
    }
    let base_seed: u64 = rand::random();
    let mut coverage = RepairCoverage::default();
    for tree in 0..num_trees {
        repair_tree(base_seed.wrapping_add(tree), &mut coverage);
    }
    println!("Repaired {num_trees} random trees (base seed {base_seed}): {coverage:?}.");
    for (case, count) in [
        ("consistent trees", coverage.consistent_trees),
        ("orphan aborts", coverage.orphan_aborts),
        ("unblocked parents", coverage.unblocked_parents),
        ("failed blocked parents", coverage.failed_blocked_parents),
        ("tree failures", coverage.tree_failures),
        ("repaired trees run to quiescence", coverage.continued_trees),
    ] {
        assert!(count > 0, "the injected trees never reached {case}");
    }
}

#[test]
fn planned_rounds_keep_the_tree_consistent() {
    let Some(coverage) = run_trees(Raising::Repeated) else {
        return;
    };
    for (case, count) in [
        ("grouped rounds", coverage.grouped_rounds),
        ("deferred events", coverage.deferred_events),
        ("registrations", coverage.registrations),
        ("absent events", coverage.absent_events),
        ("applied cancels", coverage.applied_cancels),
        ("propagated failures", coverage.propagated_failures),
        ("limit failures", coverage.limit_failures),
        ("skipped commits", coverage.skipped_commits),
        ("unblocked parents", coverage.unblocked_parents),
        (
            "cancels planned with other events",
            coverage.crowded_cancels,
        ),
        ("payload budget deferrals", coverage.budget_deferrals),
        (
            "large registrations with budget deferrals",
            coverage.reserved_registrations,
        ),
        ("full rounds", coverage.full_rounds),
        (
            "perturbations that break an invariant",
            coverage.broken_perturbations,
        ),
        (
            "perturbations that break nothing",
            coverage.harmless_perturbations,
        ),
    ] {
        assert!(count > 0, "the random trees never reached {case}");
    }
    coverage.assert_outcomes_reached(&["Applied", "Stale"]);
}

#[test]
fn events_raised_once_are_each_answered_and_quiesce_the_tree() {
    let Some(coverage) = run_trees(Raising::Once) else {
        return;
    };
    for (case, count) in [
        ("grouped rounds", coverage.grouped_rounds),
        ("deferred events", coverage.deferred_events),
        ("registrations", coverage.registrations),
        ("applied cancels", coverage.applied_cancels),
        ("propagated failures", coverage.propagated_failures),
        ("limit failures", coverage.limit_failures),
        ("skipped commits", coverage.skipped_commits),
        ("unblocked parents", coverage.unblocked_parents),
        ("full rounds", coverage.full_rounds),
    ] {
        assert!(count > 0, "the random trees never reached {case}");
    }
    coverage.assert_outcomes_reached(&["Applied", "Stale", "Untracked"]);
}
