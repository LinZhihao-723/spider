# Slice S3 `s3-planner`: notes

Role: implementer. Worktree: `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`, on top of S1 and S2). Only `cache/tree.rs` and `state/tree_committer.rs` changed in this slice.

## What changed

- `cache/tree.rs`:
  - Constants: `MAX_EVENTS_PER_ROUND = 256`, `MAX_ROUND_PAYLOAD = 16 MiB`, `MAX_CHILDREN_PER_JOB = 10_000`, `MAX_JOB_CHILD_BYTES = 16 MiB` (planner side; S4's `state/limits.rs` should reuse or mirror them), and a private `MAX_PROPAGATED_MESSAGE_CHARS = 1024` for the ancestors' "descendant job … failed: …" message.
  - `TreeEventKind::registers()` and `reads_every_job()` (`CancelRoot`, `TerminalFailure`, registering `RegularTasksDone`).
  - `TreeStructure`: `sorted_job_ids`, `ancestors` (in-tree ancestors, nearest first), `jobs_to_collect(events)` (shared by the committer and the property test, so they cannot drift), `insert`, `set_active_children`.
  - `ChildSetSummary { count, bytes }`; `PlannerInput::RegularTasksDone { outputs, children: Option<ChildSetSummary> }`; `PlannerInput` is now `Clone`.
  - `EventOutcome::Absent`; `PlannedRegistration`; `Plan` gains `registration` and `payload_bytes`.
  - `plan_events` (still `pub(crate)`, same signature as S2):
    - planning order: the first `CancelRoot`, then the rest in slice order, then the first registering event; later registering events are deferred;
    - deferral: an event that would update a job already updated in the round; an event that would raise the payload above 16 MiB unless the plan writes nothing yet; events past the 256th; and any later event of a job that was deferred for one of those reasons (keeps per-job order);
    - a planned event takes its input (outputs move into the update, no copy); a deferred one keeps it.
  - The overlay (`state`, `job_state`, `is`, `node`, `parent`, `active_children`, `is_abortable`, `abort_target`, `blocking_ancestors`, `merge`) with `NodeState::{Present, Absent}`. A node missing from the structure reads `Absent`; a node in the structure whose state was not collected panics (a collect gap is a bug).
  - Planners: `plan_start`, `plan_regular_done` (no children → `plan_success`; over a limit → `fail_and_abort(…, "child-job limit exceeded")`; else `Running → BlockingOnChildren` with outputs, counter `k`, registration), `plan_success`, `climb_after_success`, `unblock`, `plan_commit_done`, `plan_cleanup_done`, `plan_failure`, `fail_and_abort` (chain to the root, then `abort_rest`; no propagation for a cleanup failure or an inactive/absent parent; a `BlockingOnChildren` from-state aborts the rest even without propagation, for S6's `fail_blocked`), `abort_rest`, `plan_cancel` (skips non-root `CommitReady`), `deactivate`.
  - `pub fn check_invariants<Hasher: BuildHasher>(structure, states) -> Result<(), Violation>`: I4, then per node I5 (state present, non-terminal), I1 (parent link, only the root `Ready`, parent id smaller), I2 (no `CleanupReady` child under a blocking parent), I3 (active job ⇒ every ancestor in the tree and blocking; non-root `CommitReady` ⇒ parent blocking, `CleanupReady` or absent), then the I2 counters (≥ 1 and equal to the in-tree children). `Violation { invariant, job_id, detail }`. It is `pub` and hasher-generic so the release build has no dead code (the committer calls it only in debug builds).
- `state/tree_committer.rs`:
  - `take_events`: the oldest `CancelRoot` from anywhere in the buffer first, then events from the front up to 256; only the oldest registering event is taken, the later ones stay in the buffer in place. `MAX_EVENTS_PER_RECEIVE` is replaced by `MAX_EVENTS_PER_ROUND`.
  - `collect_inputs`: states of `structure.jobs_to_collect(events)`; an event whose job is not in `handles` gets `PlannerInput::None` without touching any JCB (fixes the S2 review issue 3: such an event is no longer fatal). `children` is always `None` until S4.
  - After the apply: `CommitterState::update_structure` writes `plan.counters`, then drops terminal nodes and handles.
  - `reply`: `EventOutcome::Absent` → `TreeOutcome::TreeClosed` (see deviation 1).
  - `assert!(plan.registration.is_none())` (the committer collects no children before S4).
  - Debug builds only: `assert_invariants` reads every handle's state and panics on a `Violation`, which poisons through the guard.
- `TreeOutcome::TreeClosed`'s docstring now covers both meanings.

## Tests

- `cache::tree::tests` (26 after the review fixes): the S2 cases on the new helper, plus the named cases of design §15 and plan S3: two siblings failing in one round (chain once, the second stale, apply order `[3,4,2,5,1]`), cancel + registration + completion in one round, cancel + failure in one round (failure stale, message not stored), two registering parents (second deferred with its input; the first registers with `PlannedRegistration`), a child set over `MAX_JOB_CHILD_BYTES` and over `MAX_CHILDREN_PER_JOB` (job fails, chain fails, siblings aborted, outputs not stored), a skipped `CommitReady` job whose aborted parent is `Absent` (commit done and commit failure stay local; the parent's own late event is `Absent`), an all-stale round (empty plan), sibling completions unblocking the parent once and the later `CommitDone` decrementing the root, a cascade to the root, cancel skipping a non-root `CommitReady`, the touch-twice deferral, the payload budget (third 5.3 MiB event deferred; a single 16 MiB+1 event admitted), the 256 cap, `jobs_to_collect`, message truncation, and one `check_invariants` corruption per invariant (I1, I2 ×2, I3 ×2, I4, I5).
- `cache::tree::property::planned_rounds_keep_the_tree_consistent` (T11): an in-memory row model; random flags, registrations (count 1-4, 5 % over the byte limit, depth ≤ 5); arrivals sampled from the current rows plus re-issued old events (stale); random round boundaries; inputs collected as the committer does (and kept when deferred); deferred events requeued at the front. After every round: every event planned or deferred once; deferred inputs unchanged and planned inputs taken; the first cancel is planned; a plan writes iff some event is `Applied`; `Absent` iff the job left the tree; the registration's event is registering and `Applied(BlockingOnChildren)`; payload ≤ 16 MiB when more than one event applies; publications, `terminated` and the apply order follow from the updates; no guard miss, no job updated twice, every edge valid; each `Applied(s)` matches the model and stores its outputs; full I1-I5 and "success is earned" on every row (including terminal children); and `check_invariants` returns `Ok`. `JOB_TREE_PROPERTY_ROUNDS` (default 3,000 in debug, 20,000 in release) and `JOB_TREE_PROPERTY_SEED` (replays one tree); a failing tree prints its seed. The test asserts that every coverage counter is non-zero. A typical debug run: 3,000 rounds over ~430 trees, up to 26 rows, ~1,050 grouped rounds, ~1,800 deferred events, ~450 registrations, ~600 `Absent` events, ~15 applied cancels, ~260 propagated failures, ~25 limit failures, ~40 skipped commits, ~90 unblocked parents.
- `state::tree_committer::tests::hooked` (+3, `CommitterAfterReceive` pause plus an `Observe@CommitterBeforePlan` round counter): two failures of one job in one round (`Applied(Failed)`, `Stale(JobAlreadyTerminated(Failed))`, one round, the first message stored); a cancel planned before a failure that arrived first (round order `[CancelRoot, TerminalFailure]`, failure `Stale(JobNoLongerRunning)`, no error stored, `CleanupReady`); a start and a failure of the same job (first round holds both, the failure is deferred and applied in the second round).

Mutation checks (each applied, run against `cache::tree` or `state::tree_committer`, then reverted):

| Defect | Caught by |
|---|---|
| no touch-twice check | `an_event_that_would_change_a_job_again_is_deferred_with_its_input` only (see the note below) |
| cancel not planned first | both cancel named tests and the property test |
| overlay not merged between events | 6 named tests and the property test |
| parent counter not decremented | 2 named tests and the property test (I2 counter) |
| `CommitReady` treated as abortable | the cancel-skip named test and the property test |
| `jobs_to_collect` without the "every job" rule | the property test (panic on an uncollected state) and the `jobs_to_collect` test |
| input taken before admission | 3 named tests (after this, the property test also checks stored outputs) |
| no payload budget | the payload named test |
| no failure propagation | 4 named tests and the property test |
| extra registering parents not deferred | the two-registrations named test and the property test |
| committer `take_events` back to "oldest only" | all three committer grouping tests |

Note on the touch-twice rule (for the report): with cancel first, registration last, and every event planned against the overlay, no event sequence that the real producers can generate makes a later event update a row that an earlier event of the same round updated. The property test, which only generates forward-consistent events (plus stale re-issues), never triggers it. The rule only fires for orders that cannot happen in production (for example a commit failure queued behind the `RegularTasksDone` that makes the job `CommitReady`), so it is a safety net, and its cost is nil. R12's "a grouped `TerminalFailure` deferred behind a completion that touched the parent" does not arise: a completion that touches the parent unblocks it, and then the failing sibling cannot exist.

## Verification

- `cargo build --workspace`, `cargo build --workspace --all-targets --all-features`, `cargo build --release -p spider-storage` (no warnings): OK.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo clippy -p spider-storage --all-targets -- -D warnings` and `--features test-hooks`: clean. `cargo +nightly fmt --all -- --check`: clean.
- `cargo nextest run -p spider-storage --all-features --run-ignored all --release --no-fail-fast` on container `jtp-impl-s3` (now stopped): two full runs, each **231/232 passed**. The single failure differed between the runs: `mariadb_test::test_get_dead_execution_managers_multiple` in the first, `mariadb_test::test_get_dead_execution_managers_marks_dead` in the second. These are the `CURRENT_TIMESTAMP`/1 s-threshold execution-manager liveness tests that the S1 note already lists as failing about 1 run in 4 on both trees. Their code and test file are untouched by this slice. Run alone, `…_multiple` passed 10/10 on the prototype afterwards (it also failed once alone, right after the first full run) and 9/9 on the baseline (`/home/lzh/dev/spider`, built into `claude/benchmarks/job-tree-proto/target-main`). Every tree, committer, publisher, stale and T9 test passed in both runs, `planned_rounds_keep_the_tree_consistent` ran its 20,000 release rounds, and no `jt_*` scratch database was left behind.
- The planner and committer unit tests also pass in a debug build (`cargo test -p spider-storage --features test-hooks --lib`), where the committer's `assert_invariants` runs after every round.

## Deviations from the plan and the design

1. **An event whose own job is `Absent` is answered `TreeClosed`, not with a planner-chosen stale error.** The planner returns `EventOutcome::Absent`; the committer replies `TreeOutcome::TreeClosed`; the JCB then maps it, as it already does after a quiescent exit, to `kind.stale_error(its cached state)`, which is terminal. Design §6 gives `Lane::stale_error(Absent)` = the lane's "no longer" error, but has no answer for a `CancelRoot` of an absent root (the root is terminal but the tree still has skipped commits or cleanups): the planner cannot know whether it is `Cancelled`, `Failed` or `Succeeded`. The JCB can. Both map to FAILED_PRECONDITION. Result: an absent-job report gets `JobAlreadyTerminated(state)` (or `JobAlreadyCancelled` for a cancel) rather than `JobNoLongerRunning`/`…CommitReady`/`…CleanupReady`.
2. **A registering `RegularTasksDone` collects every job's state**, like `CancelRoot` and `TerminalFailure`: a child set over the limits fails the job and aborts the tree, which reads every node. The plan lists only cancel and failure (design N4).
3. **Per-job order inside a round:** once an event of a job is deferred (touch, payload or cap), every event of that job that comes later in the round's slice (arrival order, not planning order) is deferred too. The design does not say; this keeps a job's events in arrival order. An extra registering event does not trigger it, so a `CancelRoot` (whose job is the root) is never held back by a deferred registration of the root (a bug the property test found in the first version). Keying on arrival order matters because the registration is planned last: a later event of the registering job that is deferred must not hold back the registration that arrived before it (review fix, see below).
4. **The planner also enforces the round order** (cancel first, registration last, extra registrations deferred), not only `take_events`, so the pure planner and the property test cover it. `take_events` keeps the plan's rules: it pulls the oldest cancel from anywhere in the buffer (the planner can only reorder what it is given) and leaves later registering events in the buffer uncollected.
5. **`fail_blocked` (repair R4) is not a separate function yet**; `fail_and_abort` handles `from == BlockingOnChildren` (no propagation, abort the rest). S6 adds the thin wrapper, to avoid dead code now.
6. **`check_invariants` is `pub` and generic over the hasher** (plan: `pub fn`; S2 kept `plan_events` `pub(crate)` for clippy's `implicit_hasher`). A `pub(crate)` version is dead code in release builds, where only debug builds call it.
7. **The 16 MiB payload budget counts the stored outputs plus the registration's child bytes.** "The first event is always admitted" (design §7.5 rule 5, §13) is implemented as: an event that writes no payload is never deferred for the budget; an event with payload is admitted over the budget only while the round carries no payload yet; and if the round's oldest event other than the first `CancelRoot` registers children, its whole input payload (outputs plus child bytes) is reserved from the start of the round, because I4 makes it plan last. Before the review fix, "first" meant "first writer in planning order", which a registration can never be while anything else in the tree writes, so a registration could be deferred forever (slice review 03, issue 1).
8. **The ancestors' error message truncates the failing job's message at 1,024 characters** (design: `truncate(msg)`, length unspecified; the column is `LONGTEXT`).

## For the next slice (S4)

- `collect_inputs` must fill `PlannerInput::RegularTasksDone.children` with a `ChildSetSummary` (the count and the summed compressed bytes of every TCB's prepared children) when the event `registers()`, and keep the prepared children next to it in `Queued` (plan S4). The `assert!(plan.registration.is_none())` in `run_tree_state_transaction` is the placeholder to replace with the INSERT and the child binding.
- `Plan.registration` carries `parent_id`, `root_id`, `resource_group_id`, `child_depth`, `children` and `event_index`. The planner already set the parent's counter to `k` in `plan.counters`; `update_structure` writes it, so S4 must add the child `TreeNode`s (with `active_children = 0`) and handles after the apply, and must not set the counter again.
- The children's ready-task publications are not produced by the planner; S4 stages them from the prepared children after binding.
- `TreeStructure::insert` exists for the child nodes.
- `MAX_CHILDREN_PER_JOB` and `MAX_JOB_CHILD_BYTES` are in `cache/tree.rs`; S4's service-level per-report limits should use the same values.
- On a deterministic payload error, the round's events must be requeued with their inputs cleared (as `requeue_all` does) except that the registration becomes a `TerminalFailure`; the planner's deferred inputs are only valid for the round's own pre-state.
- The debug invariant check reads every node's JCB state after every round; with S4's bigger trees in debug-build tests this is O(nodes) read locks per round.

## Line counts (approximate)

- `cache/tree.rs`: non-test 738 → 1,501 (+763, about half docstrings); tests 334 → 1,538.
- `state/tree_committer.rs`: non-test 758 → 827 (+69); tests 361 → 525.
- About 830 non-test lines against the plan's 550; the overshoot is docstrings, `check_invariants` (≈ 200 lines with its checker type) and the shared `jobs_to_collect`.

## Review fixes (slice review 03)

All six issues are fixed. No rebuttal.

1. **Major: payload budget starving a registration.** Fixed in the planner (`cache/tree.rs`), with no change to `take_events`:
   - `RoundOrder` (replaces `planning_order`) also records `oldest_registration`: the first registering event's index, if it is the oldest event of the round other than the first `CancelRoot`.
   - `plan_events` reserves that event's input payload (`PlannerInput::payload_bytes`: outputs plus child bytes, an upper bound of what its plan writes) from the start of the round and releases the reservation when the registration itself is planned (last, as I4 requires). `Plan::admits` counts the reservation as carried payload. An event without payload is never deferred for the budget. That exemption is needed because a reservation over the budget would otherwise defer the cancel.
   - The per-job deferral now keys on arrival order (deviation 3), so a later event of the registering job that is deferred cannot hold back the registration.
   - Liveness: each round plans (answers) the oldest event other than the first cancel. Deferred events go back to the front in round order, so every event becomes the oldest within a bounded number of rounds. The property test now asserts this every round.
   - Test: `a_registration_deferred_for_the_payload_budget_is_planned_in_the_next_round`. This is the review's two-round scenario: deferred in round 1, planned in round 2 with the competing completion deferred. A cancel is still planned against a 16 MiB + 1 reservation, and a small registration leaves room for others. `the_payload_budget_defers_events_but_always_admits_the_first` also checks that a failure (no payload) joins a round that is already over budget.
2. **`take_events` untested for leaving extra registrations.** New pure test `take_events_takes_the_oldest_cancel_first_and_leaves_later_registrations_in_the_buffer`. The buffer `[Reg 2, CommitDone 3, Reg 4, Cancel 1, CleanupDone 5]` gives the round `[Cancel 1, Reg 2, CommitDone 3, CleanupDone 5]`, and the buffer keeps `[Reg 4]` with no input.
3. **Touch-twice tests drive impossible sequences.**
   - `Plan::admits`'s docstring now says that the producers never send such an event, so the check only guards against a broken producer.
   - The planner unit test stays as the check of that safety net.
   - The committer test is renamed to `an_injected_event_that_would_change_its_job_again_waits_for_the_next_round`. It is kept, not dropped, because it is the only committer-level test of deferred events being requeued with their inputs.
   - For the report's traps: the rule never fires for real producers.
4. **Property test weak on `check_invariants` and cancels.**
   - The model's checks now return `Result` (a local `ensure!`). After the full check passes on every round, the test perturbs one random structure job's state (any of the 8) or counter (0 to 5). It then asserts that `check_invariants` returns `Err` exactly when the model, restricted to what the structure sees, finds a violation. The restriction hides terminal children from I2 and skips the "success is earned" consequence, which `check_invariants` cannot see by design. The perturbation is then undone.
   - The cancel rate is raised from 1 % to 2 % per round, with a new counter for applied cancels that share their round with another planned event.
   - New coverage counters, each asserted non-zero:
     - payload-budget deferrals;
     - large registrations planned in a round with budget deferrals;
     - perturbations that break an invariant, and perturbations that break nothing.
   - To exercise the budget, the generator now sometimes produces:
     - 6 MiB outputs (5 %);
     - registrations of exactly `MAX_JOB_CHILD_BYTES` (10 %);
     - registrations of `MAX_JOB_CHILD_BYTES / 2 .. MAX_JOB_CHILD_BYTES` (10 %).
   - The old "payload ≤ 16 MiB when more than one event applies" check is replaced. The test now checks that `payload_bytes` equals the stored outputs plus the registration's bytes, and that a round over the budget carries a single event's payload (zero-payload events may now join it).
   - A typical debug run (3,000 rounds, ~470 trees):
     - ~1,050 grouped rounds, ~1,700 deferred events, ~410 registrations, ~600 `Absent` events;
     - ~30 applied cancels, ~24 of them with other events in the round;
     - ~115 budget deferrals, 7 large registrations with budget deferrals;
     - ~790 breaking and ~1,330 harmless perturbations.
5. **`submit_event`'s `# Errors`** now also names "the job is no longer in the tree".
6. **Style.**
   - In `cache/tree.rs`, `impl Plan` follows `struct Plan` and `impl Violation` follows `struct Violation`. The `pub` items come first, then `pub(crate) fn plan_events`, then the private const and the private types.
   - `Overlay::parent` has a `# Returns`.
   - `InvariantChecker::check_counters` uses `expect` instead of tolerating a missing node.

Mutation checks of the fixes (each applied, run against `cache::tree` and `state::tree_committer` in debug, then reverted):

| Defect | Caught by |
|---|---|
| no reservation for the oldest registration (the review's bug) | the new named test, and the property test's liveness check (9 of 10 debug runs; the release run plans 20,000 rounds) |
| per-job deferral by planning order instead of arrival order | the property test (liveness) |
| `check_invariants` always `Ok` | the property test's perturbation check and the corruption test |
| no exemption for events without payload | the new named test (cancel against a reservation) and the payload test |
| `take_events` takes every registering event | the new `take_events` test |

For the report's traps: if the registration is planned last (for I4) and the payload budget is enforced in planning order, the registration loses every round while siblings keep completing. The budget has to follow arrival order for the oldest event.

Verification after the fixes:

- `cargo build --workspace` and `cargo build --release -p spider-storage`: OK, no warnings.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`, plus `-p spider-storage --all-targets` with and without `--features test-hooks`: clean.
- `cargo +nightly fmt --all -- --check`: clean.
- `cargo nextest run -p spider-storage --all-features --run-ignored all --release --no-fail-fast` on container `jtp-impl-s3fix` (now stopped): two full runs, each **234/234 passed**. This includes the property test's 20,000 release rounds. No `jt_*` scratch database was left behind.
- Debug build, `cargo test -p spider-storage --features test-hooks --lib`: 141/141 passed. Here the committer's `assert_invariants` runs after every round. The property test also passed 30 more debug runs with random seeds.
- The kind cluster `clp-rc2` was left stopped: `clp-rc2-control-plane` is still `Exited (137)`. It was not started or deleted.

