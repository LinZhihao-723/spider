# Review: style and simplicity of the whole prototype (style reviewer)

Verdict: **fix_needed**. Nothing here changes the design's semantics or blocks the benchmarks. But the diff breaks several rules that `rust-coding-style.md` and `rust-pr-review.md` document: a dropped docstring, missing `# Type Parameters` sections, a positional `Vec` of join handles, an `impl` that does not follow its type, and a lint failure on the current nightly. The review guide says that more than one or two such violations means a self-audit pass is due. The biggest simplification wins are structural: the `test-hooks` cfg sprawl, the phantom generic parameters, the dead `InternalJobOrchestration` methods, and the round-shaping rules that are written twice. Fixing those would shrink the storage diff noticeably before the line count goes into the report.

Reviewed: `git diff 278d06f` plus the untracked files in `/home/lzh/dev/spider-job-tree-proto` on 2026-10-09. That is 31 modified files (+4,388/−1,029) and 15 new files (12,158 lines). I read every production file of the diff in full. I read the tests for style, duplication and assertion form, but not for coverage, which the slice reviewer's slice reviews cover. I also re-read both guides, design §5.2, §9 and §11, and the brief.

Mechanical checks (worktree target dir; no code changed, no MariaDB started):
- `cargo +nightly fmt --all -- --check`: clean.
- `cargo +nightly clippy -p spider-storage -p spider-core -p spider-proto-rust -p spider-execution-manager --all-targets --all-features -- -D warnings`: **fails** with one error (issue 9). Rerun with `-A deprecated`: clean.
- `cargo +nightly clippy -p spider-storage --all-targets -- -D warnings` (default features, so the `not(test-hooks)` paths): clean.
- Greps over the added lines:
  - No `unwrap()`.
  - No sigil spacing errors (`= ?x` / `= %x`).
  - No `#[allow]`.
  - No capitalized or period-terminated `expect`/`#[error]` messages.
  - The added inline `//` comments are clean. The only new one, `state/publisher.rs:111`, justifies a `let _ =`, as the review guide requires.

Host: the relayed user request asked for the kind cluster to be killed. `clp-rc2-control-plane` was already `Exited (137)` from about 6 hours earlier, so there was nothing to kill. It was neither started nor deleted.

Paths below are relative to `components/spider-storage/` unless they say otherwise.

## Major

### 1. The `test-hooks` feature is threaded through production code by about 80 `cfg` attributes

- **Where.** Twelve source files carry `#[cfg(feature = "test-hooks")]` or `#[cfg(not(feature = "test-hooks"))]`: `state/tree_committer.rs` 21, `state/service.rs` 15, `cache/tree.rs` 9, `cache/job.rs` 8, `cache/task.rs` 7, `state/runtime.rs` 7, `db/mariadb.rs` 5, and others.
- **Layers that exist only to carry the hooks:**
  - `ServiceState::new` / `new_with_hooks` / `build` (`state/service.rs:137-167, 1172`).
  - `create_runtime` / `create_runtime_with_hooks` / `build_runtime` (`state/runtime.rs:180-243`).
  - `SharedTaskControlBlock::succeed_task_instance_with_children` / `succeed_task_instance_with_hooks` / `succeed_task_instance_impl` (`cache/task.rs:471-656`).
  - Parameters that exist only with the feature, on `write_plan`, `apply_plan` and `TreeControlBlock::new` (`state/tree_committer.rs:724-794`, `cache/tree.rs:202-206`).
- **Paired `#[cfg]` / `#[cfg(not)]` statements:**
  - `cache/job.rs:538-551` (the same call, written twice);
  - `state/runtime.rs` `build_runtime` (`new_with_hooks` vs `new`);
  - `db/mariadb.rs:113-116`;
  - `state/service.rs:283` `#[cfg(not(feature = "test-hooks"))] drop(repaired_job_ids);`, which only silences an unused variable. That variable is the return value of `repair_tree`, and only a hook consumes it.
- **Simplification that keeps the semantics.** Always compile `TestHooks`, `HookPoint` and `HookContext`. Without the feature, `TestHooks` becomes a zero-sized type whose `hit` returns `Ok(())` and whose `observe` does nothing.
  - Pass contexts as closures (`hit(point, || HookContext::round(..))`) so that production never builds the context's `Vec`s.
  - Every call site, the `_with_hooks`/`_impl`/`build` layers and the paired statements then collapse to one path.
  - Only `DbError::Injected`, `InternalError::InjectedFault` and `schema_ddl_statements_at_connect` would keep a `cfg`.
- **Why it matters.** The non-hook build is what ships, and today it is a different program text from the one the tests exercise.

### 2. Phantom generic parameters

- **`SharedJobControlBlock` / `JobControlBlock`.** They keep `DbConnectorType: InternalJobOrchestration` with `_db_connector: PhantomData<fn() -> DbConnectorType>` (`cache/job.rs:1, 909, 1160`). The JCB no longer holds a DB connector: job-level writes moved to the committer.
  - The parameter forces three-parameter types on `JobCache`, `SharedJobMap`, the GC and every helper signature.
  - Its docstring is now misleading: "The type of the DB-layer connector of the job's tree" (`cache/job.rs:59, 1143`).
  - Drop it from the JCB and from `JobCache`. `TreeEnv` already carries the real `db`. The review guide's rename rule applies: a wide mechanical edit is not a reason to defer.
- **`StagingPublisher<InboundQueueSenderType>`.** It is generic only through `_inner: PhantomData` (`state/publisher.rs:27-31, 96`). The wrapped sender is moved into the lane tasks and never stored.
  - Make the struct non-generic and make `spawn` generic.
  - `ServiceJobCache` and the `StagingPublisher<InboundQueueSenderType>` spelling in `ServiceState` (`state/service.rs:67-73`) then simplify too.

### 3. Dead trait surface: `InternalJobOrchestration::{start, set_state, commit_outputs, cancel, fail}`

- **No production caller.** Since the committer took over job-level writes, nothing in `src/` calls these five methods (`db/protocol.rs:274-362`). Their only users are `tests/fence_test.rs:56-125` and the pre-existing `tests/mariadb_test.rs`.
- **What they still cost.** The prototype re-implemented all five over a new `transition_one_job` plus `ensure_valid_transition` (`db/mariadb.rs:217-235, 422-490, 1414`). It also kept `fetch_job_state_for_update` alive for them, and `MockDbConnector` and `tests/scheduling_infra.rs::NoopDbConnector` must keep implementing them.
- **The rule.** The review guide's trait-surface rule says not to keep trait methods that the consuming layer never calls.
- **Design context.** Design §9.3 keeps them as wrappers "in PR 3 and deleted in PR 9". That staging fits a real PR series, not a prototype whose line count goes into the report. For the prototype, delete them and port the fence assertions in `fence_test.rs` to `apply_tree_transition`, which `tree_transition_test.rs:250` already partly covers. Either way, report the line count without them, so the prototype's count is comparable with the design's estimate.

### 4. CI does not run what the prototype relies on

- **`assert_invariants` never runs in CI.** It runs after every tree state transaction, but only under `#[cfg(debug_assertions)]` (`state/tree_committer.rs:37, 673, 801`). CI runs `cargo nextest run --all --all-features --run-ignored all --release` (`taskfiles/test.yaml:322`), and the workspace has no `[profile.release]` override, so the per-round invariant check is never exercised there.
  - Gate it on `any(test, feature = "test-hooks")`, or make it a cheap always-on check.
- **CI tests the unboxed future.** The `unboxed-round` feature exists for benchmark B6 (`Cargo.toml` `[features]`, `state/tree_committer.rs:510-513`). `--all-features` turns it on, so CI tests the unboxed round future and never the `Box::pin` path that ships.
  - Remove the feature once B6 is measured, or keep it out of `--all-features`, for example as a `cfg` flag the benchmark sets.
- **Related.** The integration-test modules are gated by `#[cfg(feature = "test-hooks")]` in `tests/test_spider_storage.rs:1-21`, so a plain `cargo test -p spider-storage` silently skips most of the T-tests. `required-features = ["test-hooks"]` on the `[[test]]` target would make the skip loud.

## Minor

### 5. `validate_session` lost its docstring (`state/service.rs:1164`)

Carried over from the slice reviewer's S6 issue 1 and still open: the base docstring ("Validates that the given `session_id` matches …" plus `# Errors` naming `StaleSession`) is gone.

### 6. Missing `# Type Parameters` sections

- `spawn_body<ReturnType>` (`state/service.rs:1329`).
- `forward<Entry, SendFuture>` (`state/publisher.rs:137`).
- The test helper `race<FirstType, SecondType>` (`tests/job_tree_race_test.rs:481`).

### 7. Positional `Vec` of join handles in `PartialRuntime`

- **What happens.** `join_handles: Vec<JoinHandle<..>>` is filled by `vec![pool]` and then `push(gc)`. It is emptied by two `pop()`s in reverse order, each with an `expect` (`state/runtime.rs:270-310, 329`).
- **The rule.** The coding style guide calls this out: a `Vec` indexed by position encodes ordering assumptions.
- **Fix.** Use named `Option<JoinHandle<..>>` fields, `task_instance_pool` and `job_cache_gc`.
- **Same pattern elsewhere.**
  - `StagingPublisher`'s lane channels carry positional tuples `(ResourceGroupId, JobId, Vec<TaskIndex>)` (`state/publisher.rs:28-30`). A small struct, or `Publication` plus a resource group, would name them.
  - `ReportTaskSuccessRequest::Unpacked` gained `Vec<(Vec<u8>, Vec<u8>)>` for the children (`spider-proto-rust/src/unpack/storage.rs:94-101`).

### 8. `impl UpdateStatement` does not follow its type (`db/mariadb.rs:1126-1140`)

`struct UpdateRow` sits between `struct UpdateStatement` and `impl UpdateStatement`. The review guide says an `impl` block must come immediately after its type.

### 9. Clippy fails on the current nightly (`test_hooks.rs:477`)

- **The error.** `AtomicUsize::fetch_update` is deprecated in `1.100.0-nightly (2026-08-22)` ("renamed to `try_update`"). So the lint task (`cargo +nightly clippy --workspace --all-targets --all-features -- -D warnings`) errors. `main` has no `fetch_update`.
- **Caveat.** Whether CI hits this depends on the nightly that `toolchains:rust` installs.
- **Fix.** `try_update`, or a plain `compare_exchange` loop.

### 10. The round-shaping rules are written twice

- **Two owners.** `take_events` (`state/tree_committer.rs:1086-1111`) already takes the oldest `CancelRoot` first, caps a round at `MAX_EVENTS_PER_ROUND`, and leaves every registering event after the first in the buffer. `plan_events` / `RoundOrder::new` (`cache/tree.rs:807-835, 1406-1449`) then enforce cancel-first, the cap (`position >= MAX_EVENTS_PER_ROUND`), and "one registration per round" (`extra_registrations`) again.
- **Dead in production.** With `take_events` in front, the planner's cap and extra-registration deferrals are unreachable. Only the planner unit tests and the property test reach them, because they feed arbitrary batches.
- **Fix.** Pick one owner.
  - Option A: `take_events` keeps only cancel-first and the cap, and the planner keeps the rest, including moving the registration last.
  - Option B: the planner `debug_assert!`s the shape that `take_events` guarantees.
- **Docstring.** `RoundOrder::new`, the function that encodes this order, has no docstring.

### 11. Two outcome enums that map 1:1, and a misleading variant name

- **The two enums.** `EventOutcome { Applied, Stale, Absent }` (`cache/tree.rs:553`) is converted variant by variant into `TreeOutcome { Applied, Stale, TreeClosed }` in `reply` (`state/tree_committer.rs:956-960`).
- **The misleading name.** `TreeClosed` is sent in two different cases:
  - when the committer has exited;
  - when the job merely left a still-live tree (`EventOutcome::Absent`).
- **Fix.** Use one enum, and name the variant for what it means, for example `Untracked`.

### 12. Wrappers that only the tests use

- **The wrappers.** `succeed_task_instance` exists at three layers, each forwarding to `_with_children` with `Vec::new()`:
  - `cache/task.rs:450`;
  - `cache/job.rs:479`;
  - `state/service.rs:566`.
- **Who calls them.** Production calls only the `_with_children` variants: `grpc.rs` calls the service's, the service calls the JCB's, and the JCB calls the TCB's through `_impl`. The wrappers serve the tests and `scheduling_infra`.
- **Fix.** Keep one `succeed_task_instance(.., children)` per layer, and pass `Vec::new()` in the tests.
- **Also.** `SharedJobControlBlock::collect_state` (`cache/job.rs:825`) is an alias of `state()`.

### 13. `ensure_uncached_root` swallows a DB error and doubles the cache-miss read

- **The swallowed error.** `Err(_) => Ok(())` (`state/service.rs:1092-1097`) is the "pattern-match-and-ignore" that the review guide flags. The docstring justifies it, but it costs a second round trip.
- **The double read.** On a cache miss, `get_job_state` reads `get_state_and_parent` and then `get_state` again (`state/service.rs:442-449`).
- **Fix.**
  - Return the state that `get_state_and_parent` already read.
  - For outputs and errors, propagate the probe's error instead of ignoring it.

### 14. The `ChildSetSummary` reservation is computed in two places

- **Reserve.** `prepare_report` reserves the summary of `children` and releases it on `InvalidChild` (`state/service.rs:1234-1318`).
- **Release.** The caller rebuilds the same summary from the prepared children, only to release it if the TCB rejects the report (`state/service.rs:658-667`).
- **Fix.** Return the reserved summary with the children, or use a small RAII reservation guard, so that reserve and release cannot disagree.

### 15. `Plan::merge` is bent to fit repair

- **Repair.** `Repair::merge` invents an event index from `self.plan.outcomes.len()` (`cache/tree.rs:978-983`), and `plan_repair` then clears the outcomes (`:961`).
- **Tree failure.** `plan_tree_failure` passes the root as "the event's job" with index 0, then clears the outcomes too (`:1806-1810`).
- **Fix.** A `Plan::merge_updates(event_plan)` that takes no event index, outcome or registration would remove both workarounds.

### 16. Smaller simplifications in the committer and planner

- **`write_plan` mutates the plan.** It takes `&mut Plan` and moves `plan.updates` out (`state/tree_committer.rs:734-735`), so `Plan::is_empty()` silently turns true for the rest of the round. Pass the updates by value.
- **Resource group lookup.** Each publication's resource group is looked up through `state.handle()` into a temporary map, before `update_structure` drops the handles (`state/tree_committer.rs:657-661`). The planner already knows every node's `resource_group_id`, so it could carry it in `Publication`.
- **`check_invariants` generic.** `check_invariants<Hasher: BuildHasher>` and `InvariantChecker<'_, Hasher>` (`cache/tree.rs:746, 987-992`) are generic only to satisfy `clippy::implicit_hasher` on a `pub` function. It is used only inside the crate, so make it `pub(crate)`, like `plan_events`, and drop the generic.
- **Limits split across two modules.** `MAX_CHILDREN_PER_JOB`, `MAX_JOB_CHILD_BYTES`, `MAX_ERROR_MESSAGE_CHARS`, `MAX_EVENTS_PER_ROUND` and `MAX_ROUND_PAYLOAD` are defined in `cache/tree.rs:32-46` and re-exported by `state/limits.rs`; the other limits are defined in `limits.rs`. The last two re-exports have no user. Give the limits one home.
- **`drop_prepared_children`.** It uses `drop(std::mem::take(&mut ..))` (`cache/task.rs:230-235`); `.clear()` says the same thing.
- **`ensure_root`.** `parent_id.map_or(Ok(()), |parent_id| Err(..))` (`state/service.rs` `ensure_root`) reads more plainly as an `if let Some(parent_id)` early return.

### 17. `let _ =` without justification

- **`cache/tree.rs:257`.** `submit` creates the receiver itself and still holds it, so this send cannot fail. Use an `expect`, or drop the `let _`.
- **`state/tree_committer.rs:961, 1123`.** Here the waiter may have dropped its future, so the discard is right, but the review guide asks for a one-line reason.

### 18. The test-only `fence_id` in the production `DatabaseConfig`

- **What it is.** `config.rs:47-54` is a `#[serde(skip)]` production field. Every deployment leaves it at 0; only tests set it, so that tests sharing one database don't fence each other. Design §9.1/§9.2 fixes the fence row at `id = 0`, and the prototype widened it to `BIGINT`.
- **Fix.** The tests already have `ScratchDatabase`. Give the tests that share a database their own scratch database, and keep the design's single row, or set the ID through a test-only constructor. The four-line docstring also goes beyond the contract (issue 21).

### 19. Test duplication and test-style issues

- **Duplicated helpers.** Three different `JobShape` structs, three `build_job`s and two `child`s are spread over `tests/job_tree_race_test.rs:44-150`, `tests/job_tree_recovery_test.rs:37-128` and `tests/registration_test.rs:41-160`. So are `serialized_output`, `start_root`/`start_job`, `instance`/`create_instance`, `run_commit`, `run_cleanup`, `children_of` and `assert_states`; `job_tree_test.rs` repeats some of them too.
  - `job_tree_infra.rs` exists for this.
  - The style guide's test-data rule asks for one base constant plus per-test overrides: one `JobShape` with a `BASE` constant and `..BASE` overrides.
- **Assertions on collections.** `assert!(ready.is_empty(), ..)` and `assert!(..prepared_children.is_empty())` (`cache/task.rs:2096, 2186-2187`) should be `assert_eq!(.., [])`, so that a failure prints the contents.
- **A `where` clause.** The `race` helper uses one (`tests/job_tree_race_test.rs:489-493`). Edition 2024 allows the bounds inline: `FirstType: Future<Output: Send + 'static> + Send + 'static`.
- **A test that asserts nothing.** `committer_future_size` (`state/tree_committer.rs:1547`) is a B6 probe that only prints. It belongs in the benchmark harness.
- **Property-test placement.**
  - The property module (`cache/tree.rs:2992-4156`) makes `tree.rs` 4,156 lines, about 2,300 of them tests. Move it to its own file.
  - The two property tests duplicate the parsing of `JOB_TREE_PROPERTY_ROUNDS` and `JOB_TREE_PROPERTY_SEED`.
  - `Coverage::max_rows` is printed but never asserted.

### 20. Docstring defects

- **Circular `# Returns`.** `build_runtime`'s `# Returns` says "Forwards [`create_runtime`]'s return values" (`state/runtime.rs:232`), while `create_runtime` forwards `build_runtime`. `create_runtime_with_hooks` (`:201`) has the same shape. Each should describe the tuple itself.
- **Broken reflow.** `fail_and_abort`'s docstring breaks a line after "characters. If the job's parent is" (`cache/tree.rs:1671-1673`).
- **Incomplete depth rule.** `TreeStructure::from_rows` says a node's depth is "its parent's depth plus 1 if the parent is among `rows`, else 1" (`cache/tree.rs:316-320`). It omits the root's depth of 0, which the code assigns.
- **Misleading type parameter doc.** The JCB type-parameter doc names a DB connector that the JCB no longer holds (issue 2).
- **Missing docstrings on non-trivial private functions:**
  - `RoundOrder::new` (`cache/tree.rs:1420`);
  - `HookFilter::matches` (`test_hooks.rs:165`, needs a `# Returns` "Whether …");
  - `UpdateStatement::has_outputs` (`db/mariadb.rs`);
  - `JobControlBlock::hook_context` (`cache/job.rs:1189`);
  - the planner test helper `error_of` (`cache/tree.rs:1991`).

### 21. Pleiades's comment rule: docstrings that explain the new feature's mechanism

The added inline comments are clean. Several new docstrings, though, go past the contract and explain how the new feature works, which the CLAUDE.md rule says to omit unless asked. The rationale belongs in the design doc. Trim these to one or two sentences each:
- `DatabaseConfig::fence_id` (`config.rs:47-53`);
- the second paragraph of `check_fence` (`db/mariadb.rs:168-169`);
- the second paragraphs of `upgrade_jobs_schema`, `bump_fence` and `kill_fence_blockers` (`db/mariadb.rs:1433-1436, 1574-1577, 1663-1667`);
- the "since the server discards the connection's session either way" clause of `close_detached_connection`;
- the first bullet of `Plan::admits`, "The producers of the events never send an event that would, so this only guards against a broken producer" (`cache/tree.rs:631-632`);
- the last sentence of `PoisonOnAbnormalExit` (`state/tree_committer.rs:363-364`);
- the new paragraph on `SharedJobControlBlock` (`cache/job.rs:53-54`).

### 22. Names

- **`PlannerInput::None`** (`cache/tree.rs:508`). It reads like `Option::None` at call sites such as `queued.input = Some(PlannerInput::None)`. Use `Empty`.
- **`CommitterState::handles`** (`state/tree_committer.rs:183`). It holds JCBs, so `jcbs`.
- **`TaskGraph::recover_blocked`** (`cache/task.rs` `recover_blocked`). It also builds the `CommitReady` graph, so something like `recover_without_regular_tasks`.
- **`UpdateStatement::plan`** (`db/mariadb.rs:1159`). It shares a word with the planner's `Plan`. It groups and chunks updates, so `group`.
- **`ValidatedJobSubmission::into_parts_with_compressed`** (`job_submission.rs:165`). It returns a root submission's compressed parts typed as `ChildJobSubmission`.
- **`ShutdownHandle::fatal()`** (`state/shutdown.rs`). An action named by an adjective; `shut_down_fatally()` reads better.

### 23. Leftovers

- `ChildJobSubmission` derives `Default` without a user (`job_submission.rs:28`).
- `Runtime::is_fatal` (`state/runtime.rs:145`) is public API that only the tests call.
- `SET SESSION wait_timeout = 900` (`db/mariadb.rs:75`) buries a design invariant in a string literal: it must stay above sqlx's 10-minute idle timeout (§11.3). Use a named seconds constant.

### 24. For Pleiades to decide: the "changes a job twice" deferral

`Plan::admits` defers an event that would update a row the round already updates (`cache/tree.rs:642-648`), and its own docstring says only a broken producer can trigger it. The coding style guide's "assert preconditions instead of tolerating them" points to a `debug_assert!`. But design §7.5 lists it as a deferral rule, so changing it changes the documented behavior. Flagged, not recommended either way.

## Not flagged, on purpose

These look heavy but are design-mandated, so I did not count them as needless complexity:
- the blocker-killing fence bump with its access-denied fallback (§11.3);
- `wait_timeout` in `after_connect` (§11.3);
- `persistent(false)` on `has_failed_child` (§9.3);
- `FORCE INDEX (PRIMARY)` and exact from-state guards (§9.4);
- the overlay;
- the payload budget with its reservation for the oldest registration (§7.5);
- the per-lane staging publisher (§7.6);
- the poison guard (§7.7);
- `spawn_body` for handler bodies (§10.10);
- the `PreparedChild` prebuild outside every lock (§6.2).

The removal of `InternalError::UnexpectedJobTermination` follows the rule that error variants left without a producer are deleted.
