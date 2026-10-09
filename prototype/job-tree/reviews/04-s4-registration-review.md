# Review: S4 `s4-registration` (slice reviewer)

I reviewed the S4 changes in `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f` plus the untracked files). Most of the slice is in `state/service.rs`, `state/tree_committer.rs`, `cache/job.rs`, `cache/task.rs`, `db/mariadb.rs`, `grpc.rs` and `bin/grpc_server.rs`. I checked them against:
- the slice note `slices/04-s4-registration.md`;
- design sections 4, 6.2, 7.1-7.5, 9.4, 10, 11.5 and 13;
- the S1-S3 reviews;
- the Rust review and style guides.

Paths below are relative to `components/spider-storage/`.

Verdict: **fix_needed**. There are three major issues, and all three are limit gaps that turn an in-limit workload into a crash loop of the whole service:
1. The new 32 MiB decode limit lets a task failure message exceed `max_allowed_packet`.
2. A job's stored outputs are bounded only per report, not per job.
3. The per-job child limits are checked only after every child is built and held in memory.

Everything else is minor: the culprit of a payload error can be misattributed, some coverage is missing, the prepared children are built on the async worker, and there are a few style points.

Housekeeping: the kind cluster was not started. `clp-rc2-control-plane` was already `Exited (137)` when I checked. I did not delete it with `kind delete cluster`, because that cannot be undone and the cluster is already stopped.

## What holds up

- **The flag ordering.**
  - `has_prepared_children.store(true, Relaxed)` happens before the reporting task's `AcqRel` `fetch_sub` (`cache/job.rs:493-501`). The last task loads the flag after its own `AcqRel` RMW (`cache/job.rs:523`).
  - The RMW chain is a release sequence, so every store of the flag happens before the last task's load.
  - The children themselves pass through the TCB mutex (`cache/task.rs:571`, `cache/task.rs:275-281`).
  - The `registers` mismatch check in `collect_inputs` (`state/tree_committer.rs:848-857`) cannot fire on a real sequence. I checked three paths: the `Stopping` requeue (the children stay in `Queued`), a deferral (the input is kept), and a non-`Running` job (`None`).
- **Exactly one DB transaction.**
  - The parent's `Running → BlockingOnChildren` UPDATE with outputs, the fence, and every INSERT chunk run between one BEGIN and one COMMIT (`db/mariadb.rs:349-389`).
  - The payload branch rolls back and leaves the parent's failure to the next round, as design §6.2 step 5 says. `apply_plan` and `bind_children` never run on that path.
- **Locks.**
  - `bind_children` reads only the parent's immutable fields (`cache/job.rs:124-145`) and takes no JCB lock.
  - `insert_many` takes the map lock with no JCB lock held.
  - Children are published only after `insert_many`, so no ready hint names an uncached child (`state/tree_committer.rs:619-646`).
  - `take_prepared_children` takes one TCB lock at a time, under the JCB read lock, the same order as phase A.
- **Stale vs. rejected.**
  - Validation runs before the TCB flip (`state/service.rs:556-583`).
  - A duplicate or stale instance stores nothing, because `base.succeed_task_instance` fails before `prepared_children` is assigned (`cache/task.rs:571-579`).
  - A rejected report of a stale instance reaches `fail_task_instance`, which answers stale. It is neither a crash nor a double failure.
- **Ordering.**
  - A cancel in the same round makes the registration `Stale`, and its children are dropped with the `Queued`.
  - `take_events` takes one registration per round, and the S3 reservation keeps the oldest registration from being starved (S3 issue 1 is fixed).
  - `fail_registration` keeps the reply sender, and `requeue_all` restores the round in arrival order. The mock test checks that a co-event is planned again and answered once.
- **Child IDs.** `child_inserts` and `bind_children` both index `events[registration.event_index].children`, in the same order. `insert_children` checks the count of returned IDs. The 1,200-row test checks the order across chunks.

## Issues

### 1. Major: the 32 MiB decode limit lets a task failure message exceed `max_allowed_packet`, which crash-loops the service

- **Where:**
  - `bin/grpc_server.rs:106-109` raises `max_decoding_message_size` to 32 MiB for the whole `TaskInstanceManagementService`, which includes `ReportTaskFailure`.
  - `cache/tree.rs:1476`: `fail_and_abort` binds the failing job's own message untruncated. Only the ancestors' copy is cut, to `MAX_PROPAGATED_MESSAGE_CHARS` (`cache/tree.rs:1487-1492`).
  - `db/mariadb.rs:1146` lets every statement carry at least one row. `error_message` is not counted in the statement payload or the round budget.
- **Scenario:**
  1. A task fails with a 20 MiB error message, such as a captured stderr. Before S4, tonic's default 4 MiB limit rejected that request. Now it is decoded.
  2. With retries exhausted, `TerminalFailure` plans `UPDATE … SET error_message = ?` with the 20 MiB bind. MariaDB 10.11's default `max_allowed_packet` is 16 MiB.
  3. MariaDB answers errno 1153 and drops the connection. The round has no registration, so the error is fatal (§11.5): the service exits non-zero.
  4. After the restart, the job is `Running` in the DB, the task runs again, fails with the same message, and the service crashes again. Every tree on the service goes down with it.
  5. If a registration happens to share the round, the innocent parent is failed first (issue 4), and the next round is fatal anyway.
- **Fix:**
  - Cap the stored failure message, for example at a few KiB, as the ancestor chain already does. Do it in `fail_task_instance` or at the gRPC boundary, for every lane.
  - Or apply the 32 MiB limit only where children are carried, for example as a separate service for `ReportTaskSuccess`. Tonic's limit is per service, not per method.
  - Add a test with a failure message larger than `max_allowed_packet`.

### 2. Major: a job's stored outputs are bounded only per report, so one row's UPDATE can exceed `max_allowed_packet`

- **Where:**
  - `state/service.rs:1079-1085` checks `MAX_REPORT_OUTPUT_BYTES`, but per report.
  - The job outputs are the union of the outputs of every output-producing task (`read_output_payloads`).
  - The round budget always admits the first payload event (`cache/tree.rs:576-579`).
  - `UpdateStatement::plan` puts at least one row in a statement (`db/mariadb.rs:1146-1147`).
- **Scenario:**
  1. A job has five output tasks, and each reports 3.9 MiB of outputs. Every report passes the 4 MiB check.
  2. The last task's `RegularTasksDone` plans one `Running → Succeeded`/`CommitReady`/`BlockingOnChildren` UPDATE that binds 19.5 MiB of serialized outputs.
  3. The result is 1153 with no registration in the round: fatal, then the same crash loop as issue 1 after the restart.
  4. On `main` the same job got a non-fatal `commit_outputs` error. The committer's fatal-on-any-`DbError` rule (§11.5) makes this a regression.
  5. The design's claim that "the limits make errno 1153 … unreachable" (§6.2 step 5) is false for outputs.
- **Fix:**
  - Add a per-job outputs limit that the planner enforces, failing the job the way `plan_regular_done` fails a child-limit violation (`cache/tree.rs:1316-1322`): "job outputs exceed …" instead of the row write. Make the limit at most `max_statement_payload`, or at least `max_allowed_packet` minus a margin.
  - Record this as a design deviation in the report.

### 3. Major: the per-job child limits are checked only after every report's children are decoded, built and held in memory

- **Where:**
  - `state/service.rs:1095-1120` checks count and bytes per report.
  - `cache/task.rs:575` keeps each report's `PreparedChild`s, each with a full built `TaskGraph` and copies of both compressed buffers, in the TCB until the last task finishes.
  - The union is checked against `MAX_CHILDREN_PER_JOB` and `MAX_JOB_CHILD_BYTES` only at planning (`cache/tree.rs:1316`).
- **Scenario:**
  1. A job has 200 regular tasks. Each one reports 10,000 small children, or 16 MiB of compressed children, which is within every per-report limit.
  2. The storage process builds and holds 2,000,000 prebuilt task graphs, or 3.2 GiB of compressed blobs plus their built graphs, before the last task's `RegularTasksDone` lets the planner say "child-job limit exceeded".
  3. The process is killed for running out of memory. After the restart, every task of the job reruns and reports the same children: a crash loop.
  4. T10 ("a limit violation … is a job failure, not a crash") holds only for a violation within one report.
- **Fix:**
  - Keep running per-job totals in `JobExecutionState`: an `AtomicUsize` for the count and an `AtomicU64` for the bytes. Add to them with `fetch_add` in phase A, before the children are stored.
  - Reject the report, or mark the job for the limit failure, as soon as a total would pass the per-job limit, so that at most one report's worth of children over the limit is ever held.
  - The design (§6.2 step 3) has the same gap; record it as a deviation.

### 4. Minor: the payload branch blames the registration for a payload error of any statement in the round

- **Where:** `db/mariadb.rs:369-372` maps errno 1153 or 1406 from *any* statement to `PayloadRejected`, including the other jobs' UPDATEs. `state/tree_committer.rs:621-628` then fails the round's registering parent.
- **Scenario:** with `max_allowed_packet` set to 8 MiB, job X's 10 MiB outputs (issue 2) and parent P's 1 MiB registration share one round, within the 16 MiB budget.
  1. X's UPDATE gets 1153.
  2. P is failed with "child payload rejected by the database" and the round is requeued.
  3. The next round hits X's 1153 again and is fatal.

  I found no sequence in which the wrong failure becomes durable:
  - the culprit is deterministic and is planned again;
  - a cancel in between makes P's failure stale too.

  Still, the log blames P, and design §6.2's claim that "the culprit is unambiguous" does not hold for this code.
- **Fix:** classify only the errors of `insert_children` as `PayloadRejected`. With issues 1 and 2 fixed, an UPDATE cannot be a payload culprit.

### 5. Minor (tests): the real errno classification is never exercised

- **Where:** `db/mariadb.rs:1807-1824`, `tests/registration_test.rs:615-641`, and `state/tree_committer.rs:1789-1851`.
- **Scenario:**
  - Both payload tests inject `DbError::Injected { deterministic_payload: true }` at `CommitterBeforeDbTransaction`.
  - A mutation that makes `is_payload_rejection` always return `false` passes every test.
  - So does mapping 1153 to `PayloadRejected` only after the rollback.
  - Whether a real 1153 even reaches the classifier as a `Database` error, rather than as an I/O error after MariaDB closes the connection, is unknown (deviation 8).
- **Fix:**
  - Add a unit test of `is_payload_rejection` on constructed MySQL errors.
  - Or add one MariaDB test that lowers `@@GLOBAL.max_allowed_packet` on the scratch container and registers a child just over the packet size (`max_statement_payload` is read at connect, so build the connector afterwards). Then record which error class comes back.

### 6. Minor (tests): two design-named registration cases are only unit-tested

- **Where:** `cache/tree.rs:1316` and `cache/tree.rs:2136-2150`. Design §15 names both cases.
- **What is missing:**
  - **Union over the limit.** "A registration whose children exceed `MAX_JOB_CHILD_BYTES` across two reports" is covered by the planner unit test only. No integration test runs two in-limit reports whose union breaks the limit. Such a test would check that the parent is `Failed` with "child-job limit exceeded", that there are no child rows, and that the prepared children are dropped.
  - **A co-event in the payload branch.** In `a_rejected_registration_payload_fails_the_parent_in_the_next_round`, the co-event is a `Start` that is already `Stale` in round 1. The interesting requeue is never exercised: a co-event of another job whose planned UPDATE was rolled back, collected again and re-planned. One example is a sibling's `RegularTasksDone` in a two-level tree.

### 7. Minor: child preparation runs CPU-bound on a runtime worker, with a copy of every blob

- **Where:** `state/service.rs:1121-1128` and `cache/task.rs:322-347`.
- **Scenario:**
  - One report with 10,000 children of 100 tasks each costs about 1.4 s of decode and build time at design B2's 143 µs per child. All of it runs synchronously inside the spawned body, on a tokio worker.
  - The committers on that worker stall for that long.
  - `PreparedChild::create` also clones both compressed buffers (deviation 5): up to 16 MiB of copies per report, on top of issue 3.
- **Fix:**
  - Run the decode (`ValidatedJobSubmission::create`) in `spawn_blocking`, or yield between children.
  - Add an accessor on `ValidatedJobSubmission` that returns the compressed buffers instead of cloning them, per the guide's "pass ownership instead of borrowing when the callee clones".

### 8. Minor: a rejected success report is logged as "Task instance succeeded." and answered OK

- **Where:** `state/service.rs:587-593`, and deviation 4.
- **Scenario:**
  - After the rejection, `fail_task_instance` returns `Running` or `Failed`, and the handler logs "Task instance succeeded." with that state.
  - The EM gets `Void`, so neither it nor an operator reading the success log can tell that the success became a failure. Only the earlier `warn!` line says so.
- **Fix:** log a distinct message on the rejection path. Also record in the report that the formal implementation needs a status for this case.

### 9. Minor (trap for S6): a recovered job is always a root at depth 0

- **Where:** `cache/job.rs:231-244`: `recover` passes `None` and depth `0` to `from_parts`.
- **Scenario:**
  - S6 recovers a child at depth 8 through this path.
  - Its JCB says depth 0, so `prepare_report` (`state/service.rs:1089`) lets it register children down to depth 16.
  - The structure's `child_depth` disagrees with the JCB's `depth`.
- **Fix:** S6 must pass `parent_id` and `depth` from the loaded rows.

### 10. Minor (style)

- `PreparedChild::has_commit` and `has_cleanup` (`cache/task.rs:299-300`) are never read. `bind_children` reads the JCB's flags instead. Remove them.
- The `Forwards` docstrings are circular and stale:
  - `SharedTaskControlBlock::succeed_task_instance` (`cache/task.rs:397-414`) still lists leaf errors, although it only delegates;
  - `succeed_task_instance_with_children`, `succeed_task_instance_with_hooks` and `succeed_task_instance_impl` (`cache/task.rs:428-434, 457-463, 548-554`) forward to `Self::succeed_task_instance`, which they do not call.

  Put the leaf errors on `succeed_task_instance_impl`, and forward to it.
- `ReportRejection::TooManyChildren` says "per job" (`state/error.rs:55`), but the check is per report (`state/service.rs:1095`).
- `TaskGraph::take_prepared_children` is `pub` (`cache/task.rs:275`). The only caller is in the same crate, so it can be `pub(crate)`, like the committer-only `collect_*` methods.
