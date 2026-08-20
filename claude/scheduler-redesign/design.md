# Resource-Group-Aware Scheduler Core

This document is the complete specification of the resource-group-aware scheduler core and its dispatch queue. It is self-contained: the admission policy and the correctness proof are stated here in full. Where it refers to existing behaviour it names the component in the current codebase.

## 1. Overview

The scheduler core is a single-threaded, tick-based loop that drains ready tasks from storage, decides which to dispatch, and publishes assignments into per-resource-group queues. Execution managers pull assignments over gRPC, either **pinned** to a resource group or **general** (no resource group named, may receive anything).

### 1.1 What this design enables

- **An almost lock-free tick-based core.** Assignments are made in a prefetching mode, so execution managers dispatch already-decided work asynchronously without taking any lock against the scheduler.
- **Correct handling of both dispatch modes.** A resource-group-dedicated execution manager dispatches only from its own group, while general execution managers serve every group with near-round-robin fairness — and every assignment remains reachable by a general execution manager, so a group whose dedicated managers are absent is still fully served.
- **Auto-balancing buffer occupancy.** The dispatch queue can never be filled by a single resource group. When a new group arrives it always finds room to publish assignments, without any capacity being statically reserved for it.
- **An improved job lifecycle.** Active → pending → retire replaces the current round-robin implementation's deferred retirement, giving a job several chances to be refilled before being dropped.
- **A core any runtime can spawn.** The core owns its scheduling state outright, in arenas rather than through shared handles, so the core future is `Send` and needs no dedicated thread with a `LocalSet` ([§2](#2-components)).
- **Room to grow.** The serialized design is sized for a small constant number of resource groups, which meets the CLP package integration requirement, and leaves per-group parallelization available should that change.

### 1.2 Structural properties

Two properties drive the structure:

- **An assignment is stored exactly once.** It lives in its resource group's queue and nowhere else, so exactly-once dispatch is structural rather than protocol-enforced. General execution managers are steered by a separate *broadcast queue* of hints carrying no payload.
- **Buffer space is accounted per resource group.** A single shared queue would converge to being owned by whichever group drains slowest, starving every other group of dispatch slots. Per-group queues with a threshold derived from live free space prevent that.

## 2. Components

| Component | Owner | Shared with service |
|---|---|---|
| Inbound queue | spider-storage | no (gRPC) |
| Global task set | core | no |
| Finalized job table | core | no |
| Job registry | core | no |
| `rg_table` (dispatch queue endpoints) | resource group registry | **yes**, via `Arc<DashMap<…>>` |
| `rg_units` (scheduling state) | core | no |
| Active RG list | core | no |
| Broadcast queue (hints) | core writes, service reads | **yes** |
| Reschedule queue | EM registry writes, core reads | — |
| Session manager | spider-execution-manager | **yes** |

Everything the dispatch service touches is `Send + Sync`. Everything else is core-private, and the core **owns it outright**: job entries live in a generational arena ([§3.3](#33-job-registry)) and resource group scheduling units in an append-only `Vec` ([§3.4](#34-resource-group-registry)). Scheduling positions — an RG's active job list, its pending job queue, the per-tick downgrade buffer, the active RG list — hold keys and indices into those arenas, never co-owning handles.

The core is single-threaded by construction, but nothing in it is single-threaded by *type*: there is no `Rc` and no `RefCell` anywhere in the scheduling structures. The core future is therefore `Send`, and the existing `spider-scheduler` runtime spawns it like any other core, with no dedicated thread and no `LocalSet`. That is the point of the arena ownership model; the properties below are what it costs and what it buys.

## 3. Data structures

### 3.1 Global task set

`HashSet<(JobId, TaskId)>`. Deduplicates tasks read from the inbound queue and the reschedule queue.

An entry is inserted when a task enters the job registry and **removed when its assignment is published to a dispatch queue**, or when the job entry still holding it is removed from the registry. The set therefore tracks what is *currently buffered in the core*, not what has ever been seen — which is what allows a rescheduled assignment to be re-admitted after the execution manager that held it was lost.

The second removal path is why the job registry hands a removed entry back **by value** ([§3.3](#33-job-registry)): the entry's remaining `ready_tasks` are drained straight out of the set in the same operation, with no borrow to release first.

### 3.2 Finalized job table

`HashSet<JobId>`. Records jobs that have reached commit-ready or cleanup-ready.

Consulted when upserting **regular** tasks into the job registry: a regular task belonging to a finalized job is dropped rather than scheduled. Storage and the reschedule queue can both deliver regular tasks for a job that has since finalized, and without this filter they would be scheduled and wasted.

### 3.3 Job registry

A generational arena of job entries, plus a secondary index from job ID to key.

```rust
slotmap::new_key_type! { pub struct JobKey; }

struct JobRegistry {
    jobs:      SlotMap<JobKey, JobEntry>,
    by_job_id: HashMap<JobId, JobKey>,
}

struct JobEntry {
    job_id: JobId,
    rg_id: ResourceGroupId,
    ready_tasks: VecDeque<TaskIndex>,   // no duplicates
    downgrade_counter: u32,
}
```

The registry is the sole owner of every entry. A scheduling position — an RG's active job list, its pending job queue, or the per-tick downgrade buffer — holds a `JobKey` and resolves it against the arena when it needs the entry. Exactly one position holds any given key ([§8.3](#83-other-invariants)); `by_job_id` is the registry's own index, not a second position.

**Methods on `JobEntry`:**

- `insert_tasks(task_indices: Vec<TaskIndex>)` — appends ready tasks and resets `downgrade_counter` to `DOWNGRADE_LIVES`.
- `get_next_task() -> Option<TaskIndex>` — the next ready task, or `None` when the queue is empty.

**Methods on the registry:**

- `upsert(job_id, rg_id, task_indices) -> UpsertOutcome` — `Exist` or `New(JobKey)`.
- `get_mut(key) -> Option<&mut JobEntry>` — `None` means the job is gone.
- `remove_by_job_id(job_id) -> Option<JobEntry>` — used by step 2 to finalize.
- `remove(key) -> Option<JobEntry>` — used by step 5 to retire.
- `clear()`

Both removals drop the `by_job_id` index entry and return the `JobEntry` **by value**, so the caller drains the returned entry's remaining `ready_tasks` straight out of the global task set ([§3.1](#31-global-task-set)) without holding a borrow on the registry.

`downgrade_counter` governs retirement and is described in [§5.5](#55-promotion-downgrade-and-retirement). `DOWNGRADE_LIVES` is a compile-time constant, currently `1`.

#### Why there is no `finalized` flag

Earlier revisions carried a `finalized: Cell<bool>` on each shared entry, deliberately placed outside the entry's `RefCell` so that setting and reading it could not panic against a live borrow guard. Three mechanisms — the flag, the entry's removal from the registry, and `get_next_task`'s error variant — all encoded the same fact. The arena collapses them into one: **the key no longer resolves**.

Every use of the flag was exactly that statement.

- Step 2 marked an entry finalized and removed it from the registry. It now calls `remove_by_job_id`, which is the whole operation; there is nothing left to mark.
- Step 4's regular-task path tested the active job for finalization before scheduling from it. It now calls `get_mut` and treats `None` as the finalized signal.
- Promotion scanned `pending_jobs` discarding finalized jobs. Same: a key that fails to resolve is a job that is gone, and is dropped from the queue.
- `get_next_task` returned `Err` for a finalized job so no further regular task could be dispatched. A caller that cannot obtain the entry cannot call it at all, so the variant has nothing left to report and the method returns a plain `Option`.

This is why job entries need **generational** keys rather than plain indices. A scheduling position is *expected* to hold a key to an entry that step 2 has already removed, and dereferencing it afterwards is how the position discovers the removal — the stale dereference is the mechanism, not a bug to be avoided. With a plain index the arena would be free to hand the same slot to the next job created, and that scan would then read a live but unrelated entry: it would schedule tasks under another job's position and swap the wrong entry out of the active list, silently and with no failure to observe. A generation makes the same lookup return `None`, which is the answer the scanner wanted.

### 3.4 Resource group registry

Split into a shared part and a core-private part. The shared part is the only thing that crosses a thread boundary; the core-private part never leaves the core.

**Shared — `rg_table: Arc<DashMap<ResourceGroupId, RgDispatchQueueEndpoints>>`**

```rust
struct RgDispatchQueueEndpoints {
    sender: async_channel::Sender<TaskAssignment>,
    reader: RgDispatchQueueReader,       // cloneable, Arc-backed
}

#[derive(Clone)]
struct RgDispatchQueueReader {
    inner: Arc<RgDispatchQueueReaderInner>,
}

struct RgDispatchQueueReaderInner {
    receiver: async_channel::Receiver<TaskAssignment>,
    living_hint: AtomicUsize,
    rg_id: ResourceGroupId,
    session_id: SessionId,               // session in which this RG was created
}

struct RgDispatchQueueWriter {
    sender: async_channel::Sender<TaskAssignment>,
    reader: RgDispatchQueueReader,
}
```

The two sides of a group's queue are two types. The **reader** is the read side and the only path that may take from the queue: it reports `rg_id` and `session_id`, blocks a pinned execution manager on `recv_pinned`, and serves a general one with `consume_hint_and_try_recv`. The **writer** is the write side — `try_send` to publish an assignment, `queue_len` for the group's current occupancy `S`, `living_hint` for the count `H` **by value**, `increment_living_hint` and `decrement_living_hint`, and `hint`, which returns the reader clone the broadcast queue carries ([§3.5](#35-broadcast-queue)). The writer carries no `Arc` of its own: exactly one scheduling unit owns a group's writer, and the reader inside it is already the shared handle.

**The writer holds a reader clone, not a bare counter handle.** The counter and the sender are two halves of one group's queue, and handing them to the scheduling unit separately would make a mismatched pair constructible — a sender for one group next to a counter for another, which nothing in the type system rejects and which would publish assignments into one queue while hinting for a different one. Pairing them inside the writer fixes the correspondence at construction, and the unit never sees the two separately. The writer reaches the counter through its reader's private `inner`, which is legal because both types live in the same module: the reader exposes no accessor to anyone, and the writer is not "anyone".

**The read side has no way to observe or modify the counter other than `consume_hint_and_try_recv`.** The reader publishes neither the count nor a handle to it, so "decrement, then attempt the pop" is not a convention a call site is trusted to follow but the only shape the read side can express. This is a design property, not an implementation detail: the requirements of [§8.2](#82-requirements-the-proof-depends-on) that constrain the consumer — the decrement being immediate, and paired with exactly one pop attempt — hold for every caller because there is no second way to touch `H`.

Both endpoints live in the table because **either side may create a resource group first**. A pinned execution manager can connect before any task for its group has been scheduled, so the service creates the channel pair on demand and blocks on the reader. When the core first schedules into that group it performs the same create-or-get and takes the group's writer into its scheduling unit.

Methods on the table: `get_dispatch_queue_reader(rg_id)` (create if absent, used by the service), `get_or_create(rg_id)` (used by the core), `clear()`. `RgDispatchQueueEndpoints::writer()` packs the entry's sender together with a clone of *that entry's* reader; it is the only way a writer is built.

**Core-private:**

```rust
rg_units:       Vec<RgSchedulingUnit>,            // append-only within a session
rg_index:       HashMap<ResourceGroupId, usize>,  // consulted only on group creation
active_rg_list: Vec<usize>,
last_served_rg: Option<ResourceGroupId>,
```

**Append-only invariant.** Within a session, `rg_units` is only ever pushed to. A resource group is never removed individually; the only removal is a wholesale flush on session bump ([§9](#9-session-bump)). Positions in `rg_units` are therefore stable for the lifetime of a session, `rg_index` is consulted only when a group is created, and the decision loop indexes straight into `rg_units` without a hash lookup per decision.

That is also why a plain `Vec` is the right structure here while job entries need a generational arena. The hazard generations guard is index reuse after an individual removal — a stale index resolving to a *different* occupant of the same slot. `rg_units` never removes an individual element, so no index is ever stale while the `Vec` is live, and the only event that invalidates indices invalidates all of them at once. Generational keys would guard a hazard that cannot arise, and would obscure the invariant that the `Vec` type already states. The obligation this pushes onto the session bump is recorded in [§9](#9-session-bump).

```rust
struct RgSchedulingUnit {
    rg_id: ResourceGroupId,
    active_jobs: Vec<JobKey>,                // capacity: active_job_list_capacity
    pending_jobs: VecDeque<JobKey>,
    finalize_queue: VecDeque<(JobId, FinalizeKind)>,
    writer: RgDispatchQueueWriter,
    rr_arm: usize,                           // index into active_jobs
    is_active: bool,
}

enum FinalizeKind { Commit, Cleanup }
```

`finalize_queue` carries the kind because commit and cleanup dispatch different task IDs.

### 3.5 Broadcast queue

An **unbounded** `async_channel` of `RgDispatchQueueReader`.

Every element is a **hint**, and a hint names a resource group and nothing else: it is what tells a general execution manager which group to try next, so that general capacity rotates over the groups rather than favouring whichever one it happened to look at. Hints are the only mechanism steering general execution managers; a pinned one never consults the broadcast queue at all.

A hint is an invitation, not a promise, and it is worth being explicit about how weak a claim it makes:

- **A hint does not guarantee an entry.** By the time its holder reaches the named group's queue, a pinned execution manager may have taken the assignment the hint was published for. The holder then finds the queue empty, discards the hint, and moves on to the next one. §8.1 shows this costs nothing: a hint is consumed either way, and coverage is maintained over the group's whole queue rather than over any individual assignment.
- **A hint does not name an entry.** It says only "try this group". If the group's queue holds several assignments, which one the holder receives is whatever the channel yields — the hint carries no claim on a particular assignment, and two holders of hints for the same group take different assignments without coordinating.

The **living hint** counter, `living_hint`, is the per-group half of this structure: an atomic count of how many hints naming that group are currently outstanding in the broadcast queue. The core increments it when it publishes a hint and a general execution manager decrements it when it takes one, so the count is what the publishing rule of [§5.4](#54-step-4--dispatch-queue-filling) consults to decide whether another hint is needed. That rule caps each group's outstanding hints at that group's peak queue occupancy ([§8.1](#81-coverage)).

The counter is a plain `AtomicUsize` inside `RgDispatchQueueReaderInner` ([§3.4](#34-resource-group-registry)), not a separately shared handle. The reader's `Arc` is the one sharing mechanism the group's queue has, and the writer reaches the counter through the reader clone it holds — which is why a hint and the counter it accounts for can never name different groups.

The queue must be unbounded because those peaks are not simultaneous: their sum can exceed `dispatch_queue_capacity`, and a rejected hint send would break the coverage invariant with no way to detect or repair it.

Sending a hint clones an `Arc`, not a channel receiver.

### 3.6 Reschedule queue

An unbounded channel written by the EM registry when an execution manager is lost or times out, drained by the core in step 1. Entries are filtered identically to inbound entries: assignments carrying a stale session are dropped, and regular tasks for finalized jobs are dropped.

### 3.7 Session manager

The global session manager from spider-execution-manager. Provides `bump()` and `current()`.

## 4. Tick overview

Each tick runs five steps in order. Steps 2 and 3 are skipped when polling results are not ready; **steps 4 and 5 always run**, so the dispatch queue continues to be refilled from already-buffered tasks while a storage poll is in flight.

## 5. The tick

### 5.1 Step 1 — collect polling results

The core drains the inbound queue's three lanes (ready, commit-ready, cleanup-ready) through separate gRPC calls, and drains the reschedule queue. A background coroutine assembles the results into:

- **Commit-ready** — `Vec<(ResourceGroupId, JobId)>`
- **Cleanup-ready** — `Vec<(ResourceGroupId, JobId)>`
- **Regular** — `Vec<(ResourceGroupId, JobId, Vec<TaskIndex>)>`, grouped by job, sorted and deduplicated within the batch

Results are ready only when all three gRPC calls have returned. The reschedule queue is then drained into the same form; it is empty in the overwhelming majority of ticks.

Each gRPC response carries the storage session ID. **Session bump detection happens here**, and the bump is applied at the end of this step ([§9](#9-session-bump)). Every later step is written as if no bump had occurred — no step-2-through-5 code path branches on it.

When results are ready, the core issues the next round of polls in the background, with per-lane fetch counts derived from remaining buffer capacity as in the current round-robin implementation.

When results are not ready, steps 2 and 3 are skipped and the tick proceeds to step 4.

### 5.2 Step 2 — process polling results

Build an empty map `rg_updates: HashMap<ResourceGroupId, RgUpdate>` where

```rust
struct RgUpdate {
    finalized: Vec<(JobId, FinalizeKind)>,
    new_jobs:  Vec<JobKey>,
}
```

**Commit-ready and cleanup-ready are processed first**, so that a regular task arriving in the same batch as its job's finalization is correctly discarded:

- Skip if already in the finalized job table.
- Insert into the finalized job table.
- `remove_by_job_id` the job from the job registry. Drain the returned entry's `ready_tasks` out of the global task set: those tasks were admitted but will never be published, and leaving them in the set would block their re-admission for good. Whatever scheduling position still holds the entry's key learns of the removal the next time it resolves it.
- Append `(job_id, kind)` to `rg_updates[rg_id].finalized`.

**Regular tasks**, batched by job:

- Skip the whole batch if the job is in the finalized job table.
- For each task, test the global task set; remove already-present tasks from the batch by swapping with the last element. Order within the batch need not be preserved. Insert the survivors into the global task set.
- Skip the batch if it is now empty.
- `upsert` the survivors into the job registry. If the outcome is `New`, append the key to `rg_updates[rg_id].new_jobs`.

### 5.3 Step 3 — apply updates to the RG registry

For each entry in `rg_updates`:

- Look the group up in `rg_index`. If absent, create the scheduling unit (via `rg_table.get_or_create`, taking the group's writer), push it onto `rg_units`, and record its index in `rg_index`. This is the only place either is written.
- Append the finalized jobs to `finalize_queue`.
- Place each new job key: append to `active_jobs` if it is below `active_job_list_capacity`, otherwise push to the back of `pending_jobs`.
- If `is_active` is false, set it and append the unit's index to `active_rg_list`.

Only newly created job entries need placement: an entry that already existed is by construction already held by an active resource group, in either its active list or its pending queue.

### 5.4 Step 4 — dispatch queue filling

This is the scheduling policy. The active RG list is **not** modified during the step; a per-tick copy `rg_rr_list` is used, and deactivations are buffered and applied at the end.

**a. Set up.** Destructure the core's fields — `rg_units`, the job registry, the global task set, the broadcast queue writer — into separate `&mut` bindings, so the step's inner operations borrow disjoint parts of the core rather than the whole of it. See [`try_make_assignment`](#try_make_assignment) below. Create `jobs_to_retire: Vec<JobKey>`.

**b. Single pass over `active_rg_list`** doing three things at once:

- copy each unit's index into `rg_rr_list`;
- accumulate `sum(dispatch_queue_size)` to compute `F = dispatch_queue_capacity - sum`;
- record the index `k` of `last_served_rg`.

Set `arm = (k + 1) % rg_rr_list.len()`, or `0` if `last_served_rg` is absent from the list. Rotating the arm rather than the list costs nothing and prevents the same resource group from always being visited first — which matters because `F` shrinks as the tick proceeds, so the first group visited sees the largest threshold.

`F` is an upper bound on the number of assignments this tick may publish. The queues drain concurrently, so the true free space only grows; bounding by the value measured at the start of the tick is what makes the loop terminate.

**c. Promotion pass.** For each unit, top up `active_jobs` to `active_job_list_capacity` from `pending_jobs` following [§5.5](#55-promotion-downgrade-and-retirement).

**d. Round-robin loop.** Terminates when `F == 0` or `rg_rr_list` is empty.

```
loop {
    if F == 0 || rg_rr_list.is_empty() { break }
    let unit = &mut rg_units[rg_rr_list[arm]];
    match unit.try_make_assignment(&mut F, job_registry, &broadcast_writer) {
        Ok((job_id, task_id)) => {
            global_task_set.remove(&(job_id, task_id));
            F -= 1;
            last_served_rg = Some(unit.rg_id);
            arm = (arm + 1) % rg_rr_list.len();
        }
        Err(_) => {
            rg_rr_list.swap_remove(arm);
            if arm == rg_rr_list.len() { arm = 0 }
            // arm is NOT incremented: swap_remove moved the tail element into
            // this slot, and incrementing would skip it.
        }
    }
}
```

**e. Apply buffered state.** Process the downgrade buffer ([§5.5](#55-promotion-downgrade-and-retirement)), then for every unit that reported `NoTask`, deactivate it if and only if it has **no schedulable tasks and an empty dispatch queue** — clear `is_active` and swap-remove it from `active_rg_list`.

The empty-queue condition is required for correctness, not tidiness: `F` is computed over `active_rg_list` only, so a deactivated group still holding assignments would make its occupancy invisible and let the core over-admit. The cost is one wasted visit per tick until such a group's queue drains.

Note the two removals are distinct. Removal from `rg_rr_list` means "nothing more from this group this tick" and is triggered by any `Err`. Deactivation is persistent and is decided only here.

#### `try_make_assignment`

A method on `RgSchedulingUnit`, taking `&mut F`, `&mut JobRegistry`, and the broadcast queue writer as parameters. It needs `&mut RgSchedulingUnit` and `&mut` the job arena **at the same time**: it reads and mutates the unit's `active_jobs` and `rr_arm` while resolving the keys they hold into entries it pops tasks from.

That is why it is a method on the unit and not on the core. `self.rg_units[i].try_make_assignment(&mut self.job_registry, …)` reached through a method on the core's `&mut self` is rejected — the method call borrows all of `self`, so the argument cannot borrow a field of it. The caller therefore destructures the core's fields first ([step a](#54-step-4--dispatch-queue-filling)) and passes two disjoint `&mut` bindings, which the borrow checker accepts because they name different fields. This is checked statically: there is no runtime borrow to fail.

1. If `finalize_queue`, `active_jobs`, and `pending_jobs` are all empty, return `Err(NoTask)`.
2. If the group's own dispatch queue size `S >= F`, return `Err(DispatchQueueFull)`. This is the dynamic threshold with α = 1 ([§6](#6-admission-policy)).
3. **If `finalize_queue` is non-empty, pop it** and dispatch that finalization task.
4. Otherwise take a regular task:
   - Resolve the key of the active job at `rr_arm`. If it does not resolve, the job has been removed — swap it out for the next key in `pending_jobs` that does resolve.
   - Call `get_next_task()` on the resolved entry. If it yields nothing, decrement that job's `downgrade_counter`; if the counter reaches zero, buffer the job for downgrade and swap in the next pending job, otherwise advance `rr_arm` to the next active job.
5. Publish the resulting assignment through the unit's writer, **in this order**:
   - `writer.try_send(assignment)` — into the group's own queue first, so a pinned execution manager can take it immediately;
   - then compare `writer.living_hint()` `H` against `writer.queue_len()` `S`. If `H >= S`, done. Otherwise `writer.increment_living_hint()` and send `writer.hint()` — a reader clone — to the broadcast queue. Should that send ever fail, `writer.decrement_living_hint()` rolls the count back; the queue is unbounded, so it cannot fail for want of capacity ([§8.2](#82-requirements-the-proof-depends-on)), and the rollback exists so that a failure which is not supposed to happen cannot leave `H` overstating coverage.

The ordering in step 5 is normative; reversing it lets a general execution manager consume a hint for an assignment that is not yet visible. See [§8](#8-invariants).

Any operation that leaves the group with nothing further to schedule returns `Err(NoTask)`.

#### Finalization priority

Finalization tasks are dispatched ahead of regular tasks whenever present, rather than alternating with them. Cross-group fairness is provided entirely by the outer quantum-1 rotation over `rg_rr_list` — each group receives exactly one assignment per outer cycle regardless of what it chooses internally — so the inner arm only controls the mix *within* a group, and there is nothing to protect by alternating.

Prioritising is preferable there because a finalization task completes a job and releases its state, a cleanup task stops a cancelled job from wasting further resources, and each job contributes at most one. Starvation of regular tasks is self-limiting: the finalize queue is fed by job completions, which require regular tasks to have been dispatched, so if regular dispatch stalls the finalization rate falls to zero on its own. A finalized job also contributes no regular tasks, having been removed from the job registry.

### 5.5 Promotion, downgrade, and retirement

Promotion pops `pending_jobs` until it finds a job with at least one schedulable task. A key that fails to resolve is discarded outright: the job has been removed from the registry, and the popped key was the last thing referring to it.

Downgrades are buffered and applied once, at the end of step 4:

- **Active → pending.** Re-inserted at the **head** of `pending_jobs` with `downgrade_counter` reset to `DOWNGRADE_LIVES`, so a job that is refilled promptly returns to active quickly.
- **Promotion failed, `downgrade_counter != 0`.** Decrement and append to the **back** of `pending_jobs`.
- **Promotion failed, `downgrade_counter == 0`.** Buffer in `jobs_to_retire`.

With `DOWNGRADE_LIVES = 1` a job that stops producing tasks moves active → pending head → pending back → retired, and any arriving task resets the counter and restores it.

A resource group may therefore exhaust its active and pending jobs within a tick; its pending queue refills before the next one. If it ends the tick with no active jobs, no pending jobs, no finalization tasks, and an empty dispatch queue, it deactivates.

### 5.6 Step 5 — job retirement

Remove every key in `jobs_to_retire` from the job registry. Batching retirement here rather than inside the decision loop keeps the loop free of registry mutation.

Retirement buffers keys rather than job IDs. A retired job is one that stopped producing tasks, so its `ready_tasks` is empty and there is nothing to drain out of the global task set — but the key still matters: a job that was removed and re-created between the buffering and the removal is a *different* job, and a stale key declines to remove it where a job ID would have removed it by mistake.

## 6. Admission policy

### 6.1 The problem

A single bounded buffer shared by groups that drain at different rates always ends up owned by the **slowest** drainer: the fast group's entries leave and the slow group's do not. No admission *order* fixes this — round-robin admission is already fair; it is buffer *residency* that is unfair. The remedy is to bound each group's residency by a threshold that adapts to how contested the buffer currently is.

Static partitioning (`B/N` each) would do that, but it wastes the buffer whenever a group is idle and must be recomputed whenever the group count changes.

### 6.2 The rule

This is the dynamic threshold scheme (Choudhury & Hahne, *Dynamic Queue Length Thresholds for Shared-Memory Packet Switches*, IEEE/ACM ToN, 1998), the same mechanism shared-buffer switch ASICs implement:

> A group may be admitted more work only while its own queue occupancy is below `α × F`, where `F` is the buffer's **current free space**.

The threshold is identical for every group and is recomputed from live state on every check. The group count never appears in it.

With `N` groups all backlogged, each sits at the threshold `T`, so `N·T + F = B` and `T = α·F` give

```
F = B / (1 + Nα)          T = αB / (1 + Nα)
```

At α = 1 this is `B/(N+1)` per backlogged group, with `B/(N+1)` left free. The reserve is the point, not waste: it is what lets a newly-active group be admitted immediately instead of waiting for incumbents to drain.

The scheme adapts in both directions with no coordination. A group that drains or goes idle raises `F`, which raises `T` for everyone else. A new group filling up lowers `F`, so incumbents stop being admitted and shrink toward the new `T` as they drain. Crucially, a group whose execution managers keep up never approaches the threshold at all and is therefore always admissible; only a group that is *accumulating* gets clamped.

### 6.3 Recorded decisions

**Sharing coefficient α = 1**, fixed, not configurable. α controls only how much headroom is reserved for groups that are not yet backlogged, and the trade is sharply asymmetric. Going from α = 1 to α = 2 multiplies per-group depth by `2(1+N)/(1+2N)` and headroom by `(1+N)/(1+2N)` — at `N = 7` that is **+6% depth for −47% headroom**, because per-group share saturates at `1/N` while headroom keeps shrinking. Going the other way, α = ½ buys headroom at a real cost in buffer utilization (77% vs 87.5% at `N = 7`). α = 1 is the balance point.

**No per-group ceiling.** With a single active group, that group may occupy half the dispatch queue. Accepted deliberately: the expected group count is small (≤ 4 for the CLP integration), and a lone group holding half the buffer starves nobody.

**No push-out.** The threshold prevents a group from *growing* past its limit, but a group shrinks only by draining — so a group that stops draining entirely freezes at whatever occupancy it had reached and is never reclaimed. Accepted for the same reason. If the group count grows, the extensions are a static per-group ceiling `C` (which binds in the low-`N` regime where the dynamic threshold is permissive, while the threshold binds at high `N`) and push-out of a dormant group's queue tail back into its job queues.

### 6.4 Implementation requirements

Two properties are required for the threshold comparison to mean anything:

**`F` is recomputed on every decision**, by decrement, not once per tick. Otherwise every group's threshold is measured against the same starting free space and a single tick admits `N × T`, overshooting the equilibrium.

**Admission is interleaved across groups at quantum 1.** The equilibrium is a fixed point that groups reach by creeping toward a falling threshold; it presumes interleaved arrivals, which a switch gets from the wire and a tick-based producer does not. Filling one group to its threshold before considering the next produces a staircase, not an equilibrium. With `B = 256`, α = 1 and five backlogged groups, batch-fill per group yields `64, 64, 64, 64, 0` and zero free space, while quantum-1 rotation yields `42, 42, 42, 42, 42` with 41 free. Every threshold in the batch case was read correctly at the moment it was read; the outcome is still wrong, because a threshold is only meaningful when all groups are measured against the same buffer state.

The rotation in [§5.4](#54-step-4--dispatch-queue-filling) satisfies both.

## 7. Task dispatching

### 7.1 Pinned execution manager

The service looks up the group in `rg_table`, creating it if absent, and clones the `RgDispatchQueueReader`. It then blocks on `recv_pinned` until an assignment arrives or the request's wait time expires.

The reader is re-fetched per request, so a session bump that replaces `rg_table` is picked up on the next call.

### 7.2 General execution manager

The service blocks on the broadcast queue. On receiving an `RgDispatchQueueReader` it calls `consume_hint_and_try_recv`, which decrements `living_hint` — **immediately, with no await point between the receive and the decrement** — and then attempts one `try_recv` on the group's queue. On success the assignment is returned. On empty the hint was stale: the service moves on to the next hint.

The decrement and the pop attempt are one method rather than two steps at the call site because that is the only counter access the read side has ([§3.4](#34-resource-group-registry)); a caller cannot decrement twice, decrement without attempting, or await in between.

No republication is required. The core's per-admission check alone maintains coverage.

### 7.3 Session check and EM registry

Both paths verify that the assignment's session matches the current session before returning, and retry otherwise. Dispatched assignments are recorded in the EM registry exactly as in the current implementation, which is what feeds the reschedule queue when an execution manager is lost.

## 8. Invariants

### 8.1 Coverage

For a single resource group, write `H` for its `living_hint`, `S` for the size reported by its dispatch queue, and `q` for the number of assignments physically in that queue. Let `m` be the number of general execution managers that have executed `H -= 1` but have not yet completed their pop attempt — `m` increments at exactly the instant `H` decrements, so the two are one logical event.

Define **coverage** `C = H + m`. This is the number of pop attempts still owed to the group: each outstanding hint will eventually produce one, and each in-flight execution manager is already committed to one.

> **Invariant.** `C ≥ q`

Every queued assignment is covered by either an outstanding hint or an in-flight general execution manager, so a group with no pinned execution managers is still fully served.

**Proof.** By induction over operations; initially `H = m = q = 0`.

| Operation | ΔC | Δq | Preserves the invariant |
|---|---|---|---|
| Core: publish + hint check | `+1` iff `H < S` | `+1` | see below |
| General: `H -= 1` / `m += 1` | `0` | `0` | trivially |
| General: attempt succeeds | `−1` | `−1` | trivially |
| General: attempt finds empty | `−1` | `0` | `Empty` *means* `q = 0`, and `C ≥ m ≥ 1` before the step |
| Pinned: pop | `0` | `−1` | trivially |

The publish case, writing `q₁ = q₀ + 1`:

- **Check fires** (`H < S`): `C₁ = C₀ + 1 ≥ q₀ + 1 = q₁`, by the induction hypothesis.
- **Check does not fire** (`H ≥ S`): `C₀ ≥ H ≥ S ≥ q₁`, so no hint is needed.

The last step uses `S ≥ q`, which holds because the assignment is sent to the queue *before* `S` is read, and any concurrent pop that lowers `S` below `q₁` belongs to a consumer that has already lowered `q` correspondingly. ∎

**One hint per assignment is sufficient.** The only operation that increases `q` is a publish, by exactly one, and the check runs immediately after every publish — so the deficit `q − C` never exceeds 1 at a check point. The gap between `S` and `H` may be arbitrarily large, but that gap is not a deficit: it is accounted for entirely by `m`, because a general execution manager entering flight leaves `C` unchanged. Every "missing" hint belongs to an execution manager already committed to taking an assignment.

**The check never under-publishes.** The implementation tests `H < S`, not `C < q`. Since `m ≥ 0` and `S ≥ q`, `C < q ⟹ H ≤ C < q ≤ S ⟹ H < S`, so whenever a hint is required the check fires. The converse does not hold, so it occasionally publishes when it need not — over-publication, which costs a bounded number of stale hints and no correctness.

**The broadcast queue needs no garbage collection, but it cannot be statically bounded.** Immediately after any check, `H ≤ S`, and `H` increases only via that check, so `H` never exceeds that group's *peak* queue occupancy. Each group's hint share is therefore self-limiting, and stale hints are consumed and discarded by the general execution managers that encounter them.

It does **not** follow that the total is bounded by `dispatch_queue_capacity`. `H` is decremented only by a general execution manager, so a pinned execution manager draining a group lowers `S` while leaving `H` at the old peak. The bound is `Σ_r peak(S_r)`, and those peaks are not simultaneous — a group backlogged alone reaches `B/2` under α = 1, so `N` groups that go backlogged in turn, each drained by its own pinned execution managers, can leave up to `N·B/2` hints outstanding with the buffer empty. The queue is therefore unbounded ([§3.5](#35-broadcast-queue)); the alternative is a rejected send, which breaks the invariant silently.

**Memory ordering.** `living_hint` is incremented only by the core and decremented only by general execution managers, so the check-then-increment needs no compare-exchange: a decrement landing between the read and the increment only lowers `H` further, which strengthens the condition that triggered publication. `Acquire`/`Release` suffices; sequential consistency is not required. Under acquire-release the core may miss a concurrent decrement and read `H` too high, skipping a publication — which is safe, because an unobserved decrement corresponds precisely to an execution manager that has entered flight and is therefore already counted in `m`.

### 8.2 Requirements the proof depends on

Four, each of which fails silently:

1. The assignment is sent to the group's queue **before** `H` is compared against `S`.
2. Neither send may block or fail. **No channel in this design is bounded by its type** — both the per-group queues and the broadcast queue are unbounded, and the algorithm is what bounds them. The admission threshold `S < F` is the sole limit on a group's occupancy, and hence on the buffer's; the publishing rule `H < S` is the sole limit on a group's hints. A channel capacity would be a redundant second limit measured against the wrong quantity — a per-group bound against a buffer-wide budget — whose only reachable effect is to reject a send the proof requires to succeed.
3. No await point between the broadcast queue receive and `H -= 1`, so cancellation cannot strand a decrement. The region must also not panic.
4. Hints carry a session generation; a general execution manager holding a stale-generation hint discards it **without touching any counter**.

### 8.3 Other invariants

**Dedup scope.** A `(JobId, TaskId)` is in the global task set exactly while it is buffered in the core, from job-registry insertion to assignment publication.

**Occupancy accounting.** Every resource group holding assignments is in `active_rg_list`, so `F` never over-states free space.

**Job placement.** A job entry is owned by the job registry, and its key is held by exactly one scheduling position: an active job list, a pending job queue, or the per-tick downgrade buffer.

This invariant no longer has a runtime safety net, and no longer needs one. It used to be enforced by `RefCell`, where violating it turned a borrow into a panic; the arena replaces that with two static properties. Two simultaneous mutable borrows of the arena are rejected by the borrow checker, at compile time, so a path that would have panicked does not build. And a key held past its entry's removal fails lookup rather than aliasing a different job, so a duplicated key degrades into a `None` rather than into two positions mutating one entry. What remains is a scheduling-correctness invariant — a job in two positions is dispatched from twice per round — not a liveness one.

No path in the design needs two job entries mutably at once. One that did would use `SlotMap::get_disjoint_mut`, which takes the keys together and returns `None` if they alias; it would not reach for a second borrow of the arena.

## 9. Session bump

Detected in step 1 from the session ID returned by the inbound-queue gRPC calls, and applied at the end of that step, in this order:

1. **Bump the global session ID.** Assignments already taken from the broadcast queue now fail their session check and are dropped by the service.
2. **Clear the resource group registry** — every entry in `rg_table`, and `rg_units`, `rg_index`, and `active_rg_list` **together, in this one operation**.
3. **Clear the job registry** — both `jobs` and `by_job_id`.
4. **Clear the global task set and the finalized job table.** Required: storage replays its ready tasks after a bump, and stale dedup entries would cause every replayed task to be dropped with nothing left in the registry to schedule it from.
5. **Drain the broadcast queue.**

Per-group queues are not drained explicitly; they are dropped with `rg_table`. An execution manager still blocked on an old reader keeps it alive but its assignments fail the session check.

**Step 2 is a correctness requirement, not tidiness.** Clearing `rg_units` is the one event that invalidates positions in it, and `rg_units` is a plain `Vec` precisely because that event is the only one ([§3.4](#34-resource-group-registry)). Indices carry no generation, so a surviving `active_rg_list` entry or `rg_index` value would resolve against the new session's units — either out of bounds, or, once the new session has re-created a few groups, silently against the wrong group. Both index holders must therefore be cleared in the same operation as the `Vec` they index; the type system does not check this, and it is why the three are named together above rather than left to the reader.

Job keys need no such care. A key held across the bump fails to resolve against the cleared arena, which is the same `None` any stale key produces — the bump is not a special case for them.

## 10. Configuration

| Parameter | Meaning |
|---|---|
| `dispatch_queue_capacity` | `B` — total dispatch buffer, sized to the storage poll interval plus burst headroom |
| `active_job_list_capacity` | Active jobs per resource group. A constant applied to each group, not a global budget divided among them |
| `tick_interval_ms` | Main loop cadence |
| `storage_poll_timeout_ms` | Inbound queue poll timeout |
| `ready_task_capacity`, `commit_ready_task_capacity`, `cleanup_ready_task_capacity` | Per-lane inbound fetch bounds |
| `DOWNGRADE_LIVES` | Compile-time constant, currently `1` |

α is fixed at 1 and is not configurable.

## 11. Non-goals

- **Dispatch rates across resource groups are not equalized.** Which group is served depends on execution manager arrivals, which the scheduler does not control. The guarantee is non-interference — a group with available execution managers is never blocked because another group is backlogged — plus reachability, not rate.
- **General capacity is distributed proportionally to backlog**, not equally per group, because hint volume tracks queue occupancy. This is what makes the "pinned execution managers went offline" case self-correcting, with no health signal required.
- **No minimum guarantee per resource group.** If per-tenant SLAs become necessary, the shape to adopt is a guaranteed floor with elastic borrowing above it, layered on the threshold rather than replacing it.
- **An assignment dispatched but not delivered is not recovered here.** That window is covered by the EM registry's liveness path.

## 12. Prototyping and benchmark

- Prototyping branch: [spider/components/spider-scheduler-new at spider-scheduler-new-prototyping · LinZhihao-723/spider](https://github.com/LinZhihao-723/spider/tree/spider-scheduler-new-prototyping/components/spider-scheduler-new)
- Benchmark result: [spider/claude/scheduler-redesign at spider-scheduler-new-prototyping · LinZhihao-723/spider](https://github.com/LinZhihao-723/spider/tree/spider-scheduler-new-prototyping/claude/scheduler-redesign)
