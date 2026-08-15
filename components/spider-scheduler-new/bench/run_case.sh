#!/usr/bin/env bash

# Runs one benchmark case of the prototype scheduler end to end: the scheduler process, this host's
# worker processes, and the wall clock around them.
#
# Every dimension of a case -- resource groups, jobs, tasks, workers, buffer sizes -- comes from the
# case table compiled into the two binaries and is selected here by name alone, so this script never
# restates a benchmark parameter.

set -euo pipefail

# The line the scheduler prints once it is serving.
readonly READY_MARKER="BENCH_SCHEDULER_READY"

# How often the scheduler's log is re-checked for the readiness line, in seconds.
readonly READY_POLL_INTERVAL_SECS=0.1

# The number of trailing log lines printed when a process fails.
readonly FAILURE_LOG_LINES=40

# The scheduler's process ID, tracked globally so that an early exit can still stop it.
scheduler_pid=""

# This host's worker process IDs, tracked globally for the same reason.
worker_pids=()

print_usage() {
    cat <<'USAGE'
Usage: run_case.sh <case> <output-dir>

  <case>        The benchmark case to run: A, B, C, D, E, or SMOKE.
  <output-dir>  The directory the run's result files are written into.

Environment:
  BENCH_LISTEN              The address the scheduler binds (default 127.0.0.1:50151).
  BENCH_ENDPOINT            The endpoint the workers dial (default http://$BENCH_LISTEN).
  BENCH_NUM_SHARDS          The number of worker processes this host runs (default 1). A
                            multi-node run instead starts bench-workers on each host with a
                            different --index over the same --num-shards.
  BENCH_CHANNEL_POOL_SIZE   How many gRPC channels each worker process spreads its workers over.
                            Unset gives every worker a channel of its own, which is what the
                            benchmark contract requires; set it only to narrow the pool
                            deliberately. The figure is recorded in every worker result file.
                            Sharing a channel between workers inflates client-side dispatch latency
                            by orders of magnitude while server-side latency stays put: on case A,
                            p90 client latency was 487us at one channel per worker, 6.75ms at 2.5
                            workers per channel, and 6.75ms with a p50 of 2.1ms at 10 workers per
                            channel.
  BENCH_BIN_DIR             Where the two binaries live (default <repo>/target/release).
  BENCH_SKIP_BUILD          Set to any value to skip the release build.
  BENCH_READY_TIMEOUT_SECS  How long to wait for the scheduler to serve (default 60).
USAGE
}

die() {
    printf 'run_case.sh: %s\n' "$1" >&2
    exit 1
}

report_failure() {
    local what=$1 reason=$2 log_file=$3
    printf 'run_case.sh: %s %s. Last %s lines of %s:\n' \
        "$what" "$reason" "$FAILURE_LOG_LINES" "$log_file" >&2
    tail -n "$FAILURE_LOG_LINES" "$log_file" >&2 || true
}

stop_process() {
    local pid=$1
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    local pid
    if ((${#worker_pids[@]} > 0)); then
        for pid in "${worker_pids[@]}"; do
            stop_process "$pid"
        done
    fi
    stop_process "$scheduler_pid"
}

wait_for_ready() {
    local pid=$1 log_file=$2 timeout_secs=$3
    local deadline=$((SECONDS + timeout_secs))
    while ((SECONDS < deadline)); do
        if grep -q -- "$READY_MARKER" "$log_file"; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            report_failure "the scheduler" "exited before it was ready" "$log_file"
            return 1
        fi
        sleep "$READY_POLL_INTERVAL_SECS"
    done

    report_failure "the scheduler" "did not become ready within ${timeout_secs}s" "$log_file"
    return 1
}

main() {
    if [[ $# -ne 2 ]]; then
        print_usage >&2
        exit 2
    fi
    local case_name=$1 output_dir=$2

    local script_dir repo_root
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    repo_root=$(cd -- "${script_dir}/../../.." && pwd)

    local bin_dir=${BENCH_BIN_DIR:-${repo_root}/target/release}
    local listen=${BENCH_LISTEN:-127.0.0.1:50151}
    local endpoint=${BENCH_ENDPOINT:-http://${listen}}
    local num_shards=${BENCH_NUM_SHARDS:-1}
    local ready_timeout_secs=${BENCH_READY_TIMEOUT_SECS:-60}
    local -a channel_pool_args=()
    if [[ -n ${BENCH_CHANNEL_POOL_SIZE:-} ]]; then
        channel_pool_args=(--channel-pool-size "$BENCH_CHANNEL_POOL_SIZE")
    fi
    local case_lower
    case_lower=$(printf '%s' "$case_name" | tr '[:upper:]' '[:lower:]')

    if [[ -z ${BENCH_SKIP_BUILD:-} ]]; then
        (cd "$repo_root" && cargo build --release -p spider-scheduler-new --bins)
    fi
    local binary
    for binary in bench-scheduler bench-workers; do
        [[ -x "${bin_dir}/${binary}" ]] || die "${bin_dir}/${binary} is missing or not executable"
    done

    local log_dir="${output_dir}/logs"
    mkdir -p "$output_dir" "$log_dir"

    local scheduler_log="${log_dir}/${case_lower}-scheduler.log"
    : >"$scheduler_log"
    trap cleanup EXIT
    "${bin_dir}/bench-scheduler" \
        --case "$case_name" \
        --listen "$listen" \
        --output "$output_dir" \
        >"$scheduler_log" 2>&1 &
    scheduler_pid=$!
    wait_for_ready "$scheduler_pid" "$scheduler_log" "$ready_timeout_secs" || exit 1
    printf 'run_case.sh: case %s is serving on %s; starting %s worker process(es).\n' \
        "$case_name" "$endpoint" "$num_shards"

    local started_ns
    started_ns=$(date +%s%N)
    worker_pids=()
    local -a worker_logs=()
    local index
    for ((index = 0; index < num_shards; index++)); do
        local worker_log="${log_dir}/${case_lower}-workers-${index}.log"
        "${bin_dir}/bench-workers" \
            --case "$case_name" \
            --endpoint "$endpoint" \
            --output "$output_dir" \
            --index "$index" \
            --num-shards "$num_shards" \
            ${channel_pool_args[@]+"${channel_pool_args[@]}"} \
            >"$worker_log" 2>&1 &
        worker_pids+=("$!")
        worker_logs+=("$worker_log")
    done

    local num_failures=0 exit_code
    for ((index = 0; index < num_shards; index++)); do
        exit_code=0
        wait "${worker_pids[index]}" || exit_code=$?
        if ((exit_code != 0)); then
            report_failure "worker shard ${index}" "failed with exit code ${exit_code}" \
                "${worker_logs[index]}"
            num_failures=$((num_failures + 1))
        fi
    done
    worker_pids=()

    # A shard that died leaves a workload that can never drain, so the scheduler is stopped here
    # rather than waited out for the whole of its own run timeout.
    local scheduler_was_stopped=0
    if ((num_failures != 0)); then
        stop_process "$scheduler_pid"
        scheduler_was_stopped=1
    fi

    exit_code=0
    wait "$scheduler_pid" || exit_code=$?
    scheduler_pid=""
    if ((exit_code != 0 && scheduler_was_stopped == 0)); then
        report_failure "the scheduler" "failed with exit code ${exit_code}" "$scheduler_log"
        num_failures=$((num_failures + 1))
    fi

    local ended_ns wall_clock_ms
    ended_ns=$(date +%s%N)
    wall_clock_ms=$(((ended_ns - started_ns) / 1000000))

    local run_file="${output_dir}/${case_lower}-run.json"
    printf '{"case":"%s","wall_clock_ms":%s,"num_worker_shards":%s,"endpoint":"%s","finished_at":"%s"}\n' \
        "$case_name" "$wall_clock_ms" "$num_shards" "$endpoint" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$run_file"

    if ((num_failures != 0)); then
        die "case ${case_name} failed: ${num_failures} process(es) exited non-zero"
    fi
    printf 'run_case.sh: case %s finished in %s ms; results in %s.\n' \
        "$case_name" "$wall_clock_ms" "$output_dir"
}

main "$@"
