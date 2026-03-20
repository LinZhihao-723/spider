#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SSH_PASSWORD=""
OUTPUT_DIR="$WORKSPACE_ROOT/claude/cache-impl"
PORT=50051
WORKERS_PER_MACHINE=32
NUM_RUNS=5

SERVER_HOST="lzh@10.1.0.18"
SERVER_IP="10.1.0.18"
CLIENT_HOSTS=(
    "lzh@10.1.0.10"
    "lzh@10.1.0.11"
    "lzh@10.1.0.12"
    "lzh@10.1.0.13"
    "lzh@10.1.0.14"
    "lzh@10.1.0.9"
    "lzh@10.1.0.16"
    "lzh@10.1.0.17"
)
WORKER_COUNTS=(16 32 64 128 256)
REMOTE_DIR="/tmp/spider-bench"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ssh-password) SSH_PASSWORD="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done
[[ -z "$SSH_PASSWORD" ]] && { echo "--ssh-password required"; exit 1; }

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15"
ssh_run() { sshpass -p "$SSH_PASSWORD" ssh $SSH_OPTS "$@" || true; }
ssh_exec() { sshpass -p "$SSH_PASSWORD" ssh $SSH_OPTS "$@"; }
rsync_to() { sshpass -p "$SSH_PASSWORD" rsync -az -e "ssh $SSH_OPTS" "$1" "$2"; }

echo "=== Build ==="
cargo build --release -p spider-bench
echo "=== Deploy ==="
ssh_run "$SERVER_HOST" "mkdir -p $REMOTE_DIR"
rsync_to "$WORKSPACE_ROOT/target/release/cache-server" "$SERVER_HOST:$REMOTE_DIR/"
for h in "${CLIENT_HOSTS[@]}"; do
    ssh_run "$h" "mkdir -p $REMOTE_DIR"
    rsync_to "$WORKSPACE_ROOT/target/release/worker-client" "$h:$REMOTE_DIR/" &
done
wait
echo "Done."

mkdir -p "$OUTPUT_DIR"
CSV="$OUTPUT_DIR/scaling_results.csv"
echo "benchmark,compression,total_workers,run,server_total_ms" > "$CSV"

kill_server() {
    ssh_run "$SERVER_HOST" "kill \$(pgrep -f 'cache-server.*--port $PORT') 2>/dev/null"
    sleep 1
}
kill_clients() {
    for h in "${CLIENT_HOSTS[@]}"; do ssh_run "$h" "kill \$(pgrep -f worker-client) 2>/dev/null" & done; wait
}
trap 'kill_server; kill_clients' EXIT

run_one() {
    local bench=$1 comp=$2 input_size=$3 total_workers=$4 run_id=$5
    shift 5
    local extra="$*"
    local nm=$(( (total_workers + WORKERS_PER_MACHINE - 1) / WORKERS_PER_MACHINE ))

    kill_server

    local log="$REMOTE_DIR/s_${bench}_${comp}_${total_workers}_${run_id}.log"
    ssh_run "$SERVER_HOST" "bash -c '$REMOTE_DIR/cache-server \
        --benchmark $bench --port $PORT --num-workers $total_workers \
        --compression $comp --input-size $input_size $extra \
        >$log 2>&1 &'"

    for _ in $(seq 1 30); do
        ssh_exec "$SERVER_HOST" "nc -z 127.0.0.1 $PORT" 2>/dev/null && break
        sleep 1
    done

    local pids=() outs=()
    local offset=0
    for ((m=0; m<nm; m++)); do
        local host="${CLIENT_HOSTS[$m]}"
        local nw=$WORKERS_PER_MACHINE
        local rem=$((total_workers - offset))
        (( rem < nw )) && nw=$rem
        local out="/tmp/spider-bench-c-${m}.out"
        outs+=("$out")
        ssh_exec "$host" "$REMOTE_DIR/worker-client \
            --server-addr http://$SERVER_IP:$PORT \
            --num-workers $nw --compression $comp --input-size $input_size \
            --worker-id-offset $offset" >"$out" 2>&1 &
        pids+=($!)
        offset=$((offset + nw))
    done
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done

    sleep 2
    local slog
    slog=$(ssh_run "$SERVER_HOST" "cat $log")
    local stotal
    stotal=$(echo "$slog" | grep "server_total_time:" | awk '{print $2}')
    [[ -z "$stotal" ]] && stotal="0"

    echo "$bench,$comp,$total_workers,$run_id,$stotal" >> "$CSV"
    echo "      run $run_id: server_total=${stotal}ms"

    kill_server
    rm -f "${outs[@]}" 2>/dev/null || true
}

echo "=== Benchmarks (${NUM_RUNS} runs each) ==="
for tw in "${WORKER_COUNTS[@]}"; do
    echo ""
    echo "==== Workers: $tw ===="
    for comp in none zstd; do
        for bench_args in "flat 1024 --num-tasks 10000" "neural-net 128 --neural-net-layers 10 --neural-net-width 1000 --neural-net-fan-in 25"; do
            bench=$(echo "$bench_args" | awk '{print $1}')
            input_size=$(echo "$bench_args" | awk '{print $2}')
            extra=$(echo "$bench_args" | cut -d' ' -f3-)
            echo "  $bench comp=$comp input=$input_size"
            for run in $(seq 1 $NUM_RUNS); do
                run_one "$bench" "$comp" "$input_size" "$tw" "$run" $extra
            done
        done
    done
done

echo ""
echo "=== CSV: $CSV ==="
cat "$CSV"
echo ""
echo "=== Plotting ==="
python3 "$SCRIPT_DIR/plot_scaling.py" "$CSV" "$OUTPUT_DIR"
echo "=== Done ==="
