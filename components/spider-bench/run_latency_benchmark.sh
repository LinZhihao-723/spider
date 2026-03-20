#!/usr/bin/env bash
# Scaling benchmark: per-task avg latency for register & submit, client + server side.
# 5 runs per combo, results in CSV + plot.
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
CLIENT_HOSTS=("lzh@10.1.0.10" "lzh@10.1.0.11" "lzh@10.1.0.12" "lzh@10.1.0.13" "lzh@10.1.0.14" "lzh@10.1.0.9" "lzh@10.1.0.16" "lzh@10.1.0.17")
WORKER_COUNTS=(16 32 64 128 256)
REMOTE_DIR="/tmp/spider-bench"

while [[ $# -gt 0 ]]; do
    case "$1" in --ssh-password) SSH_PASSWORD="$2"; shift 2 ;; *) echo "Unknown: $1"; exit 1 ;; esac
done
[[ -z "$SSH_PASSWORD" ]] && { echo "--ssh-password required"; exit 1; }

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15"
ssh_run() { sshpass -p "$SSH_PASSWORD" ssh $SSH_OPTS "$@" || true; }
ssh_exec() { sshpass -p "$SSH_PASSWORD" ssh $SSH_OPTS "$@"; }
rsync_to() { sshpass -p "$SSH_PASSWORD" rsync -az -e "ssh $SSH_OPTS" "$1" "$2"; }

echo "=== Build & Deploy ==="
cargo build --release -p spider-bench
ssh_run "$SERVER_HOST" "mkdir -p $REMOTE_DIR"
rsync_to "$WORKSPACE_ROOT/target/release/cache-server" "$SERVER_HOST:$REMOTE_DIR/"
for h in "${CLIENT_HOSTS[@]}"; do
    ssh_run "$h" "mkdir -p $REMOTE_DIR"
    rsync_to "$WORKSPACE_ROOT/target/release/worker-client" "$h:$REMOTE_DIR/" &
done
wait
echo "Done."

mkdir -p "$OUTPUT_DIR"
CSV="$OUTPUT_DIR/latency_results.csv"
echo "benchmark,compression,total_workers,run,client_register_avg_ms,client_submit_avg_ms,server_register_avg_ms,server_submit_avg_ms,server_total_ms" > "$CSV"

kill_server() { ssh_run "$SERVER_HOST" "kill \$(pgrep -f 'cache-server.*--port $PORT') 2>/dev/null"; sleep 1; }
trap kill_server EXIT

run_one() {
    local bench=$1 comp=$2 input_size=$3 total_workers=$4 run_id=$5
    shift 5
    local extra="$*"
    local nm=$(( (total_workers + WORKERS_PER_MACHINE - 1) / WORKERS_PER_MACHINE ))

    kill_server

    local slog="$REMOTE_DIR/s_${bench}_${comp}_${total_workers}_${run_id}.log"
    ssh_run "$SERVER_HOST" "bash -c '$REMOTE_DIR/cache-server \
        --benchmark $bench --port $PORT --num-workers $total_workers \
        --compression $comp --input-size $input_size $extra \
        >$slog 2>&1 &'"

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
    local slog_content
    slog_content=$(ssh_run "$SERVER_HOST" "cat $slog")

    # Parse server metrics
    local s_total s_reg s_sub
    s_total=$(echo "$slog_content" | grep "server_total_time:" | awk '{print $2}')
    s_reg=$(echo "$slog_content" | grep "server_register_avg:" | awk '{print $2}')
    s_sub=$(echo "$slog_content" | grep "server_submit_avg:" | awk '{print $2}')
    [[ -z "$s_total" ]] && s_total="0"
    [[ -z "$s_reg" ]] && s_reg="0"
    [[ -z "$s_sub" ]] && s_sub="0"

    # Parse client metrics from first client
    local cout
    cout=$(cat "${outs[0]}" 2>/dev/null || echo "")
    local c_reg c_sub
    c_reg=$(echo "$cout" | grep -A4 "RegisterTaskInstance" | grep "avg:" | head -1 | awk '{print $2}')
    c_sub=$(echo "$cout" | grep -A4 "SubmitTaskResult" | grep "avg:" | head -1 | awk '{print $2}')
    [[ -z "$c_reg" ]] && c_reg="0"
    [[ -z "$c_sub" ]] && c_sub="0"

    echo "$bench,$comp,$total_workers,$run_id,$c_reg,$c_sub,$s_reg,$s_sub,$s_total" >> "$CSV"
    echo "      run $run_id: c_reg=${c_reg} c_sub=${c_sub} s_reg=${s_reg} s_sub=${s_sub} total=${s_total}"

    kill_server
    rm -f "${outs[@]}" 2>/dev/null || true
}

echo "=== Benchmarks ($NUM_RUNS runs each) ==="
for tw in "${WORKER_COUNTS[@]}"; do
    echo ""
    echo "==== Workers: $tw ===="
    for comp in none zstd; do
        for ba in "flat 1024 --num-tasks 10000" "neural-net 128 --neural-net-layers 10 --neural-net-width 1000 --neural-net-fan-in 25"; do
            bench=$(echo "$ba" | awk '{print $1}')
            isz=$(echo "$ba" | awk '{print $2}')
            extra=$(echo "$ba" | cut -d' ' -f3-)
            echo "  $bench comp=$comp"
            for run in $(seq 1 $NUM_RUNS); do
                run_one "$bench" "$comp" "$isz" "$tw" "$run" $extra
            done
        done
    done
done

echo ""
cat "$CSV"
echo ""
echo "=== Plotting ==="
python3 "$SCRIPT_DIR/plot_latency.py" "$CSV" "$OUTPUT_DIR"
echo "=== Done ==="
