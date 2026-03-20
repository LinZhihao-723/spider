#!/usr/bin/env bash
# Final benchmark: resumable, with stuck detection and auto-retry.
# Max 32 workers per node, up to 256 workers, 5 runs each, 4 combos.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SSH_PASSWORD="$1"
OUTPUT_DIR="$WORKSPACE_ROOT/claude/cache-impl/run-6"
PORT=50051
NUM_RUNS=10
MAX_PER_NODE=32
MAX_RETRIES=3

SERVER_HOST="lzh@10.1.0.7"
SERVER_IP="10.1.0.7"
CLIENT_HOSTS=("lzh@10.1.0.9" "lzh@10.1.0.10" "lzh@10.1.0.11" "lzh@10.1.0.12" "lzh@10.1.0.13" "lzh@10.1.0.14" "lzh@10.1.0.16" "lzh@10.1.0.17")
REMOTE_DIR="/tmp/spider-bench"

WORKER_COUNTS=(16 32 64 128 256)

sr() { sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$@" || true; }

kill_all() {
    for ip in 10.1.0.7 10.1.0.9 10.1.0.10 10.1.0.11 10.1.0.12 10.1.0.13 10.1.0.14 10.1.0.16 10.1.0.17; do
        sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 "lzh@$ip" \
            'kill -9 $(pgrep -f "cache-server\|worker-client") 2>/dev/null; true' 2>/dev/null &
    done
    wait
    sleep 1
}

echo "=== Build & Deploy ==="
cargo build --release -p spider-bench
sr "$SERVER_HOST" "mkdir -p $REMOTE_DIR"
sshpass -p "$SSH_PASSWORD" rsync -az -e "ssh -o StrictHostKeyChecking=no" target/release/cache-server "$SERVER_HOST:$REMOTE_DIR/"
for h in "${CLIENT_HOSTS[@]}"; do
    sr "$h" "mkdir -p $REMOTE_DIR"
    sshpass -p "$SSH_PASSWORD" rsync -az -e "ssh -o StrictHostKeyChecking=no" target/release/worker-client "$h:$REMOTE_DIR/" &
done
wait
echo "Done."

mkdir -p "$OUTPUT_DIR"
CSV="$OUTPUT_DIR/final_results.csv"

# Fresh CSV
echo "benchmark,compression,workers,run,server_total_ms,server_reg_avg_ms,server_sub_avg_ms,client_reg_avg_ms,client_sub_avg_ms,tasks_registered" > "$CSV"

run_exists() {
    grep -q "^${1},${2},${3},${4}," "$CSV" 2>/dev/null
}

# Run one benchmark. Returns 0 on success, 1 on failure/stuck.
run_one() {
    local bench=$1 comp=$2 input_size=$3 tw=$4 run_id=$5
    shift 5
    local extra="$*"

    if run_exists "$bench" "$comp" "$tw" "$run_id"; then
        echo "      run $run_id: SKIP"
        return 0
    fi

    local num_nodes=$(( (tw + MAX_PER_NODE - 1) / MAX_PER_NODE ))
    (( num_nodes > 8 )) && num_nodes=8

    kill_all

    local slog="$REMOTE_DIR/final_${bench}_${comp}_${tw}_${run_id}.log"
    sr "$SERVER_HOST" "bash -c '$REMOTE_DIR/cache-server --benchmark $bench --port $PORT --num-workers $tw --compression $comp --input-size $input_size $extra >$slog 2>&1 &'"
    sleep 2

    local pids=() outs=()
    local offset=0 remaining=$tw
    for ((m=0; m<num_nodes; m++)); do
        local host="${CLIENT_HOSTS[$m]}"
        local nw=$(( remaining / (num_nodes - m) ))
        remaining=$((remaining - nw))
        if (( nw == 0 )); then continue; fi
        local out="/tmp/spider-bench-final-${m}.out"
        outs+=("$out")
        # Per-client timeout: generous enough for 10k tasks but catches stuck
        timeout 120 sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" \
            "$REMOTE_DIR/worker-client --server-addr http://$SERVER_IP:$PORT --num-workers $nw --compression $comp --input-size $input_size --worker-id-offset $offset" >"$out" 2>&1 &
        pids+=($!); offset=$((offset + nw))
    done

    # Wait with overall timeout detection
    local all_ok=true
    for p in "${pids[@]}"; do
        if ! wait "$p" 2>/dev/null; then
            all_ok=false
        fi
    done

    # Wait for server shutdown
    sleep 5

    local slog_content
    slog_content=$(sr "$SERVER_HOST" "cat $slog")

    local s_total s_reg s_sub reg_count
    s_total=$(echo "$slog_content" | grep "server_total_time:" | awk '{print $2}')
    s_reg=$(echo "$slog_content" | grep "server_register_avg:" | awk '{print $2}')
    s_sub=$(echo "$slog_content" | grep "server_submit_avg:" | awk '{print $2}')
    reg_count=$(echo "$slog_content" | grep -A1 "Register (server)" | grep "count:" | awk '{print $2}')

    [[ -z "$s_total" ]] && s_total="0"
    [[ -z "$s_reg" ]] && s_reg="0"
    [[ -z "$s_sub" ]] && s_sub="0"
    [[ -z "$reg_count" ]] && reg_count="0"

    local c_reg="0" c_sub="0"
    local cout=$(cat "${outs[0]}" 2>/dev/null || echo "")
    c_reg=$(echo "$cout" | grep -A4 "RegisterTaskInstance" | grep "avg:" | head -1 | awk '{print $2}')
    c_sub=$(echo "$cout" | grep -A4 "SubmitTaskResult" | grep "avg:" | head -1 | awk '{print $2}')
    [[ -z "$c_reg" ]] && c_reg="0"
    [[ -z "$c_sub" ]] && c_sub="0"

    rm -f "${outs[@]}" 2>/dev/null || true

    # Check success
    if [[ "$reg_count" != "10000" ]]; then
        echo "      run $run_id: FAILED (tasks=${reg_count}/10000)"
        kill_all
        return 1
    fi

    echo "$bench,$comp,$tw,$run_id,$s_total,$s_reg,$s_sub,$c_reg,$c_sub,$reg_count" >> "$CSV"
    echo "      run $run_id: total=${s_total}ms OK"
    kill_all
    return 0
}

echo "=== Benchmark: ${#WORKER_COUNTS[@]} worker counts × 4 combos × $NUM_RUNS runs ==="
for tw in "${WORKER_COUNTS[@]}"; do
    nn=$(( (tw + MAX_PER_NODE - 1) / MAX_PER_NODE ))
    (( nn > 8 )) && nn=8
    echo ""
    echo "======== Workers: $tw ($nn nodes, $MAX_PER_NODE/node) ========"
    for comp in none zstd; do
        for bench_cfg in "flat 1024 --num-tasks 10000" "neural-net 128 --neural-net-layers 10 --neural-net-width 1000 --neural-net-fan-in 25"; do
            bench=$(echo "$bench_cfg" | awk '{print $1}')
            isz=$(echo "$bench_cfg" | awk '{print $2}')
            extra=$(echo "$bench_cfg" | cut -d' ' -f3-)
            echo "  --- $bench, $comp ---"
            for run in $(seq 1 $NUM_RUNS); do
                for attempt in $(seq 1 $MAX_RETRIES); do
                    if run_one "$bench" "$comp" "$isz" "$tw" "$run" $extra; then
                        break
                    fi
                    if (( attempt < MAX_RETRIES )); then
                        echo "      Retrying (attempt $((attempt+1))/$MAX_RETRIES)..."
                        sleep 2
                    else
                        echo "      GIVING UP on run $run after $MAX_RETRIES attempts"
                    fi
                done
            done
        done
    done
done

echo ""
echo "=== Plotting ==="
python3 "$SCRIPT_DIR/plot_final.py" "$CSV" "$OUTPUT_DIR"
echo "=== Done ==="
