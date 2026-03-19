#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT=50051
NUM_WORKERS=16

echo "Building benchmarks (release)..."
cargo build --release -p spider-bench

SERVER_BIN="$WORKSPACE_ROOT/target/release/cache-server"
CLIENT_BIN="$WORKSPACE_ROOT/target/release/worker-client"

run_benchmark() {
    local bench_name=$1
    local compression=$2
    shift 2

    echo ""
    echo "============================================================"
    echo "  Benchmark: $bench_name (compression=$compression)"
    echo "============================================================"

    "$SERVER_BIN" --benchmark "$bench_name" --port "$PORT" --num-workers "$NUM_WORKERS" \
        --compression "$compression" "$@" &
    SERVER_PID=$!

    sleep 2
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server exited unexpectedly"
        exit 1
    fi

    "$CLIENT_BIN" --server-addr "http://[::1]:$PORT" --num-workers "$NUM_WORKERS" \
        --compression "$compression"

    wait "$SERVER_PID" 2>/dev/null || true
}

for compression in none zstd; do
    run_benchmark flat "$compression" --num-tasks 10000

    run_benchmark neural-net "$compression" \
        --neural-net-layers 10 \
        --neural-net-width 1000 \
        --neural-net-fan-in 25
done

echo ""
echo "All benchmarks complete."
