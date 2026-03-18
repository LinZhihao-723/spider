#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT=50051
NUM_WORKERS=16

# Ensure protoc is available for building.
if ! command -v protoc &>/dev/null; then
    if [[ -x /tmp/protoc/bin/protoc ]]; then
        export PROTOC=/tmp/protoc/bin/protoc
    else
        echo "ERROR: protoc not found. Install it or set PROTOC env var."
        exit 1
    fi
fi

echo "Building benchmarks (release)..."
cargo build --release -p spider-bench

SERVER_BIN="$WORKSPACE_ROOT/target/release/cache-server"
CLIENT_BIN="$WORKSPACE_ROOT/target/release/worker-client"

run_benchmark() {
    local bench_name=$1
    shift

    echo ""
    echo "============================================================"
    echo "  Benchmark: $bench_name"
    echo "============================================================"

    # Start server in background.
    "$SERVER_BIN" --benchmark "$bench_name" --port "$PORT" "$@" &
    SERVER_PID=$!

    # Wait for server to be ready.
    sleep 2
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server exited unexpectedly"
        exit 1
    fi

    # Run the benchmark.
    "$CLIENT_BIN" --server-addr "http://[::1]:$PORT" --num-workers "$NUM_WORKERS"

    # Wait for server to finish.
    wait "$SERVER_PID" 2>/dev/null || true

    echo "  Benchmark $bench_name complete."
}

# Scenario 1: 10k independent tasks (flat).
run_benchmark flat --num-tasks 10000

# Scenario 2: 10k neural-network tasks.
run_benchmark neural-net \
    --neural-net-layers 10 \
    --neural-net-width 1000 \
    --neural-net-fan-in 10

echo ""
echo "All benchmarks complete."
