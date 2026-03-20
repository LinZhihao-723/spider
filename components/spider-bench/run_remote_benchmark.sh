#!/usr/bin/env bash
# =============================================================================
# Remote benchmark runner
#
# Usage:
#   ./run_remote_benchmark.sh \
#       --server-host user@host --client-host user@host \
#       [--port 50051] [--num-workers 32] \
#       [--ssh-key ~/.ssh/id_rsa] [--ssh-password 'pass']
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT=50051
NUM_WORKERS=32
SSH_KEY=""
SSH_PASSWORD=""
SERVER_HOST=""
CLIENT_HOST=""
REMOTE_DIR="/tmp/spider-bench"

usage() {
    echo "Usage: $0 --server-host USER@HOST --client-host USER@HOST [OPTIONS]"
    echo "  --port PORT          (default: 50051)"
    echo "  --num-workers N      (default: 32)"
    echo "  --ssh-key PATH"
    echo "  --ssh-password PASS  (requires sshpass)"
    echo "  --remote-dir PATH    (default: /tmp/spider-bench)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-host) SERVER_HOST="$2"; shift 2 ;;
        --client-host) CLIENT_HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --ssh-key) SSH_KEY="$2"; shift 2 ;;
        --ssh-password) SSH_PASSWORD="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$SERVER_HOST" || -z "$CLIENT_HOST" ]] && { echo "ERROR: --server-host and --client-host required."; usage; }

if [[ -n "$SSH_PASSWORD" ]] && ! command -v sshpass &>/dev/null; then
    echo "ERROR: sshpass not found. Install it or add to PATH."
    exit 1
fi

SSH_BASE_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15"
[[ -n "$SSH_KEY" ]] && SSH_BASE_OPTS="$SSH_BASE_OPTS -i $SSH_KEY"

# ssh_run HOST COMMAND — runs command via SSH, always returns 0 (ignores remote exit code)
ssh_run() {
    local host=$1; shift
    if [[ -n "$SSH_PASSWORD" ]]; then
        sshpass -p "$SSH_PASSWORD" ssh $SSH_BASE_OPTS "$host" "$@" || true
    else
        ssh $SSH_BASE_OPTS "$host" "$@" || true
    fi
}

# ssh_exec HOST COMMAND — runs command via SSH, propagates exit code
ssh_exec() {
    local host=$1; shift
    if [[ -n "$SSH_PASSWORD" ]]; then
        sshpass -p "$SSH_PASSWORD" ssh $SSH_BASE_OPTS "$host" "$@"
    else
        ssh $SSH_BASE_OPTS "$host" "$@"
    fi
}

rsync_to() {
    local src=$1 dst=$2
    if [[ -n "$SSH_PASSWORD" ]]; then
        sshpass -p "$SSH_PASSWORD" rsync -az -e "ssh $SSH_BASE_OPTS" "$src" "$dst"
    else
        rsync -az -e "ssh $SSH_BASE_OPTS" "$src" "$dst"
    fi
}

SERVER_HOSTNAME="${SERVER_HOST#*@}"

# =============================================================================
echo "=== Step 1: Build ==="
cargo build --release -p spider-bench
echo "Done."

# =============================================================================
echo "=== Step 2: Deploy ==="
ssh_run "$SERVER_HOST" "mkdir -p $REMOTE_DIR"
ssh_run "$CLIENT_HOST" "mkdir -p $REMOTE_DIR"
rsync_to "$WORKSPACE_ROOT/target/release/cache-server" "$SERVER_HOST:$REMOTE_DIR/cache-server"
rsync_to "$WORKSPACE_ROOT/target/release/worker-client" "$CLIENT_HOST:$REMOTE_DIR/worker-client"
echo "Done."

# =============================================================================
echo "=== Step 3: Benchmarks ==="

kill_server() {
    ssh_run "$SERVER_HOST" "kill \$(pgrep -f 'cache-server.*--port $PORT') 2>/dev/null"
    sleep 1
}

trap kill_server EXIT

run_bench() {
    local name=$1 compression=$2 input_size=$3
    shift 3

    echo ""
    echo "============================================================"
    echo "  $name  compression=$compression  input=$input_size"
    echo "============================================================"

    kill_server

    # Start server — use 'bash -c' to ensure proper backgrounding
    local log="$REMOTE_DIR/server_${name}_${compression}.log"
    ssh_run "$SERVER_HOST" "bash -c '$REMOTE_DIR/cache-server \
        --benchmark $name --port $PORT --num-workers $NUM_WORKERS \
        --compression $compression --input-size $input_size $* \
        >$log 2>&1 &'"

    # Wait for port to be open
    echo -n "  Waiting for server..."
    for _ in $(seq 1 30); do
        if ssh_exec "$SERVER_HOST" "nc -z 127.0.0.1 $PORT" 2>/dev/null; then
            echo " ready."
            break
        fi
        echo -n "."
        sleep 1
    done

    # Run client
    echo "  Running client ($NUM_WORKERS workers)..."
    ssh_exec "$CLIENT_HOST" "$REMOTE_DIR/worker-client \
        --server-addr http://$SERVER_HOSTNAME:$PORT \
        --num-workers $NUM_WORKERS \
        --compression $compression \
        --input-size $input_size" 2>&1

    # Fetch server stats
    sleep 2
    echo "  --- Server stats ---"
    ssh_run "$SERVER_HOST" "cat $log"

    kill_server
}

for comp in none zstd; do
    run_bench flat "$comp" 1024 --num-tasks 10000
    run_bench neural-net "$comp" 128 \
        --neural-net-layers 10 --neural-net-width 1000 --neural-net-fan-in 25
done

echo ""
echo "=== All benchmarks complete ==="
