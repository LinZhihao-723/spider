#!/usr/bin/env bash
# =============================================================================
# Remote benchmark runner — deploys and runs cache-server and worker-client on
# separate machines via SSH + rsync.
#
# Usage:
#   ./run_remote_benchmark.sh \
#       --server-host user@server-node \
#       --client-host user@client-node \
#       [--port 50051] \
#       [--num-workers 128] \
#       [--ssh-key ~/.ssh/id_rsa] \
#       [--ssh-password 'mypass']
#
# Authentication:
#   - Key-based: use --ssh-key (recommended)
#   - Password-based: use --ssh-password (requires sshpass installed)
#   - If neither is specified, uses default SSH agent / interactive prompt
#
# Requirements:
#   - SSH access to both nodes
#   - rsync installed on the local machine and both remote nodes
#   - sshpass installed locally if using --ssh-password
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
PORT=50051
NUM_WORKERS=128
SSH_KEY=""
SSH_PASSWORD=""
SERVER_HOST=""
CLIENT_HOST=""
REMOTE_DIR="/tmp/spider-bench"

usage() {
    cat <<EOF
Usage: $0 --server-host USER@HOST --client-host USER@HOST [OPTIONS]

Required:
  --server-host USER@HOST   SSH destination for the cache-server
  --client-host USER@HOST   SSH destination for the worker-client

Optional:
  --port PORT               Server listen port (default: 50051)
  --num-workers N           Number of concurrent workers (default: 128)
  --ssh-key PATH            SSH private key file
  --ssh-password PASS       SSH password (requires sshpass)
  --remote-dir PATH         Remote directory for binaries (default: /tmp/spider-bench)
EOF
    exit 1
}

# Parse arguments
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

if [[ -z "$SERVER_HOST" || -z "$CLIENT_HOST" ]]; then
    echo "ERROR: --server-host and --client-host are required."
    usage
fi

# Validate sshpass is available if password auth is requested.
if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass &>/dev/null; then
        echo "ERROR: --ssh-password requires 'sshpass' to be installed."
        echo "  Install with: apt-get install sshpass  (or brew install sshpass)"
        exit 1
    fi
fi

# Build SSH/rsync command wrappers that handle key or password auth.
SSH_BASE_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

if [[ -n "$SSH_KEY" ]]; then
    SSH_BASE_OPTS="$SSH_BASE_OPTS -i $SSH_KEY"
fi

ssh_cmd() {
    local host=$1; shift
    if [[ -n "$SSH_PASSWORD" ]]; then
        sshpass -p "$SSH_PASSWORD" ssh $SSH_BASE_OPTS "$host" "$@"
    else
        ssh $SSH_BASE_OPTS "$host" "$@"
    fi
}

rsync_cmd() {
    local src=$1 dst=$2
    if [[ -n "$SSH_PASSWORD" ]]; then
        sshpass -p "$SSH_PASSWORD" rsync -az --progress -e "ssh $SSH_BASE_OPTS" "$src" "$dst"
    else
        rsync -az --progress -e "ssh $SSH_BASE_OPTS" "$src" "$dst"
    fi
}

# Extract the hostname (without user@) for the server address the client connects to.
SERVER_HOSTNAME="${SERVER_HOST#*@}"

# =============================================================================
# Step 1: Build locally
# =============================================================================
echo "=== Step 1: Building release binaries locally ==="
cargo build --release -p spider-bench
echo "Build complete."

SERVER_BIN="$WORKSPACE_ROOT/target/release/cache-server"
CLIENT_BIN="$WORKSPACE_ROOT/target/release/worker-client"

# =============================================================================
# Step 2: Deploy binaries
# =============================================================================
echo ""
echo "=== Step 2: Deploying binaries to remote nodes ==="

echo "Creating remote directories..."
ssh_cmd "$SERVER_HOST" "mkdir -p $REMOTE_DIR"
ssh_cmd "$CLIENT_HOST" "mkdir -p $REMOTE_DIR"

echo "Syncing cache-server to $SERVER_HOST..."
rsync_cmd "$SERVER_BIN" "$SERVER_HOST:$REMOTE_DIR/cache-server"

echo "Syncing worker-client to $CLIENT_HOST..."
rsync_cmd "$CLIENT_BIN" "$CLIENT_HOST:$REMOTE_DIR/worker-client"

echo "Deploy complete."

# =============================================================================
# Step 3: Run benchmarks
# =============================================================================

cleanup() {
    echo "Cleaning up remote server processes..."
    ssh_cmd "$SERVER_HOST" "pkill -f 'cache-server.*--port $PORT' 2>/dev/null || true" 2>/dev/null || true
}
trap cleanup EXIT

run_remote_benchmark() {
    local bench_name=$1
    local compression=$2
    local input_size=$3
    shift 3
    local extra_args="$*"

    echo ""
    echo "============================================================"
    echo "  Benchmark: $bench_name (compression=$compression, input_size=$input_size)"
    echo "  Server: $SERVER_HOST | Client: $CLIENT_HOST"
    echo "============================================================"

    # Kill any leftover server from a previous run.
    ssh_cmd "$SERVER_HOST" "pkill -f 'cache-server.*--port $PORT' 2>/dev/null || true" 2>/dev/null || true
    sleep 1

    # Start server on the remote node.
    local server_log="$REMOTE_DIR/server_${bench_name}_${compression}.log"
    ssh_cmd "$SERVER_HOST" "nohup $REMOTE_DIR/cache-server \
        --benchmark $bench_name \
        --port $PORT \
        --num-workers $NUM_WORKERS \
        --compression $compression \
        --input-size $input_size \
        $extra_args \
        </dev/null >$server_log 2>&1 &"

    echo "Server starting on $SERVER_HOST:$PORT..."

    # Wait for server to be ready (poll until the port is listening).
    local retries=30
    local ready=false
    for i in $(seq 1 $retries); do
        if ssh_cmd "$SERVER_HOST" "ss -tln | grep -q ':${PORT}\b'" 2>/dev/null; then
            ready=true
            break
        fi
        sleep 1
    done

    if [[ "$ready" != "true" ]]; then
        echo "ERROR: Server did not start within ${retries}s. Server log:"
        ssh_cmd "$SERVER_HOST" "cat $server_log" 2>/dev/null || true
        return 1
    fi
    echo "Server ready."

    # Run client on the client node.
    echo "Starting client on $CLIENT_HOST ($NUM_WORKERS workers)..."
    ssh_cmd "$CLIENT_HOST" "$REMOTE_DIR/worker-client \
        --server-addr http://$SERVER_HOSTNAME:$PORT \
        --num-workers $NUM_WORKERS \
        --compression $compression \
        --input-size $input_size" 2>&1

    # Wait for server to finish and display its stats.
    sleep 2
    echo ""
    echo "--- Server-side stats ---"
    ssh_cmd "$SERVER_HOST" "cat $server_log" 2>/dev/null || echo "(no server log)"

    # Cleanup server.
    ssh_cmd "$SERVER_HOST" "pkill -f 'cache-server.*--port $PORT' 2>/dev/null || true" 2>/dev/null || true
}

echo ""
echo "=== Step 3: Running benchmarks ==="

for compression in none zstd; do
    run_remote_benchmark flat "$compression" 1024 --num-tasks 10000

    run_remote_benchmark neural-net "$compression" 128 \
        --neural-net-layers 10 \
        --neural-net-width 1000 \
        --neural-net-fan-in 25
done

echo ""
echo "============================================================"
echo "  All remote benchmarks complete."
echo "============================================================"
