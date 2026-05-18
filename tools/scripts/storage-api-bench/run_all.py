#!/usr/bin/env python3
"""Runs the full storage API benchmark matrix and generates a report.

For each protocol in (grpc, rest), starts the server, runs the client for each
workload in (flat, deep, mixed) writing JSON to `<data-dir>/<protocol>_<workload>.json`,
then stops the server. Finally stages the six JSON files into `<data-dir>/reports/`
and runs `generate_report.py` to produce charts and `benchmark_report.md` there.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "tools/scripts/storage-api-bench"
DEFAULT_CONFIG = ROOT / "components/spider-storage-api-bench/config/default.toml"
WORKLOADS = ("flat", "deep", "mixed")
PROTOCOLS = ("grpc", "rest")


def main() -> int:
    args = parse_args()
    config = args.config.resolve()
    data_dir = args.data_dir.resolve()
    reports_dir = data_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    binds = parse_binds(config)

    for protocol in PROTOCOLS:
        host, port = binds[protocol]
        print(f"\n=== Starting {protocol} server on {host}:{port} ===", flush=True)
        server = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_server.py"),
                "--protocol",
                protocol,
                "--config",
                str(config),
            ],
            cwd=ROOT,
            start_new_session=True,
        )
        try:
            if not wait_for_port(host, port, args.startup_timeout, server):
                print(
                    f"{protocol} server did not start listening on {host}:{port} "
                    f"within {args.startup_timeout}s",
                    file=sys.stderr,
                )
                return 1
            for workload in WORKLOADS:
                output = data_dir / f"{protocol}_{workload}.json"
                print(f"\n--- Running {protocol} client / {workload} ---", flush=True)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "run_client.py"),
                        "--protocol",
                        protocol,
                        "--workload",
                        workload,
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    check=False,
                )
                if result.returncode != 0:
                    print(
                        f"Client {protocol}/{workload} exited with {result.returncode}",
                        file=sys.stderr,
                    )
                    return result.returncode
        finally:
            stop_server(server)

    for protocol in PROTOCOLS:
        for workload in WORKLOADS:
            src = data_dir / f"{protocol}_{workload}.json"
            if src.exists():
                shutil.copy2(src, reports_dir / src.name)

    print(f"\n=== Generating report in {reports_dir} ===", flush=True)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "generate_report.py"),
            "--data-dir",
            str(reports_dir),
        ],
        cwd=ROOT,
        check=False,
    ).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=pathlib.Path, default=ROOT / "data")
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the server to bind its port (default: 600).",
    )
    return parser.parse_args()


def parse_binds(config: pathlib.Path) -> dict[str, tuple[str, int]]:
    binds: dict[str, tuple[str, int]] = {}
    in_server = False
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_server = stripped == "[server]"
            continue
        if not in_server or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key == "rest_bind":
            binds["rest"] = split_host_port(value)
        elif key == "grpc_bind":
            binds["grpc"] = split_host_port(value)
    if "rest" not in binds or "grpc" not in binds:
        raise SystemExit(f"Could not find rest_bind/grpc_bind in {config}")
    return binds


def split_host_port(value: str) -> tuple[str, int]:
    host, port = value.rsplit(":", 1)
    return host, int(port)


def wait_for_port(
    host: str,
    port: int,
    timeout_s: int,
    server: subprocess.Popen,
) -> bool:
    deadline = time.monotonic() + timeout_s
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False
        try:
            with socket.create_connection((connect_host, port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    try:
        os.killpg(server.pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        server.terminate()
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            server.kill()
        server.wait()


if __name__ == "__main__":
    sys.exit(main())
