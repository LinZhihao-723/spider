#!/usr/bin/env python3
"""Generates storage API benchmark charts and a Markdown report."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass

import cairo


ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "data"
WORKLOADS = ("flat", "deep", "mixed")
PROTOCOLS = ("Grpc", "Rest")
REQUEST_ORDER = (
    "add_resource_group",
    "get_job_state",
    "get_session",
    "poll_cleanup_ready_tasks",
    "poll_commit_ready_tasks",
    "poll_ready_tasks",
    "register_execution_manager",
    "register_job",
    "start_job",
    "succeed_cleanup_task_instance",
    "succeed_commit_task_instance",
    "succeed_task_instance",
)
PHASE_ORDER = (
    "db_add",
    "db_register",
    "parse_graph",
    "unframe_inputs",
    "validate",
    "create_jcb",
    "cache_insert",
    "cache_get",
    "jcb_start",
)
COLORS = {
    "grpc": (0.10, 0.34, 0.74),
    "rest": (0.10, 0.52, 0.34),
    "server_other": (0.46, 0.50, 0.56),
    "overhead": (0.91, 0.52, 0.14),
    "grid": (0.86, 0.88, 0.90),
    "text": (0.10, 0.12, 0.15),
    "muted": (0.38, 0.42, 0.46),
    "white": (1.0, 1.0, 1.0),
}
PHASE_COLORS = {
    "db_add": (0.09, 0.44, 0.70),
    "db_register": (0.07, 0.50, 0.46),
    "parse_graph": (0.50, 0.33, 0.72),
    "unframe_inputs": (0.58, 0.40, 0.18),
    "validate": (0.60, 0.58, 0.15),
    "create_jcb": (0.77, 0.29, 0.30),
    "cache_insert": (0.35, 0.61, 0.25),
    "cache_get": (0.34, 0.62, 0.78),
    "jcb_start": (0.85, 0.37, 0.18),
}


@dataclass(frozen=True)
class RequestRow:
    workload: str
    protocol: str
    category: str
    operation: str
    count: int
    server_avg_us: float
    client_avg_us: float

    @property
    def overhead_us(self) -> float:
        return max(self.client_avg_us - self.server_avg_us, 0.0)


@dataclass(frozen=True)
class RunSummary:
    workload: str
    protocol: str
    requests: int
    server_avg_us: float
    client_avg_us: float
    job_latency: dict[str, int]
    setup: dict[str, object]

    @property
    def overhead_us(self) -> float:
        return max(self.client_avg_us - self.server_avg_us, 0.0)


@dataclass(frozen=True)
class SizeRow:
    workload: str
    protocol: str
    operation: str
    count: int
    avg_bytes: float
    p50_bytes: float
    p99_bytes: float
    max_bytes: float
    total_bytes: float


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    request_rows, phase_rows, summaries, size_rows = load_results(data_dir)
    chart_paths = []
    for workload in WORKLOADS:
        chart_path = data_dir / f"avg_request_time_{workload}.png"
        draw_workload_chart(workload, request_rows, phase_rows, chart_path)
        chart_paths.append(chart_path)
    write_report(
        data_dir / "benchmark_report.md",
        chart_paths,
        request_rows,
        phase_rows,
        summaries,
        size_rows,
    )
    for path in [*chart_paths, data_dir / "benchmark_report.md"]:
        print(path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def load_results(
    data_dir: pathlib.Path,
) -> tuple[list[RequestRow], list[RequestRow], list[RunSummary], list[SizeRow]]:
    request_rows: list[RequestRow] = []
    phase_rows: list[RequestRow] = []
    summaries: list[RunSummary] = []
    size_rows: list[SizeRow] = []
    for path in sorted(data_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not {"setup", "request_latency", "server_metrics", "job_latency"} <= data.keys():
            continue
        setup = data["setup"]
        workload = setup["workload"]
        protocol = setup["protocol"]
        client_by_key = {
            (row["category"], row["operation"]): row for row in data["request_latency"]
        }
        server_by_key = {
            (row["category"], row["operation"]): row
            for row in data["server_metrics"]["request_latency"]
        }
        for key, client_row in sorted(client_by_key.items(), key=lambda item: sort_key(item[0])):
            server_row = server_by_key.get(key)
            if server_row is None:
                continue
            row = RequestRow(
                workload=workload,
                protocol=protocol,
                category=client_row["category"],
                operation=client_row["operation"],
                count=client_row["count"],
                server_avg_us=server_row["avg_us"],
                client_avg_us=client_row["avg_us"],
            )
            request_rows.append(row)
        for server_row in data["server_metrics"]["request_latency"]:
            if server_row["category"] != "phase":
                continue
            phase_rows.append(
                RequestRow(
                    workload=workload,
                    protocol=protocol,
                    category=server_row["category"],
                    operation=server_row["operation"],
                    count=server_row["count"],
                    server_avg_us=server_row["avg_us"],
                    client_avg_us=server_row["avg_us"],
                )
            )
        server_request_rows = [
            row
            for row in data["server_metrics"]["request_latency"]
            if row["category"] != "phase"
        ]
        client_request_rows = [
            row for row in data["request_latency"] if row["category"] != "phase"
        ]
        server_avg, request_count = weighted_average(server_request_rows)
        client_avg, _ = weighted_average(client_request_rows)
        summaries.append(
            RunSummary(
                workload=workload,
                protocol=protocol,
                requests=request_count,
                server_avg_us=server_avg,
                client_avg_us=client_avg,
                job_latency=data["job_latency"],
                setup=setup,
            )
        )
        for size_row in data["server_metrics"].get("request_sizes", []):
            size_rows.append(
                SizeRow(
                    workload=workload,
                    protocol=protocol,
                    operation=size_row["operation"],
                    count=size_row["count"],
                    avg_bytes=size_row["avg_bytes"],
                    p50_bytes=size_row.get("p50_bytes", 0),
                    p99_bytes=size_row.get("p99_bytes", 0),
                    max_bytes=size_row.get("max_bytes", 0),
                    total_bytes=size_row.get("total_bytes", 0),
                )
            )
    request_rows.sort(key=lambda row: (WORKLOADS.index(row.workload), sort_key((row.category, row.operation)), PROTOCOLS.index(row.protocol)))
    phase_rows.sort(key=lambda row: (WORKLOADS.index(row.workload), row.operation, PROTOCOLS.index(row.protocol)))
    summaries.sort(key=lambda row: (WORKLOADS.index(row.workload), PROTOCOLS.index(row.protocol)))
    size_rows.sort(key=lambda row: (WORKLOADS.index(row.workload), row.operation, PROTOCOLS.index(row.protocol)))
    return request_rows, phase_rows, summaries, size_rows


def sort_key(key: tuple[str, str]) -> tuple[int, str]:
    _category, operation = key
    if operation in REQUEST_ORDER:
        return REQUEST_ORDER.index(operation), operation
    return len(REQUEST_ORDER), operation


def weighted_average(rows: list[dict[str, int]]) -> tuple[float, int]:
    count = sum(row["count"] for row in rows)
    if count == 0:
        return 0.0, 0
    return sum(row["avg_us"] * row["count"] for row in rows) / count, count


def draw_workload_chart(
    workload: str,
    rows: list[RequestRow],
    phase_rows: list[RequestRow],
    output_path: pathlib.Path,
) -> None:
    workload_rows = [row for row in rows if row.workload == workload]
    operations = sorted(
        {row.operation for row in workload_rows},
        key=lambda operation: sort_key(("", operation)),
    )
    width = 1800
    row_height = 86
    margin_left = 390
    margin_right = 110
    margin_top = 150
    margin_bottom = 125
    plot_height = len(operations) * row_height
    height = margin_top + plot_height + margin_bottom
    plot_width = width - margin_left - margin_right
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(*COLORS["white"])
    ctx.paint()

    max_value = max((row.client_avg_us for row in workload_rows), default=1)
    x_max = nice_axis_max(max_value / 1000)

    draw_text(
        ctx,
        width / 2,
        48,
        f"{workload.capitalize()} Workload: Average Request Time by Operation",
        34,
        bold=True,
        align="center",
    )
    draw_text(
        ctx,
        width / 2,
        84,
        "Each bar totals client-observed latency; server time is split into measured phases where available, gray is uninstrumented server time, orange is client overhead.",
        20,
        color=COLORS["muted"],
        align="center",
    )
    draw_legend(ctx, margin_left, 116)

    # Grid.
    tick_step = choose_tick_step(x_max)
    for tick in frange(0, x_max, tick_step):
        x = margin_left + (tick / x_max) * plot_width
        ctx.set_source_rgb(*COLORS["grid"])
        ctx.set_line_width(1)
        ctx.move_to(x, margin_top - 8)
        ctx.line_to(x, margin_top + plot_height)
        ctx.stroke()
        draw_text(
            ctx,
            x,
            margin_top + plot_height + 35,
            f"{tick:.0f}",
            17,
            color=COLORS["muted"],
            align="center",
        )

    draw_text(
        ctx,
        margin_left + plot_width / 2,
        height - 35,
        "Average request latency (ms)",
        22,
        color=COLORS["muted"],
        align="center",
    )

    bar_height = 24
    for index, operation in enumerate(operations):
        center_y = margin_top + index * row_height + row_height / 2
        draw_text(
            ctx,
            margin_left - 18,
            center_y + 8,
            operation,
            20,
            align="right",
        )
        for protocol_index, protocol in enumerate(PROTOCOLS):
            row = find_row(workload_rows, operation, protocol)
            if row is None:
                continue
            y = center_y - 30 + protocol_index * 34
            cursor_x = margin_left
            phase_total_us = 0.0
            for phase in phases_for(phase_rows, workload, protocol, operation):
                phase_key = phase.operation.split(".", 1)[1]
                width_px = (phase.server_avg_us / 1000 / x_max) * plot_width
                ctx.set_source_rgb(*PHASE_COLORS.get(phase_key, COLORS["server_other"]))
                ctx.rectangle(cursor_x, y, width_px, bar_height)
                ctx.fill()
                cursor_x += width_px
                phase_total_us += phase.server_avg_us
            server_other_us = max(row.server_avg_us - phase_total_us, 0.0)
            server_other_width = (server_other_us / 1000 / x_max) * plot_width
            ctx.set_source_rgb(*COLORS["server_other"])
            ctx.rectangle(cursor_x, y, server_other_width, bar_height)
            ctx.fill()
            cursor_x += server_other_width
            overhead_width = (row.overhead_us / 1000 / x_max) * plot_width
            ctx.set_source_rgb(*COLORS["overhead"])
            ctx.rectangle(cursor_x, y, overhead_width, bar_height)
            ctx.fill()
            ctx.set_source_rgb(0.15, 0.16, 0.18)
            ctx.set_line_width(1)
            ctx.rectangle(margin_left, y, cursor_x - margin_left + overhead_width, bar_height)
            ctx.stroke()
            draw_text(ctx, margin_left - 8, y + 18, protocol, 14, color=COLORS["muted"], align="right")
            draw_text(
                ctx,
                margin_left + (row.client_avg_us / 1000 / x_max) * plot_width + 8,
                y + 18,
                f"{row.client_avg_us / 1000:.3f} ms",
                16,
                color=COLORS["text"],
            )

    # Axes.
    ctx.set_source_rgb(0.22, 0.24, 0.27)
    ctx.set_line_width(2)
    ctx.move_to(margin_left, margin_top - 8)
    ctx.line_to(margin_left, margin_top + plot_height)
    ctx.line_to(width - margin_right, margin_top + plot_height)
    ctx.stroke()
    surface.write_to_png(str(output_path))


def phases_for(
    rows: list[RequestRow],
    workload: str,
    protocol: str,
    operation: str,
) -> list[RequestRow]:
    prefix = f"{operation}."
    phases = [
        row
        for row in rows
        if row.workload == workload
        and row.protocol == protocol
        and row.operation.startswith(prefix)
    ]
    phases.sort(key=lambda row: phase_sort_key(row.operation))
    return phases


def phase_sort_key(operation: str) -> tuple[int, str]:
    phase_name = operation.split(".", 1)[1] if "." in operation else operation
    if phase_name in PHASE_ORDER:
        return PHASE_ORDER.index(phase_name), phase_name
    return len(PHASE_ORDER), phase_name


def find_row(rows: list[RequestRow], operation: str, protocol: str) -> RequestRow | None:
    for row in rows:
        if row.operation == operation and row.protocol == protocol:
            return row
    return None


def nice_axis_max(value: float) -> float:
    if value <= 0:
        return 1
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized <= 2:
        rounded = 2
    elif normalized <= 5:
        rounded = 5
    else:
        rounded = 10
    return rounded * magnitude


def choose_tick_step(x_max: float) -> float:
    return x_max / 5


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + step / 10:
        values.append(current)
        current += step
    return values


def draw_legend(ctx: cairo.Context, x: float, y: float) -> None:
    rows = [
        [
            ("DB add", PHASE_COLORS["db_add"]),
            ("DB register", PHASE_COLORS["db_register"]),
            ("Parse graph", PHASE_COLORS["parse_graph"]),
            ("Create JCB", PHASE_COLORS["create_jcb"]),
        ],
        [
            ("JCB start", PHASE_COLORS["jcb_start"]),
            ("Other server", COLORS["server_other"]),
            ("Client overhead", COLORS["overhead"]),
        ],
    ]
    draw_text(ctx, x, y - 16, "Server makeup", 17, bold=True)
    for row_index, items in enumerate(rows):
        for index, (label, color) in enumerate(items):
            item_x = x + index * 230
            item_y = y + row_index * 24
            ctx.set_source_rgb(*color)
            ctx.rectangle(item_x, item_y, 24, 16)
            ctx.fill()
            draw_text(ctx, item_x + 34, item_y + 15, label, 15, color=COLORS["muted"])



def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: int,
    color: tuple[float, float, float] = COLORS["text"],
    bold: bool = False,
    align: str = "left",
) -> None:
    ctx.select_font_face(
        "Sans",
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    _x_bearing, _y_bearing, width, _height, _x_advance, _y_advance = ctx.text_extents(value)
    if align == "center":
        x -= width / 2
    elif align == "right":
        x -= width
    ctx.set_source_rgb(*color)
    ctx.move_to(x, y)
    ctx.show_text(value)


def write_report(
    output_path: pathlib.Path,
    chart_paths: list[pathlib.Path],
    request_rows: list[RequestRow],
    phase_rows: list[RequestRow],
    summaries: list[RunSummary],
    size_rows: list[SizeRow],
) -> None:
    setup = summaries[0].setup
    summary_by_key = {(row.workload, row.protocol): row for row in summaries}
    lines = [
        "# Storage API Benchmark Report",
        "",
        "This report compares gRPC and REST for storage API benchmark runs. Request averages are weighted by request count. In each chart, every request type is shown for one workload, with gRPC and REST side by side. Each bar totals client-observed average request latency. The server-side portion is split into measured phases where available; gray is server time not covered by a phase probe, and orange is client-side overhead.",
        "",
        "## Setup",
        "",
        f"The storage API server ran on `baker3` at REST `http://10.1.0.3:8091` and gRPC `http://10.1.0.3:50051`; benchmark clients ran on `baker7`. Each run used `{setup['job_count']}` jobs, `{setup['task_count']}` tasks per job, `{setup['payload_bytes']}` byte payloads, `{setup['client_count']}` submit/monitor clients, `{setup['worker_count']}` workers, `{setup['poll_batch']}` poll batch size, and `{setup['poll_wait_ms']}` ms poll wait. The mixed workload used `{setup['flat_percent']}`% flat jobs and `{100 - setup['flat_percent']}`% deep jobs.",
        "",
        "## Request Latency Charts",
        "",
        "The chart makeup is useful for diagnosing whether a protocol difference is really transport overhead or time spent inside a storage phase. For example, `register_job.db_register` is the database insert phase, while `start_job.jcb_start` is cache/JCB scheduling work rather than a database request.",
        "",
    ]
    for chart_path in chart_paths:
        workload = chart_path.stem.rsplit("_", 1)[-1]
        lines.extend(
            [
                f"### {workload.capitalize()}",
                "",
                f"![{workload} request latency](./{chart_path.name})",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall Average Request Latency",
            "",
            "| Workload | Protocol | Requests | Server avg (us) | Client overhead (us) | Client observed avg (us) | Client observed avg (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row.workload} | {row.protocol} | {row.requests:,} | {fmt_us(row.server_avg_us)} | {fmt_us(row.overhead_us)} | {fmt_us(row.client_avg_us)} | {fmt_ms(row.client_avg_us)} |"
        )

    lines.extend(
        [
            "",
            "## End-to-End Job Latency",
            "",
            "| Workload | Protocol | Jobs | Failed jobs | p50 (ms) | p90 (ms) | p99 (ms) | max (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        job = row.job_latency
        lines.append(
            f"| {row.workload} | {row.protocol} | {job['count']} | {job['failed_jobs']} | {fmt_ms(job['p50_us'])} | {fmt_ms(job['p90_us'])} | {fmt_ms(job['p99_us'])} | {fmt_ms(job['max_us'])} |"
        )

    if size_rows:
        lines.extend(size_summary_section(size_rows))

    lines.extend(
        [
            "",
            "## Request Latency Detail",
            "",
            "| Workload | Request | Protocol | Count | Server avg (us) | Client overhead (us) | Client observed avg (us) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in request_rows:
        lines.append(
            f"| {row.workload} | `{row.operation}` | {row.protocol} | {row.count:,} | {fmt_us(row.server_avg_us)} | {fmt_us(row.overhead_us)} | {fmt_us(row.client_avg_us)} |"
        )

    if phase_rows:
        lines.extend(
            [
                "",
                "## Server Phase Timing Detail",
                "",
                "| Workload | Phase | Protocol | Count | Server avg (us) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in phase_rows:
            lines.append(
                f"| {row.workload} | `{row.operation}` | {row.protocol} | {row.count:,} | {fmt_us(row.server_avg_us)} |"
            )

    lines.extend(diagnostic_conclusions(phase_rows, summaries))

    lines.extend(
        [
            "",
            "## Protocol Summary",
            "",
            "| Workload | Faster by avg request latency | gRPC avg request (ms) | REST avg request (ms) | Faster by e2e p50 | gRPC e2e p50 (ms) | REST e2e p50 (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for workload in WORKLOADS:
        grpc = summary_by_key[(workload, "Grpc")]
        rest = summary_by_key[(workload, "Rest")]
        request_winner = "gRPC" if grpc.client_avg_us < rest.client_avg_us else "REST"
        e2e_winner = (
            "gRPC"
            if grpc.job_latency["p50_us"] < rest.job_latency["p50_us"]
            else "REST"
        )
        lines.append(
            f"| {workload} | {request_winner} | {fmt_ms(grpc.client_avg_us)} | {fmt_ms(rest.client_avg_us)} | {e2e_winner} | {fmt_ms(grpc.job_latency['p50_us'])} | {fmt_ms(rest.job_latency['p50_us'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostic_conclusions(
    phase_rows: list[RequestRow],
    summaries: list[RunSummary],
) -> list[str]:
    phase_by_key = {
        (row.workload, row.protocol, row.operation): row for row in phase_rows
    }
    summary_by_key = {(row.workload, row.protocol): row for row in summaries}
    lines = [
        "",
        "## Detailed Findings",
        "",
    ]
    missing_phase_runs = [
        f"{row.protocol} {row.workload}"
        for row in summaries
        if not any(
            phase.workload == row.workload and phase.protocol == row.protocol
            for phase in phase_rows
        )
    ]
    if missing_phase_runs:
        lines.append(
            "Phase timing is missing for "
            + ", ".join(missing_phase_runs)
            + ". Those runs still have top-level server/client request timing, but should be rerun before making phase-level conclusions for them."
        )
        lines.append("")

    for workload in WORKLOADS:
        grpc = summary_by_key[(workload, "Grpc")]
        rest = summary_by_key[(workload, "Rest")]
        request_winner = "gRPC" if grpc.client_avg_us < rest.client_avg_us else "REST"
        e2e_winner = (
            "gRPC"
            if grpc.job_latency["p50_us"] < rest.job_latency["p50_us"]
            else "REST"
        )
        lines.append(
            f"- `{workload}`: {request_winner} has lower weighted average request latency "
            f"({fmt_ms(grpc.client_avg_us)} ms gRPC vs {fmt_ms(rest.client_avg_us)} ms REST). "
            f"{e2e_winner} has lower p50 e2e job latency "
            f"({fmt_ms(grpc.job_latency['p50_us'])} ms gRPC vs {fmt_ms(rest.job_latency['p50_us'])} ms REST)."
        )

    lines.append("")
    flat_grpc_db = phase_by_key.get(("flat", "Grpc", "register_job.db_register"))
    flat_rest_db = phase_by_key.get(("flat", "Rest", "register_job.db_register"))
    if flat_grpc_db is not None and flat_rest_db is not None:
        lines.append(
            "- For flat `register_job`, most server-side time is in `register_job.db_register` "
            f"({fmt_ms(flat_grpc_db.server_avg_us)} ms gRPC, {fmt_ms(flat_rest_db.server_avg_us)} ms REST). "
            "This points at database insert/pool behavior rather than graph parsing, validation, or cache insertion."
        )
    for workload in WORKLOADS:
        grpc_start = phase_by_key.get((workload, "Grpc", "start_job.jcb_start"))
        rest_start = phase_by_key.get((workload, "Rest", "start_job.jcb_start"))
        if grpc_start is not None and rest_start is not None:
            lines.append(
                f"- `{workload}` `start_job` is almost entirely `start_job.jcb_start`; `start_job.cache_get` is near zero. "
                f"The JCB start phase is {fmt_ms(grpc_start.server_avg_us)} ms for gRPC and {fmt_ms(rest_start.server_avg_us)} ms for REST."
            )
    lines.append(
        "- `register_job.cache_insert`, `register_job.validate`, and `start_job.cache_get` are consistently tiny. They are unlikely to explain the protocol differences."
    )
    lines.append(
        "- The main remaining suspects are database insert/pool behavior for DB-backed operations and JCB start scheduling for `start_job`. The charts now separate those paths so the next benchmark run can confirm whether the gRPC/REST gap is coming from DB registration or non-DB scheduling."
    )
    return lines


def fmt_us(value: float) -> str:
    return f"{value:,.1f}"


def fmt_ms(value: float) -> str:
    return f"{value / 1000:,.3f}"


def fmt_bytes(value: float) -> str:
    return f"{value:,.0f}"


def size_summary_section(size_rows: list[SizeRow]) -> list[str]:
    by_workload_op: dict[tuple[str, str], dict[str, SizeRow]] = {}
    for row in size_rows:
        by_workload_op.setdefault((row.workload, row.operation), {})[row.protocol] = row

    by_op: dict[str, list[SizeRow]] = {}
    for row in size_rows:
        by_op.setdefault(row.operation, []).append(row)
    operations = sorted(by_op.keys())

    pair_operations = sorted(
        {
            operation
            for operation in operations
            if any(
                operation.endswith("_uncompressed_bytes")
                and operation.removesuffix("_uncompressed_bytes") + "_compressed_bytes" in by_op
                for _ in [None]
            )
        }
    )

    lines: list[str] = ["", "## Task Graph Upload Size", ""]
    lines.append(
        "Bytes recorded per `register_job` call on the storage server. `Uncompressed` is the "
        "JSON encoding of the task graph; `Compressed` is what is bound to the "
        "`serialized_task_graph` column (zstd level 3). Ratio is `compressed / uncompressed`."
    )
    lines.append("")

    if pair_operations:
        lines.extend(
            [
                "### Compression Ratio",
                "",
                "| Workload | Protocol | Operation | Count | Uncompressed avg (B) | Compressed avg (B) | Ratio |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for workload in WORKLOADS:
            for protocol in PROTOCOLS:
                for pair_op in pair_operations:
                    base = pair_op.removesuffix("_uncompressed_bytes")
                    compressed_op = base + "_compressed_bytes"
                    pair_uncompressed = by_workload_op.get((workload, pair_op), {}).get(protocol)
                    pair_compressed = by_workload_op.get((workload, compressed_op), {}).get(
                        protocol
                    )
                    if pair_uncompressed is None or pair_compressed is None:
                        continue
                    if pair_uncompressed.avg_bytes <= 0:
                        ratio_text = "n/a"
                    else:
                        ratio_text = (
                            f"{pair_compressed.avg_bytes / pair_uncompressed.avg_bytes:.3f}"
                        )
                    lines.append(
                        f"| {workload} | {protocol} | `{base}` | {pair_uncompressed.count:,} | "
                        f"{fmt_bytes(pair_uncompressed.avg_bytes)} | "
                        f"{fmt_bytes(pair_compressed.avg_bytes)} | {ratio_text} |"
                    )
        lines.append("")

    lines.extend(
        [
            "### Per-Operation Size Detail",
            "",
            "| Workload | Protocol | Operation | Count | avg (B) | p50 (B) | p99 (B) | max (B) | total (B) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in size_rows:
        lines.append(
            f"| {row.workload} | {row.protocol} | `{row.operation}` | {row.count:,} | "
            f"{fmt_bytes(row.avg_bytes)} | {fmt_bytes(row.p50_bytes)} | "
            f"{fmt_bytes(row.p99_bytes)} | {fmt_bytes(row.max_bytes)} | "
            f"{fmt_bytes(row.total_bytes)} |"
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
