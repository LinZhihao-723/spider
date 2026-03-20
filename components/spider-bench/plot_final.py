#!/usr/bin/env python3
"""
Generate two sets of 4 plots:
  - Set A: without 512 workers (final_e2e_latency.png, etc.)
  - Set B: with 512 workers (final_e2e_latency_with512.png, etc.)
Each set has:
  1. End-to-end server total time vs workers + ideal
  2. Per-task latency: client register (pre-exec) + submit (post-exec), server register + submit
  3. Absolute gap: actual - ideal (ms)
  4. Percentage overhead: (actual - ideal) / ideal * 100
"""
import csv, sys, os
from collections import defaultdict

def trimmed_mean(values):
    """Drop the 2 highest and 2 lowest, average the rest."""
    if len(values) <= 4:
        return sum(values) / len(values) if values else 0
    s = sorted(values)
    mid = s[2:-2]  # drop bottom 2 and top 2
    return sum(mid) / len(mid)

def trimmed_std(values):
    if len(values) <= 4:
        return 0
    s = sorted(values)
    mid = s[2:-2]
    m = sum(mid) / len(mid)
    return (sum((x - m) ** 2 for x in mid) / len(mid)) ** 0.5

def generate_plots(data, workers, output_dir, suffix=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    combos = sorted(set((k[0], k[1]) for k in data.keys() if k[2] in workers))
    ideal = {w: 10000.0 * 10.0 / w for w in workers}

    colors = {"flat": {"none": "#1f77b4", "zstd": "#ff7f0e"},
              "neural-net": {"none": "#2ca02c", "zstd": "#d62728"}}
    markers = {"flat": "o", "neural-net": "s"}
    ls_map = {"none": "-", "zstd": "--"}

    def series(bench, comp, col):
        m, s = [], []
        for w in workers:
            vals = [v for v in data.get((bench, comp, w), {}).get(col, []) if v > 0]
            m.append(trimmed_mean(vals) if vals else 0)
            s.append(trimmed_std(vals) if vals else 0)
        return np.array(m), np.array(s)

    def style(bench, comp):
        return (colors.get(bench, {}).get(comp, "gray"),
                markers.get(bench, "o"),
                ls_map.get(comp, "-"),
                f"{bench} ({comp})")

    tag = f" (incl. 512)" if 512 in workers else ""
    sfx = suffix

    # Plot 1: E2E
    fig, ax = plt.subplots(figsize=(10, 6))
    for bench, comp in combos:
        m, sd = series(bench, comp, "server_total_ms")
        c, mk, ls, label = style(bench, comp)
        ax.errorbar(workers, m, yerr=sd, fmt=f"{mk}{ls}", color=c, linewidth=2, capsize=4, label=label)
    ax.plot(workers, [ideal[w] for w in workers], "x:", color="black", linewidth=2, markersize=8, label="Ideal (10k×10ms/workers)")
    ax.set_xlabel("Number of Workers", fontsize=13); ax.set_ylabel("Server Total Time (ms)", fontsize=13)
    ax.set_title(f"End-to-End Latency{tag}\n(10 runs, drop top/bottom 2, avg mid 6)", fontsize=13)
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2); ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    p = os.path.join(output_dir, f"final_e2e_latency{sfx}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"Saved: {p}"); plt.close()

    # Plot 2: Per-task latency
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (col, title) in enumerate([
        ("client_reg_avg_ms", "Client: Register (pre-execution)"),
        ("client_sub_avg_ms", "Client: Submit (post-execution)"),
        ("server_reg_avg_ms", "Server: Register (pre-execution)"),
        ("server_sub_avg_ms", "Server: Submit (post-execution)"),
    ]):
        ax = axes[idx // 2][idx % 2]
        for bench, comp in combos:
            m, sd = series(bench, comp, col)
            c, mk, ls, label = style(bench, comp)
            ax.errorbar(workers, m, yerr=sd, fmt=f"{mk}{ls}", color=c, linewidth=2, capsize=3, label=label)
        ax.set_xlabel("Workers", fontsize=11); ax.set_ylabel("Avg Latency (ms)", fontsize=11)
        ax.set_title(title, fontsize=12); ax.set_ylim(bottom=0)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2); ax.set_xticks(workers)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.suptitle(f"Per-task Avg Latency{tag}\n(10 runs, drop top/bottom 2, avg mid 6)", fontsize=14)
    plt.tight_layout()
    p = os.path.join(output_dir, f"final_pertask_latency{sfx}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"Saved: {p}"); plt.close()

    # Plot 3: Absolute overhead
    fig, ax = plt.subplots(figsize=(10, 6))
    for bench, comp in combos:
        m, sd = series(bench, comp, "server_total_ms")
        gap = m - np.array([ideal[w] for w in workers])
        c, mk, ls, label = style(bench, comp)
        ax.errorbar(workers, gap, yerr=sd, fmt=f"{mk}{ls}", color=c, linewidth=2, capsize=4, label=label)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Number of Workers", fontsize=13); ax.set_ylabel("Overhead (ms)", fontsize=13)
    ax.set_title(f"Scheduling Overhead: Actual − Ideal{tag}\n(10 runs, drop top/bottom 2, avg mid 6)", fontsize=13)
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2); ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    p = os.path.join(output_dir, f"final_absolute_overhead{sfx}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"Saved: {p}"); plt.close()

    # Plot 4: Percentage overhead
    fig, ax = plt.subplots(figsize=(10, 6))
    for bench, comp in combos:
        m, sd = series(bench, comp, "server_total_ms")
        id_arr = np.array([ideal[w] for w in workers])
        pct = (m - id_arr) / id_arr * 100
        pct_sd = sd / id_arr * 100
        c, mk, ls, label = style(bench, comp)
        ax.errorbar(workers, pct, yerr=pct_sd, fmt=f"{mk}{ls}", color=c, linewidth=2, capsize=4, label=label)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Number of Workers", fontsize=13); ax.set_ylabel("Overhead (%)", fontsize=13)
    ax.set_title(f"Scheduling Framework Overhead{tag}\n(10 runs, drop top/bottom 2, avg mid 6)", fontsize=13)
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2); ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    p = os.path.join(output_dir, f"final_percentage_overhead{sfx}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); print(f"Saved: {p}"); plt.close()

def main():
    csv_file, output_dir = sys.argv[1], sys.argv[2]

    data = defaultdict(lambda: defaultdict(list))
    skipped = 0
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            tasks = int(row.get("tasks_registered", 0))
            if tasks != 10000:
                print(f"WARNING: incomplete run skipped — {row['benchmark']}/{row['compression']} "
                      f"w={row['workers']} run={row['run']}: only {tasks}/10000 tasks")
                skipped += 1
                continue
            k = (row["benchmark"], row["compression"], int(row["workers"]))
            for col in ["server_total_ms", "client_reg_avg_ms", "client_sub_avg_ms",
                        "server_reg_avg_ms", "server_sub_avg_ms"]:
                data[k][col].append(float(row[col]))
    if skipped:
        print(f"WARNING: {skipped} incomplete run(s) skipped total")

    all_workers = sorted(set(k[2] for k in data.keys()))
    without_512 = [w for w in all_workers if w != 512]

    # Set A: without 512
    print("\n--- Generating plots without 512 ---")
    generate_plots(data, without_512, output_dir, suffix="")

    # Set B: with 512 (only if we have 512 data)
    if 512 in all_workers:
        print("\n--- Generating plots with 512 ---")
        generate_plots(data, all_workers, output_dir, suffix="_with512")
    else:
        print("No 512-worker data found, skipping _with512 plots.")

if __name__ == "__main__":
    main()
