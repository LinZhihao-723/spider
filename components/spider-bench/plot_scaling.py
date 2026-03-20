#!/usr/bin/env python3
"""Plot server_total_time vs. number of workers from scaling benchmark CSV."""

import csv
import sys
import os
from collections import defaultdict

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <csv_file> <output_dir>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib/numpy not installed.")
        sys.exit(1)

    # Parse CSV: group by (benchmark, compression, total_workers) -> list of server_total_ms
    data = defaultdict(list)
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            key = (row["benchmark"], row["compression"], int(row["total_workers"]))
            val = float(row["server_total_ms"])
            if val > 0:
                data[key].append(val)

    # Group by (benchmark, compression) for plotting
    series = defaultdict(lambda: {"workers": [], "mean": [], "std": []})
    for (bench, comp, tw), values in sorted(data.items()):
        k = (bench, comp)
        series[k]["workers"].append(tw)
        series[k]["mean"].append(np.mean(values))
        series[k]["std"].append(np.std(values))

    # --- Plot: server total time vs workers ---
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"flat": {"none": "#1f77b4", "zstd": "#ff7f0e"},
              "neural-net": {"none": "#2ca02c", "zstd": "#d62728"}}
    markers = {"flat": "o", "neural-net": "s"}
    linestyles = {"none": "-", "zstd": "--"}

    for (bench, comp), s in sorted(series.items()):
        w = np.array(s["workers"])
        m = np.array(s["mean"])
        sd = np.array(s["std"])
        label = f"{bench} ({comp})"
        c = colors.get(bench, {}).get(comp, "gray")
        ax.errorbar(w, m, yerr=sd, marker=markers.get(bench, "o"),
                    linestyle=linestyles.get(comp, "-"), color=c,
                    label=label, linewidth=2, capsize=4, capthick=1.5)

    # Ideal line: 10k tasks × 10ms / num_workers, assuming zero overhead and perfect scaling.
    ideal_w = np.array([16, 32, 64, 128, 256])
    ideal_t = 10000.0 * 10.0 / ideal_w
    ax.plot(ideal_w, ideal_t, color="black", linestyle=":", linewidth=2,
            marker="x", markersize=8, label="Ideal (10k×10ms / workers)", zorder=0)

    ax.set_xlabel("Number of Workers", fontsize=13)
    ax.set_ylabel("Server Total Time (ms)", fontsize=13)
    ax.set_title("Server-side Total Time vs. Worker Count\n"
                 "(10k tasks, 10ms simulated execution, cross-node, 5 runs avg ± std)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.tight_layout()
    path = os.path.join(output_dir, "scaling_server_total_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
