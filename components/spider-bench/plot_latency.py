#!/usr/bin/env python3
"""Plot per-task avg latency (client + server, register + submit) vs. worker count."""
import csv, sys, os
from collections import defaultdict

def main():
    csv_file, output_dir = sys.argv[1], sys.argv[2]
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Parse: group by (benchmark, compression, total_workers) -> lists of each metric
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            k = (row["benchmark"], row["compression"], int(row["total_workers"]))
            for col in ["client_register_avg_ms", "client_submit_avg_ms",
                        "server_register_avg_ms", "server_submit_avg_ms"]:
                v = float(row[col])
                if v > 0:
                    data[k][col].append(v)

    # Aggregate into series per (benchmark, compression)
    metrics = ["client_register_avg_ms", "client_submit_avg_ms",
               "server_register_avg_ms", "server_submit_avg_ms"]
    titles = ["Client: RegisterTaskInstance", "Client: SubmitTaskResult",
              "Server: RegisterTaskInstance", "Server: SubmitTaskResult"]

    series = defaultdict(lambda: {m: {"w": [], "mean": [], "std": []} for m in metrics})
    for (bench, comp, tw), mdict in sorted(data.items()):
        k = (bench, comp)
        for m in metrics:
            vals = mdict.get(m, [])
            if vals:
                series[k][m]["w"].append(tw)
                series[k][m]["mean"].append(np.mean(vals))
                series[k][m]["std"].append(np.std(vals))

    colors = {"flat": {"none": "#1f77b4", "zstd": "#ff7f0e"},
              "neural-net": {"none": "#2ca02c", "zstd": "#d62728"}}
    markers = {"flat": "o", "neural-net": "s"}
    ls_map = {"none": "-", "zstd": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2][idx % 2]
        for (bench, comp), s in sorted(series.items()):
            d = s[metric]
            if not d["w"]:
                continue
            w = np.array(d["w"])
            m = np.array(d["mean"])
            sd = np.array(d["std"])
            c = colors.get(bench, {}).get(comp, "gray")
            mk = markers.get(bench, "o")
            ls = ls_map.get(comp, "-")
            label = f"{bench} ({comp})"
            ax.errorbar(w, m, yerr=sd, marker=mk, linestyle=ls, color=c,
                        label=label, linewidth=2, capsize=4, capthick=1.5)

        ax.set_xlabel("Number of Workers", fontsize=11)
        ax.set_ylabel("Avg Latency (ms)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 32, 64, 128, 256])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.suptitle("Per-task Avg Latency vs. Worker Count\n"
                 "(10k tasks, 10ms simulated execution, cross-node, 5 runs avg ± std)",
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "scaling_avg_latency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

if __name__ == "__main__":
    main()
