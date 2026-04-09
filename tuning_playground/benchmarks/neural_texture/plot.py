"""Plot the neural-texture sweep: Pareto (throughput vs loss) + per-activation spread."""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = pathlib.Path(__file__).parent
RESULTS = THIS_DIR / "results.jsonl"
OUT = THIS_DIR / "spread.png"


def load() -> tuple[dict, list[dict]]:
    meta = None
    rows = []
    with RESULTS.open() as f:
        for line in f:
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            elif "error" not in d:
                rows.append(d)
    return meta, rows


def main() -> None:
    meta, rows = load()
    if not rows:
        raise SystemExit("no successful rows in results.jsonl")

    acts = sorted({r["config"]["activation"] for r in rows})
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i) for i, a in enumerate(acts)}

    fig, (ax_pareto, ax_box) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Pareto: throughput (x) vs loss (y) ---
    for a in acts:
        sub = [r for r in rows if r["config"]["activation"] == a]
        xs = [r["msamples_per_s"] for r in sub]
        ys = [r["final_loss"] for r in sub]
        sizes = [18 + 6 * r["config"]["hidden_width"] ** 0.5 for r in sub]
        ax_pareto.scatter(xs, ys, s=sizes, c=[colors[a]], alpha=0.75, label=a,
                          edgecolors="black", linewidths=0.4)

    # Pareto frontier (min loss for given-or-better throughput)
    pts = sorted([(r["msamples_per_s"], r["final_loss"], r["config"]) for r in rows],
                 key=lambda t: -t[0])
    frontier = []
    best_loss = float("inf")
    for x, y, c in pts:
        if y < best_loss:
            best_loss = y
            frontier.append((x, y, c))
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax_pareto.plot(fx, fy, "k--", lw=1.0, alpha=0.55, label="Pareto frontier")

    ax_pareto.set_xlabel("Throughput (MSamples/s)")
    ax_pareto.set_ylabel("Final L2 loss (lower is better)")
    ax_pareto.set_yscale("log")
    ax_pareto.grid(True, alpha=0.3)
    ax_pareto.legend(loc="upper right", fontsize=8, ncol=2)
    ax_pareto.set_title("Throughput–Quality Pareto (84 configs)")

    # --- Box: throughput spread per activation ---
    data = [[r["msamples_per_s"] for r in rows if r["config"]["activation"] == a] for a in acts]
    bp = ax_box.boxplot(data, labels=acts, patch_artist=True, showfliers=True)
    for patch, a in zip(bp["boxes"], acts):
        patch.set_facecolor(colors[a])
        patch.set_alpha(0.6)
    ax_box.set_ylabel("Throughput (MSamples/s)")
    ax_box.set_title("Throughput spread across (width, depth) per activation")
    ax_box.grid(True, axis="y", alpha=0.3)
    ax_box.tick_params(axis="x", rotation=30)

    # --- Global annotation: overall spread numbers ---
    all_thr = [r["msamples_per_s"] for r in rows]
    all_loss = [r["final_loss"] for r in rows]
    spread_thr = max(all_thr) / min(all_thr)
    spread_loss = max(all_loss) / min(all_loss)
    adapter = (meta or {}).get("adapter", "unknown GPU")
    fig.suptitle(
        f"Neural-texture sweep — {adapter} (float, {len(rows)} configs) "
        f"| throughput spread {spread_thr:.1f}×  |  loss spread {spread_loss:.1f}×",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print(f"  throughput: {min(all_thr):.2f} – {max(all_thr):.2f} MS/s (spread {spread_thr:.2f}×)")
    print(f"  loss:       {min(all_loss):.5f} – {max(all_loss):.5f} (spread {spread_loss:.2f}×)")
    best = min(rows, key=lambda r: r["final_loss"])
    fast = max(rows, key=lambda r: r["msamples_per_s"])
    print(f"  best loss:  {best['config']}  ({best['final_loss']:.5f} @ {best['msamples_per_s']:.2f} MS/s)")
    print(f"  fastest:    {fast['config']}  ({fast['msamples_per_s']:.2f} MS/s @ loss {fast['final_loss']:.5f})")


if __name__ == "__main__":
    main()
