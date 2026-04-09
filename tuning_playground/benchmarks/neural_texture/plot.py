"""Plot the neural-texture sweep: Pareto (throughput vs loss) + per-activation spread."""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = pathlib.Path(__file__).parent
RESULT_FILES = {
    "float": THIS_DIR / "results_float.jsonl",
    "half":  THIS_DIR / "results_half.jsonl",
}
OUT = THIS_DIR / "spread.png"


def load(path: pathlib.Path) -> tuple[dict, list[dict]]:
    meta = None
    rows = []
    if not path.exists():
        return meta, rows
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            elif "error" not in d:
                rows.append(d)
    return meta, rows


def main() -> None:
    datasets = {p: load(path) for p, path in RESULT_FILES.items()}
    datasets = {p: (m, r) for p, (m, r) in datasets.items() if r}
    if not datasets:
        raise SystemExit("no successful rows in any results file")

    all_rows = [r for _, rows in datasets.values() for r in rows]
    acts = sorted({r["config"]["activation"] for r in all_rows})
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i) for i, a in enumerate(acts)}
    markers = {"float": "o", "half": "^"}

    fig, (ax_pareto, ax_box) = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- Pareto: throughput (x) vs loss (y), both precisions overlaid ---
    for p, (_, rows) in datasets.items():
        for a in acts:
            sub = [r for r in rows if r["config"]["activation"] == a]
            if not sub:
                continue
            xs = [r["msamples_per_s"] for r in sub]
            ys = [r["final_loss"] for r in sub]
            sizes = [18 + 6 * r["config"]["hidden_width"] ** 0.5 for r in sub]
            ax_pareto.scatter(xs, ys, s=sizes, c=[colors[a]], alpha=0.72,
                              marker=markers[p],
                              label=f"{a} ({p})" if p == "half" else None,
                              edgecolors="black", linewidths=0.4)

    # Joint Pareto frontier across both precisions
    pts = sorted([(r["msamples_per_s"], r["final_loss"]) for r in all_rows],
                 key=lambda t: -t[0])
    frontier = []
    best_loss = float("inf")
    for x, y in pts:
        if y < best_loss:
            best_loss = y
            frontier.append((x, y))
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax_pareto.plot(fx, fy, "k--", lw=1.0, alpha=0.6, label="joint Pareto frontier")

    # Activation legend via color swatches
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=colors[a], markersize=7,
                            markeredgecolor="black", markeredgewidth=0.4,
                            label=a) for a in acts]
    shape_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="",
               markersize=7, label="float"),
        Line2D([0], [0], marker="^", color="gray", linestyle="",
               markersize=7, label="half+coopvec"),
    ]
    ax_pareto.legend(handles=color_handles + shape_handles, loc="upper right",
                     fontsize=8, ncol=2)
    ax_pareto.set_xlabel("Throughput (MSamples/s)")
    ax_pareto.set_ylabel("Final L2 loss (lower is better)")
    ax_pareto.set_xscale("log")
    ax_pareto.set_yscale("log")
    ax_pareto.grid(True, which="both", alpha=0.3)
    total_configs = sum(len(r) for _, r in datasets.values())
    ax_pareto.set_title(f"Throughput–Quality Pareto ({total_configs} configs)")

    # --- Box: throughput per activation, grouped by precision ---
    positions = np.arange(len(acts))
    width = 0.35
    for i, p in enumerate(["float", "half"]):
        if p not in datasets:
            continue
        _, rows = datasets[p]
        data = [[r["msamples_per_s"] for r in rows if r["config"]["activation"] == a]
                for a in acts]
        offset = (i - 0.5) * width
        bp = ax_box.boxplot(data, positions=positions + offset, widths=width * 0.85,
                            patch_artist=True, showfliers=True,
                            boxprops={"linewidth": 0.7},
                            whiskerprops={"linewidth": 0.7},
                            medianprops={"color": "black", "linewidth": 1.0})
        face = "#b0c4de" if p == "float" else "#ffb347"
        for patch in bp["boxes"]:
            patch.set_facecolor(face)
            patch.set_alpha(0.75)

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(acts, rotation=30)
    ax_box.set_ylabel("Throughput (MSamples/s)")
    ax_box.set_yscale("log")
    ax_box.set_title("Throughput spread per activation (float vs half+coopvec)")
    ax_box.grid(True, axis="y", which="both", alpha=0.3)
    from matplotlib.patches import Patch
    ax_box.legend(handles=[Patch(facecolor="#b0c4de", alpha=0.75, label="float"),
                           Patch(facecolor="#ffb347", alpha=0.75, label="half+coopvec")],
                  loc="upper right", fontsize=9)

    # --- Global headline ---
    thr_all = [r["msamples_per_s"] for r in all_rows]
    loss_all = [r["final_loss"] for r in all_rows]
    spread_thr = max(thr_all) / min(thr_all)
    spread_loss = max(loss_all) / min(loss_all)
    adapter = next(iter(datasets.values()))[0].get("adapter", "unknown GPU")
    fig.suptitle(
        f"Neural-texture sweep — {adapter} ({total_configs} configs, float + half) "
        f"| throughput spread {spread_thr:.1f}×  |  loss spread {spread_loss:.1f}×",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    for p, (_, rows) in datasets.items():
        thr = [r["msamples_per_s"] for r in rows]
        loss = [r["final_loss"] for r in rows]
        print(f"  [{p}] n={len(rows)}  thr {min(thr):.2f}-{max(thr):.2f} MS/s "
              f"(spread {max(thr)/min(thr):.2f}×)  loss {min(loss):.5f}-{max(loss):.5f} "
              f"(spread {max(loss)/min(loss):.2f}×)")
    print(f"  [joint] thr spread {spread_thr:.2f}×  loss spread {spread_loss:.2f}×")


if __name__ == "__main__":
    main()
