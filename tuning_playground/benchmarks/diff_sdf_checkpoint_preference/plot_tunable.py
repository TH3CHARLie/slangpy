"""Plot the bool-controlled diff_sdf checkpoint sweep.

Input:  results_diff_sdf_checkpoint_preference.jsonl
Output: spread_tunable.png
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = pathlib.Path(__file__).parent
RESULT = THIS_DIR / "results_diff_sdf_checkpoint_preference.jsonl"
OUT = THIS_DIR / "spread_tunable.png"

AXIS_ORDER = ["SchedMarch", "SchedSDF", "SchedMLP"]


def load(path: pathlib.Path) -> tuple[dict | None, list[dict]]:
    meta = None
    rows = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            elif "error" not in d:
                rows.append(d)
    return meta, rows


def sched_tag(sched: dict) -> str:
    tag = ""
    for axis in AXIS_ORDER:
        tag += "R" if "Recompute" in sched[axis] else "C"
    return tag


def main() -> None:
    meta, rows = load(RESULT)
    if not rows:
        raise SystemExit(f"no rows in {RESULT.name}")

    for r in rows:
        r["_tag"] = sched_tag(r["sched"])

    by_shape: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["shape"]["n_steps"], r["shape"]["width"])
        by_shape[key].append(r)

    shapes = sorted(by_shape.keys())
    shape_labels = [f"n={n},w={w}" for (n, w) in shapes]
    tags = sorted({r["_tag"] for r in rows})
    cmap = plt.get_cmap("tab10")
    colors = {t: cmap(i % 10) for i, t in enumerate(tags)}

    fig, (ax_bar, ax_spread) = plt.subplots(1, 2, figsize=(14, 5.4))

    n_tags = len(tags)
    group_width = 0.84
    bar_w = group_width / n_tags
    x_base = np.arange(len(shapes))

    for ti, tag in enumerate(tags):
        ys = []
        for key in shapes:
            sub = [r for r in by_shape[key] if r["_tag"] == tag]
            ys.append(sub[0]["iters_per_s"] if sub else 0.0)
        offsets = x_base + (ti - (n_tags - 1) / 2) * bar_w
        ax_bar.bar(offsets, ys, width=bar_w, color=colors[tag], label=tag, edgecolor="none")

    for si, key in enumerate(shapes):
        winner = max(by_shape[key], key=lambda r: r["iters_per_s"])
        ti = tags.index(winner["_tag"])
        offset = x_base[si] + (ti - (n_tags - 1) / 2) * bar_w
        ax_bar.bar(
            offset,
            winner["iters_per_s"],
            width=bar_w,
            facecolor="none",
            edgecolor="black",
            linewidth=1.4,
        )

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(shape_labels, rotation=30)
    ax_bar.set_ylabel("Training throughput (iters/s)")
    ax_bar.set_title("fwd+bwd+sgd throughput per schedule")
    ax_bar.legend(
        loc="upper right",
        fontsize=8,
        ncol=4,
        title="Ma/SDF/MLP (C=Checkpoint, R=Recompute)",
        title_fontsize=8,
    )
    ax_bar.grid(True, axis="y", which="both", alpha=0.3)

    spreads = []
    for key in shapes:
        ts = [r["iters_per_s"] for r in by_shape[key]]
        spreads.append(max(ts) / min(ts) if min(ts) > 0 else 1.0)

    bars = ax_spread.bar(x_base, spreads, color="#b0c4de", edgecolor="black", linewidth=0.5)
    for rect, spread in zip(bars, spreads):
        ax_spread.text(
            rect.get_x() + rect.get_width() / 2,
            spread + 0.02,
            f"{spread:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for si, key in enumerate(shapes):
        winner = max(by_shape[key], key=lambda r: r["iters_per_s"])
        ax_spread.text(x_base[si], 1.02, winner["_tag"], ha="center", va="bottom", fontsize=8)

    ax_spread.axhline(1.0, color="black", linewidth=0.6)
    ax_spread.set_xticks(x_base)
    ax_spread.set_xticklabels(shape_labels, rotation=30)
    ax_spread.set_ylabel("Schedule spread (max / min throughput)")
    ax_spread.set_title("Bool checkpoint preference schedule spread")
    ax_spread.grid(True, axis="y", alpha=0.3)
    ax_spread.set_ylim(1.0, max(spreads) * 1.2)

    all_thr = [r["iters_per_s"] for r in rows]
    global_spread = max(all_thr) / min(all_thr)
    adapter = (meta or {}).get("adapter", "unknown GPU")
    winners = {max(by_shape[k], key=lambda r: r["iters_per_s"])["_tag"] for k in shapes}
    fig.suptitle(
        f"Slang autotuning - bool checkpoint preference on diff_sdf "
        f"({len(rows)} trials / {len(shapes)} shapes / {n_tags} configs, {adapter}) "
        f"| peak shape spread {max(spreads):.2f}x | {len(winners)} distinct winners",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print(
        f"  shapes: {len(shapes)}  trials: {len(rows)}  "
        f"global thr {min(all_thr):.1f}-{max(all_thr):.1f} iters/s "
        f"(spread {global_spread:.2f}x)"
    )
    print(f"  per-shape spreads: {[f'{s:.2f}' for s in spreads]}")
    print(f"  distinct winners: {len(winners)} / {n_tags}   ({sorted(winners)})")


if __name__ == "__main__":
    main()
