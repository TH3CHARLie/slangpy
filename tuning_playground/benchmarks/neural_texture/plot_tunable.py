"""Plot the tunable-driven neural_texture sweep.

Left:   per-shape grouped bar chart of throughput across the 7 activation
        impls. The fastest activation per shape is outlined in black.
Right:  per-shape spread (max/min throughput) — how much the activation
        choice actually matters at each (width, depth).

Input:  results_tunable.jsonl (produced by bench_tunable.py).
Output: spread_tunable.png.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = pathlib.Path(__file__).parent
RESULT = THIS_DIR / "results_tunable.jsonl"
OUT = THIS_DIR / "spread_tunable.png"


def load(path: pathlib.Path) -> tuple[dict, list[dict]]:
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


def shape_label(s: dict) -> str:
    return f"w{s['hidden_width']},d{s['hidden_depth']}"


def main() -> None:
    meta, rows = load(RESULT)
    if not rows:
        raise SystemExit("no rows in results_tunable.jsonl")

    # Group by shape
    by_shape: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["shape"]["hidden_width"], r["shape"]["hidden_depth"],
               r["shape"]["freq_bands"])
        by_shape[key].append(r)

    # Deterministic shape order: by width, then depth
    shapes = sorted(by_shape.keys())
    shape_labels = [f"w{w},d{d}" for (w, d, _) in shapes]

    # All activations seen (stable order across shapes)
    acts = sorted({r["activation"] for r in rows})
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i) for i, a in enumerate(acts)}

    fig, (ax_bar, ax_spread) = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- Grouped bar: one group per shape, one bar per activation ---
    n_acts = len(acts)
    group_width = 0.8
    bar_w = group_width / n_acts
    x_base = np.arange(len(shapes))

    for ai, a in enumerate(acts):
        ys = []
        for key in shapes:
            sub = [r for r in by_shape[key] if r["activation"] == a]
            ys.append(sub[0]["msamples_per_s"] if sub else 0.0)
        offsets = x_base + (ai - (n_acts - 1) / 2) * bar_w
        ax_bar.bar(offsets, ys, width=bar_w, color=colors[a], label=a,
                   edgecolor="none")

    # Highlight per-shape winner
    for si, key in enumerate(shapes):
        trials = by_shape[key]
        winner = max(trials, key=lambda r: r["msamples_per_s"])
        ai = acts.index(winner["activation"])
        offset = x_base[si] + (ai - (n_acts - 1) / 2) * bar_w
        ax_bar.bar(offset, winner["msamples_per_s"], width=bar_w,
                   facecolor="none", edgecolor="black", linewidth=1.4)

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(shape_labels, rotation=30)
    ax_bar.set_ylabel("Throughput (MSamples/s)")
    ax_bar.set_yscale("log")
    ax_bar.set_title("Throughput per activation, per shape  (black outline = winner)")
    ax_bar.legend(loc="upper right", fontsize=8, ncol=2)
    ax_bar.grid(True, axis="y", which="both", alpha=0.3)

    # --- Per-shape spread bar ---
    spreads = []
    for key in shapes:
        ts = [r["msamples_per_s"] for r in by_shape[key]]
        spreads.append(max(ts) / min(ts) if min(ts) > 0 else 1.0)

    bars = ax_spread.bar(x_base, spreads, color="#b0c4de", edgecolor="black",
                         linewidth=0.5)
    for rect, s in zip(bars, spreads):
        ax_spread.text(rect.get_x() + rect.get_width() / 2, s + 0.02,
                       f"{s:.2f}×", ha="center", va="bottom", fontsize=8)

    ax_spread.axhline(1.0, color="black", linewidth=0.6)
    ax_spread.set_xticks(x_base)
    ax_spread.set_xticklabels(shape_labels, rotation=30)
    ax_spread.set_ylabel("Activation spread (max / min throughput)")
    ax_spread.set_title("How much does activation choice matter?")
    ax_spread.grid(True, axis="y", alpha=0.3)
    ax_spread.set_ylim(1.0, max(spreads) * 1.15)

    # --- Headline ---
    all_thr = [r["msamples_per_s"] for r in rows]
    global_spread = max(all_thr) / min(all_thr)
    adapter = (meta or {}).get("adapter", "unknown GPU")
    n_winners = len({max(by_shape[k], key=lambda r: r["msamples_per_s"])["activation"]
                     for k in shapes})
    fig.suptitle(
        f"Slang autotuning — activation axis on neural_texture "
        f"({len(rows)} trials / {len(shapes)} shapes / {n_acts} activations, {adapter})  "
        f"| global thr spread {global_spread:.1f}×  | {n_winners} distinct winners",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print(f"  shapes: {len(shapes)}  trials: {len(rows)}  "
          f"global thr {min(all_thr):.1f}–{max(all_thr):.1f} MS/s "
          f"(spread {global_spread:.1f}×)")
    print(f"  per-shape spreads: {[f'{s:.2f}' for s in spreads]}")
    print(f"  distinct winners: {n_winners} / {n_acts}")


if __name__ == "__main__":
    main()
