"""Generate paper-facing figures/tables for the minimized-unroll sweep."""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt


THIS_DIR = pathlib.Path(__file__).parent
RESULTS = THIS_DIR / "results_minimized_unroll_minimal_flip.jsonl"
RESOURCE = THIS_DIR / "resource_stats_original.json"
OUT_FIG = THIS_DIR / "minimized_unroll_flip.png"
OUT_CROSSOVER_TABLE = THIS_DIR / "minimized_unroll_flip_table.md"
OUT_TABLE = THIS_DIR / "minimized_unroll_flip_resource_table.md"


def load_jsonl(path: pathlib.Path) -> tuple[dict, list[dict]]:
    meta = {}
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                rows.append(row)
    return meta, rows


def paired_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        key = (
            row["workload"],
            row["feature_width"],
            row["merged_signal_b"],
            row["effective_iters"],
        )
        grouped[key][row["strategy"]] = row

    pairs = []
    for (workload, feature_width, merged_signal_b, iters), by_strategy in grouped.items():
        if "ForceUnrollLoop" not in by_strategy or "MaxItersLoop" not in by_strategy:
            continue
        force = by_strategy["ForceUnrollLoop"]
        maxiters = by_strategy["MaxItersLoop"]
        pairs.append(
            {
                "workload": workload,
                "feature_width": feature_width,
                "merged_signal_b": merged_signal_b,
                "effective_iters": iters,
                "force_us": force["us_per_call"],
                "maxiters_us": maxiters["us_per_call"],
                "speedup": force["us_per_call"] / maxiters["us_per_call"],
            }
        )
    return sorted(
        pairs,
        key=lambda r: (0 if r["workload"] == "small" else 1, r["effective_iters"]),
    )


def write_crossover_plot(meta: dict, pairs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(
        [p["effective_iters"] for p in pairs],
        [p["speedup"] for p in pairs],
        marker="o",
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.5,
        s=58,
        alpha=0.9,
    )

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    for p in pairs:
        if (p["workload"], p["effective_iters"]) in {("small", 27), ("large", 75)}:
            ax.scatter(
                [p["effective_iters"]],
                [p["speedup"]],
                marker="*",
                color="black",
                s=120,
                zorder=5,
            )
            ax.annotate(
                f"{p['effective_iters']} iters",
                (p["effective_iters"], p["speedup"]),
                textcoords="offset points",
                xytext=(6, 7),
                fontsize=8,
            )
    ax.set_xlabel("Total unrolled iterations")
    ax.set_ylabel("Runtime ratio (ForceUnroll / MaxIters)")
    ax.set_title("Loop strategy ratio by total unrolled iterations")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.05, y=0.08)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180)


def write_resource_table(pairs: list[dict]) -> None:
    resource = json.loads(RESOURCE.read_text())
    runtime_by_key = {
        (p["workload"], p["effective_iters"], "ForceUnrollLoop"): p["force_us"]
        for p in pairs
    }
    runtime_by_key.update(
        {
            (p["workload"], p["effective_iters"], "MaxItersLoop"): p["maxiters_us"]
            for p in pairs
        }
    )

    lines = [
        "# Minimized Unroll Resource Evidence",
        "",
        "Resource stats come from `minimized-unroll-bench/build_and_run.sh` with "
        "`nvcc --ptxas-options=-v` on `sm_86`. Runtime is from "
        f"`{RESULTS.name}`.",
        "",
        "| workload | iters | strategy | runtime us | regs/thread | stack B | spill stores B | spill loads B |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in resource["rows"]:
        runtime = runtime_by_key.get(
            (row["workload"], row["effective_iters"], row["strategy"])
        )
        runtime_text = f"{runtime:.1f}" if runtime is not None else "n/a"
        lines.append(
            f"| {row['workload']} | {row['effective_iters']} | {row['strategy']} "
            f"| {runtime_text} | {row['registers']} | {row['stack_frame_bytes']} "
            f"| {row['spill_stores_bytes']} | {row['spill_loads_bytes']} |"
        )

    large_force = runtime_by_key[("large", 75, "ForceUnrollLoop")]
    large_maxiters = runtime_by_key[("large", 75, "MaxItersLoop")]
    small_force = runtime_by_key[("small", 27, "ForceUnrollLoop")]
    small_maxiters = runtime_by_key[("small", 27, "MaxItersLoop")]
    lines += [
        "",
        "Summary:",
        "",
        f"- Original small point, 27 iters: ratio {small_force / small_maxiters:.2f}x.",
        f"- Original large point, 75 iters: ratio {large_force / large_maxiters:.2f}x.",
        "- The large ForceUnroll backward kernel hits 255 registers and spills; "
        "the MaxIters version does not spill.",
    ]
    OUT_TABLE.write_text("\n".join(lines) + "\n")


def write_crossover_table(pairs: list[dict]) -> None:
    lines = [
        "# Minimized Unroll Crossover Data",
        "",
        "| workload | iters | feature width | merged B | ForceUnroll us | MaxIters us | ratio | winner |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for p in pairs:
        if abs(p["speedup"] - 1.0) < 0.02:
            winner = "Tie"
        else:
            winner = "MaxIters" if p["speedup"] > 1.0 else "ForceUnroll"
        lines.append(
            f"| {p['workload']} | {p['effective_iters']} | {p['feature_width']} "
            f"| {p['merged_signal_b']} | {p['force_us']:.1f} | {p['maxiters_us']:.1f} "
            f"| {p['speedup']:.2f} | {winner} |"
        )
    OUT_CROSSOVER_TABLE.write_text("\n".join(lines) + "\n")


def main() -> None:
    meta, rows = load_jsonl(RESULTS)
    pairs = paired_rows(rows)
    write_crossover_plot(meta, pairs)
    write_crossover_table(pairs)
    write_resource_table(pairs)
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_CROSSOVER_TABLE}")
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
