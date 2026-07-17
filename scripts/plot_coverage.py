"""Plots per-species surface coverage over simulated time from
`coverage_report`'s CSV output.

    cargo run --release --bin coverage_report -- \\
        --trajectory-path trajectory.bin --lut-path reactions.lut \\
        --lattice-width <W> --lattice-height <H> \\
        > coverage.csv
    python3 scripts/plot_coverage.py coverage.csv coverage.png

Only used to produce the README's coverage-over-time figure -- not part of
the simulation or ingestion pipeline itself, so it's the one script in
this repo that assumes `matplotlib` is installed (`pip install
matplotlib`), rather than sticking to the standard library.
"""

import csv
import sys


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <coverage.csv> <out.png>", file=sys.stderr)
        return 1
    csv_path, out_path = sys.argv[1], sys.argv[2]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    species = [k for k in rows[0].keys() if k not in ("event", "sim_time")]
    sim_time_us = [float(r["sim_time"]) * 1e6 for r in rows]
    total = sum(float(rows[0][s]) for s in species)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = {
        "vacant": "#999999",
        "O": "#d62728",
        "H": "#1f77b4",
        "CO": "#2ca02c",
        "OH": "#9467bd",
        "H2O": "#ff7f0e",
    }
    for s in species:
        pct = [100.0 * float(r[s]) / total for r in rows]
        ax.plot(
            sim_time_us,
            pct,
            label=f"{s}*" if s != "vacant" else "vacant",
            color=colors.get(s),
            linewidth=2.0,
        )

    ax.set_xlabel("simulated time (microseconds)")
    ax.set_ylabel("surface coverage (%)")
    ax.set_title("Real Pd(111) surface coverage vs. simulated time")
    ax.legend(loc="center right", frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
