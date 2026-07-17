"""Plots a spatial snapshot of the catalyst surface's *final* occupancy
state, from the raw `surface.lattice` file a real run leaves behind:

    ./target/release/kinetica \\
        --lattice-width <W> --lattice-height <H> --patches <N> --steps <S>
    python3 scripts/plot_lattice.py surface.lattice <W> <H> lattice.png

Unlike `coverage_report`/`plot_coverage.py` (aggregate coverage *over
time*, replayed from `trajectory.bin`), this only shows the state a run
ended in -- `surface.lattice` is overwritten in place as the simulation
runs, so there is no cheap way to recover an intermediate snapshot from it
alone; that would need a `trajectory.bin` replay tool of its own. What
this *does* show that the coverage plot cannot: real spatial structure
(clustering, domains, the actual hexagonal fcc(111) tiling `topology.rs`
implements) that an aggregate per-species fraction collapses away.

`surface.lattice` has no header -- it's exactly `width * height` raw
occupancy bytes, row-major (`layout::SiteLattice`'s own doc comment) --
so this reads it directly, with no Rust-side tool needed. The species
byte values below mirror `layout.rs`'s `VACANT`/`ADS_*` constants; the two
must stay in lockstep, the same convention `oc20e_format.py`'s `METALS`
list already follows for its own Rust counterpart.

Only used to produce the README's lattice-snapshot figure -- not part of
the simulation or ingestion pipeline itself, so (like `plot_coverage.py`)
this assumes `matplotlib` is installed (`pip install matplotlib`) rather
than sticking to the standard library.
"""

import math
import sys

VACANT = 0x00
ADS_O = 0x01
ADS_CO = 0x02
ADS_H = 0x04
ADS_OH = 0x08
ADS_H2O = 0x10

# Same palette `plot_coverage.py` uses, for visual consistency between the
# two figures.
SPECIES_COLORS = {
    VACANT: "#999999",
    ADS_O: "#d62728",
    ADS_H: "#1f77b4",
    ADS_CO: "#2ca02c",
    ADS_OH: "#9467bd",
    ADS_H2O: "#ff7f0e",
}
SPECIES_LABELS = {
    VACANT: "vacant",
    ADS_O: "O*",
    ADS_H: "H*",
    ADS_CO: "CO*",
    ADS_OH: "OH*",
    ADS_H2O: "H2O*",
}
# Any byte that isn't exactly one of the one-hot values above violates the
# engine's own "no site holds >1 species bit" invariant -- a corrupted or
# stale lattice file, not a state the simulator itself should ever
# produce. Rendered distinctly rather than silently miscolored or crashing
# the plot, so a corrupt file is visually obvious instead of a
# misleadingly clean-looking figure.
CORRUPT_COLOR = "#000000"


def hex_center(row, col):
    """Pixel-space center of the hexagon at offset-coordinate (row, col),
    using the standard "odd-r" pointy-top layout -- odd rows shifted +0.5
    hex-widths right, matching `topology.rs`'s `EVEN_ROW_DELTAS`/
    `ODD_ROW_DELTAS` (odd rows' NW/SW neighbors share the reference row's
    column; even rows' do not, which is what fixes which parity shifts
    right rather than left).
    """
    x = col + (0.5 if row % 2 == 1 else 0.0)
    y = row * (math.sqrt(3) / 2)
    return x, y


def main():
    if len(sys.argv) != 5:
        print(
            f"usage: {sys.argv[0]} <surface.lattice> <width> <height> <out.png>",
            file=sys.stderr,
        )
        return 1
    lattice_path, width, height, out_path = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon

    with open(lattice_path, "rb") as f:
        data = f.read()

    expected = width * height
    if len(data) != expected:
        print(
            f"error: {lattice_path} is {len(data)} bytes, expected "
            f"{width}x{height} = {expected} (wrong --lattice-width/-height?)",
            file=sys.stderr,
        )
        return 1

    fig_w = max(6.0, width / 6.0)
    fig_h = max(6.0, height / 6.0 * (math.sqrt(3) / 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    radius = 1.0 / math.sqrt(3)  # center-to-vertex radius giving unit
    # center-to-center horizontal spacing between same-row hexagons.
    corrupt_count = 0
    for idx, byte in enumerate(data):
        row, col = idx // width, idx % width
        x, y = hex_center(row, col)
        color = SPECIES_COLORS.get(byte)
        if color is None:
            color = CORRUPT_COLOR
            corrupt_count += 1
        ax.add_patch(
            RegularPolygon(
                (x, y), numVertices=6, radius=radius, orientation=0, facecolor=color,
                edgecolor="none",
            )
        )

    if corrupt_count:
        print(
            f"warning: {corrupt_count} site(s) held a byte matching no known "
            "one-hot species -- rendered black; this violates the engine's "
            "own occupancy invariant and likely means a corrupted/stale "
            "surface.lattice file",
            file=sys.stderr,
        )

    ax.set_xlim(-1, width + 1)
    ax.set_ylim(-1, height * (math.sqrt(3) / 2) + 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, matching the on-disk row order
    ax.axis("off")
    ax.set_title(
        f"Real Pd(111) surface occupancy, final state ({width}x{height} "
        "hexagonal fcc(111) lattice)"
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SPECIES_COLORS[b])
        for b in (VACANT, ADS_O, ADS_H, ADS_CO, ADS_OH, ADS_H2O)
    ]
    labels = [
        SPECIES_LABELS[b] for b in (VACANT, ADS_O, ADS_H, ADS_CO, ADS_OH, ADS_H2O)
    ]
    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
