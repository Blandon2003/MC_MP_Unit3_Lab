"""
Mini Project 3.2 - Cellular Capacity and Frequency-Reuse Planning.

Computes channels per cell, cell count, total system capacity and the co-channel
reuse ratio Q for candidate reuse factors, then quantifies cell splitting and
sectoring as alternative capacity levers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration (all inputs in one place)
# -----------------------------------------------------------------------------
TOTAL_SPECTRUM_MHZ = 50.0          # operator's total allocation (FDD uplink+downlink)
CHANNEL_BW_KHZ = 200.0             # one traffic channel
COVERAGE_AREA_KM2 = 400.0          # fixed service area
CELL_RADIUS_KM = 2.0               # baseline cell radius R
N_CANDIDATES = [1, 3, 4, 7, 9, 12]
PATH_LOSS_EXPONENT = 4.0           # used only for the SIR illustration
SECTOR_COUNT = 3

HEX_AREA_FACTOR = 3.0 * math.sqrt(3.0) / 2.0   # hexagon area = 2.598 R^2

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def hex_area(radius_km: float) -> float:
    return HEX_AREA_FACTOR * radius_km**2


def cells_in_area(radius_km: float, area_km2: float = COVERAGE_AREA_KM2) -> int:
    return int(math.floor(area_km2 / hex_area(radius_km)))


def capacity_row(n: int, radius_km: float = CELL_RADIUS_KM) -> dict:
    total_channels = int(TOTAL_SPECTRUM_MHZ * 1000.0 / CHANNEL_BW_KHZ)
    channels_per_cell = total_channels // n
    cells = cells_in_area(radius_km)
    q = math.sqrt(3.0 * n)
    # Co-channel SIR for an omni site with six first-tier interferers.
    sir_linear = q**PATH_LOSS_EXPONENT / 6.0
    sir_db = 10.0 * math.log10(sir_linear)
    return {
        "N": n,
        "Q": round(q, 3),
        "total_channels": total_channels,
        "channels_per_cell": channels_per_cell,
        "cells": cells,
        "capacity": channels_per_cell * cells,
        "sir_db": round(sir_db, 2),
    }


def sector_sir_db(n: int) -> float:
    """Approximate 120-degree sectoring: ~2 first-tier interferers instead of 6."""
    q = math.sqrt(3.0 * n)
    return round(10.0 * math.log10(q**PATH_LOSS_EXPONENT / 2.0), 2)


def main():
    rows = [capacity_row(n) for n in N_CANDIDATES]
    base = next(r for r in rows if r["N"] == 7)

    split_half = capacity_row(7, CELL_RADIUS_KM / 2.0)
    split_quarter = capacity_row(7, CELL_RADIUS_KM / 4.0)
    sector_n4 = capacity_row(4)
    sector_n7 = capacity_row(7)

    for r in (split_half, split_quarter):
        r["split_gain"] = round(r["capacity"] / base["capacity"], 2)
        r["cells_gain"] = round(r["cells"] / base["cells"], 2)

    sector_gain = sector_n4["capacity"] / sector_n7["capacity"]

    summary = {
        "config": {
            "total_spectrum_MHz": TOTAL_SPECTRUM_MHZ,
            "channel_bw_kHz": CHANNEL_BW_KHZ,
            "coverage_area_km2": COVERAGE_AREA_KM2,
            "cell_radius_km": CELL_RADIUS_KM,
            "sector_count": SECTOR_COUNT,
        },
        "rows": rows,
        "split_half": split_half,
        "split_quarter": split_quarter,
        "sectoring": {
            "omni_N7_capacity": sector_n7["capacity"],
            "sector_N4_capacity": sector_n4["capacity"],
            "gain": round(sector_gain, 2),
            "omni_N7_sir_db": sector_n7["sir_db"],
            "omni_N4_sir_db": sector_n4["sir_db"],
            "sector_N4_sir_db": sector_sir_db(4),
            "sector_N7_sir_db": sector_sir_db(7),
            "interferers_omni": 6,
            "interferers_sector": 2,
        },
    }

    # -- Table -----------------------------------------------------------------
    with open(RES / "mp32_capacity_table.csv", "w", encoding="utf-8") as fh:
        fh.write("N,Q,channels_total,channels_per_cell,cells,total_capacity,SIR_dB\n")
        for r in rows:
            fh.write(
                f"{r['N']},{r['Q']},{r['total_channels']},{r['channels_per_cell']},"
                f"{r['cells']},{r['capacity']},{r['sir_db']}\n"
            )
        fh.write(
            f"7 (R/2),{split_half['Q']},{split_half['total_channels']},"
            f"{split_half['channels_per_cell']},{split_half['cells']},"
            f"{split_half['capacity']},{split_half['sir_db']}\n"
        )
        fh.write(
            f"7 (R/4),{split_quarter['Q']},{split_quarter['total_channels']},"
            f"{split_quarter['channels_per_cell']},{split_quarter['cells']},"
            f"{split_quarter['capacity']},{split_quarter['sir_db']}\n"
        )

    with open(RES / "mp32_results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # -- Figure: capacity and Q vs N ------------------------------------------
    n_vals = [r["N"] for r in rows]
    caps = [r["capacity"] for r in rows]
    q_vals = [r["Q"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
    bars = ax1.bar([str(n) for n in n_vals], caps, color="#4c72b0", width=0.6, label="Total system capacity")
    ax1.set_xlabel("Frequency-reuse factor N")
    ax1.set_ylabel("Total system capacity (channels)", color="#2f4b7c")
    ax1.tick_params(axis="y", labelcolor="#2f4b7c")
    for bar, c in zip(bars, caps):
        ax1.text(bar.get_x() + bar.get_width() / 2, c + 90, f"{c:,}", ha="center", fontsize=8.5)

    ax2 = ax1.twinx()
    ax2.plot([str(n) for n in n_vals], q_vals, "o-", color="#c44e52", lw=2, label="Reuse ratio Q")
    ax2.set_ylabel("Co-channel reuse ratio Q = √(3N)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.grid(False)
    for x, q in zip(range(len(n_vals)), q_vals):
        ax2.annotate(f"{q:.2f}", (x, q), textcoords="offset points", xytext=(6, 6), fontsize=7.5, color="#c44e52")

    ax1.set_title("System capacity falls while Q rises as the reuse factor increases")
    fig.tight_layout()
    fig.savefig(FIG / "fig32_capacity_vs_N.png", bbox_inches="tight")
    plt.close(fig)

    # -- Figure: capacity vs Q -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(q_vals, caps, s=60, color="#4c72b0", zorder=3)
    for q, c, n in zip(q_vals, caps, n_vals):
        ax.annotate(f"N={n}", (q, c), textcoords="offset points", xytext=(7, 6), fontsize=8.5)
    ax.plot(q_vals, caps, color="#9bb7d4", lw=1.0, zorder=2)
    ax.set_xlabel("Co-channel reuse ratio Q")
    ax.set_ylabel("Total system capacity (channels)")
    ax.set_title("Capacity-interference trade-off across candidate reuse factors")
    fig.tight_layout()
    fig.savefig(FIG / "fig32_capacity_vs_Q.png", bbox_inches="tight")
    plt.close(fig)

    # -- Figure: capacity levers ----------------------------------------------
    labels = [
        "Baseline\nN=7, R=2 km",
        "Cell split\nR=1 km",
        "Cell split\nR=0.5 km",
        "3-sector\nN=4 equivalent",
    ]
    values = [
        base["capacity"],
        split_half["capacity"],
        split_quarter["capacity"],
        sector_n4["capacity"],
    ]
    colors = ["#4c72b0", "#55a868", "#2e7d4f", "#dd8452"]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 250,
            f"{v:,}\n({v / base['capacity']:.2f}x)",
            ha="center",
            fontsize=8.5,
        )
    ax.set_ylabel("Total system capacity (channels)")
    ax.set_title("Capacity levers compared with the N=7 baseline")
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig32_capacity_levers.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Total channels: {rows[0]['total_channels']}, cells per R=2 km: {rows[0]['cells']}")
    for r in rows:
        print(
            f"N={r['N']:2d}  Q={r['Q']:.3f}  ch/cell={r['channels_per_cell']:4d}  "
            f"cells={r['cells']:3d}  capacity={r['capacity']:5d}  SIR={r['sir_db']:.1f} dB"
        )
    print(
        f"Split R/2: {split_half['capacity']} ({split_half['split_gain']}x), "
        f"R/4: {split_quarter['capacity']} ({split_quarter['split_gain']}x)"
    )
    print(
        f"Sectoring: N=4 capacity {sector_n4['capacity']} vs N=7 {sector_n7['capacity']} "
        f"({sector_gain:.2f}x); sector SIR at N=4 = {sector_sir_db(4):.1f} dB"
    )


if __name__ == "__main__":
    main()
