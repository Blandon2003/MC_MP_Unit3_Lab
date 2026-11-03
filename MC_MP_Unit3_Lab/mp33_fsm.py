"""
Mini Project 3.3 - Device Connection and Handover as a Finite-State Machine.

Implements Idle -> Random-Access (four RACH steps) -> Connected, with
Connected <-> Handover-in-progress and a Radio-Link-Failure path back to Idle.
A three-cell (A -> B -> C) scenario is evaluated under a naive and a
hysteresis-plus-TTT handover policy.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

RLF_THRESHOLD = -105.0
HYSTERESIS_DB = 3.0
TTT_SAMPLES = 5
PING_PONG_WINDOW = 30
N_SAMPLES = 400
CELL_CENTERS_KM = np.array([0.0, 2.0, 4.0])
CELL_NAMES = ["A", "B", "C"]

CELL_COLORS = ["#3b7dd8", "#e08a33", "#3fa06a"]
STATE_NAMES = ["Idle", "RandomAccess", "Connected", "HandoverInProgress", "RadioLinkFailure"]

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


# -----------------------------------------------------------------------------
# Scenario signal generation
# -----------------------------------------------------------------------------
def make_scenario(n: int = N_SAMPLES, seed: int = 21, fade: bool = False):
    """Device drives A -> B -> C with overlapping RSRP footprints."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x_km = 0.05 + (3.9 * t / (n - 1.0))  # 0.05 km .. 3.95 km

    dist = np.abs(x_km[:, None] - CELL_CENTERS_KM[None, :])
    dist = np.maximum(dist, 0.05)
    rsrp = -60.0 - 30.0 * np.log10(dist / 0.1)

    shadow = np.cumsum(rng.normal(0.0, 0.25, (n, 3)), axis=0)
    rsrp += shadow + rng.normal(0.0, 0.80, (n, 3))

    if fade:
        # Deep fade on cell B while the device is near it (stress test).
        fade_profile = np.zeros(n)
        start, length = 238, 18
        fade_profile[start : start + length] = -36.0 * np.hanning(length)
        rsrp[:, 1] += fade_profile

    return x_km, rsrp


# -----------------------------------------------------------------------------
# Finite-state machine
# -----------------------------------------------------------------------------
def run_fsm(rsrp, policy: str = "hysteresis_ttt", margin: float = HYSTERESIS_DB, ttt: int = TTT_SAMPLES):
    n = rsrp.shape[1]
    events: list[dict] = []
    state_hist = ["Idle"] * n
    serving_hist = np.full(n, -1, dtype=int)
    handover_pairs: list[tuple[int, int, int]] = []
    state = "Idle"
    serving = -1
    streak = 0
    pending_target = None
    pending_at = None
    last_left: dict[int, int] = {}
    ping_pongs = 0

    def log(t, from_state, to_state, trigger, cell=None, kind="transition"):
        events.append(
            {
                "time": int(t),
                "from": from_state,
                "to": to_state,
                "trigger": trigger,
                "cell": CELL_NAMES[cell] if cell is not None else "-",
                "kind": kind,
            }
        )

    t = 0
    while t < n:
        state_hist[t] = state

        if state == "Idle":
            best = int(np.argmax(rsrp[:, t]))
            if rsrp[best, t] < RLF_THRESHOLD:
                t += 1
                continue

            log(t, "Idle", "Random-Access", "Connection request -> RACH preamble (Msg1)", best)
            log(t + 1, "Random-Access", "Random-Access", "Random Access Response + timing advance (Msg2)", best, "rach")
            log(t + 2, "Random-Access", "Random-Access", "RRC connection request (Msg3)", best, "rach")
            log(t + 3, "Random-Access", "Connected", "Contention resolution success (Msg4)", best)

            for k in range(t, min(t + 3, n)):
                state_hist[k] = "Random-Access"
            if t + 3 < n:
                state_hist[t + 3] = "Connected"
                serving_hist[t + 3] = best
            serving = best
            state = "Connected"
            streak = 0
            t += 4
            continue

        if state == "Connected":
            serving_hist[t] = serving
            if rsrp[serving, t] < RLF_THRESHOLD:
                log(
                    t,
                    "Connected",
                    "Radio-Link-Failure",
                    f"Serving RSRP {rsrp[serving, t]:.1f} dBm < {RLF_THRESHOLD:.0f} dBm",
                    serving,
                )
                state_hist[t] = "Radio-Link-Failure"
                state = "Radio-Link-Failure"
                t += 1
                continue

            others = [c for c in range(3) if c != serving]
            cand = max(others, key=lambda c: rsrp[c, t])
            entering = rsrp[cand, t] > rsrp[serving, t] + margin
            streak = streak + 1 if entering else 0
            triggered = (policy == "naive" and rsrp[cand, t] > rsrp[serving, t]) or (
                policy != "naive" and entering and streak >= ttt
            )

            if triggered:
                trigger = (
                    f"Neighbour {CELL_NAMES[cand]} RSRP {rsrp[cand, t]:.1f} > serving "
                    f"{rsrp[serving, t]:.1f} dBm"
                    if policy == "naive"
                    else f"Neighbour {CELL_NAMES[cand]} exceeds serving by >{margin:.0f} dB "
                    f"for {ttt} samples"
                )
                if cand in last_left and t - last_left[cand] < PING_PONG_WINDOW:
                    ping_pongs += 1
                log(t, "Connected", "Handover-in-progress", trigger, cand)
                handover_pairs.append((t, serving, cand))
                last_left[serving] = t
                pending_target = cand
                pending_at = min(t + 1, n - 1)
                state_hist[t] = "Handover-in-progress"
                state = "Handover-in-progress"
                t += 1
                continue

            t += 1
            continue

        if state == "Handover-in-progress":
            serving_hist[t] = serving
            if rsrp[serving, t] < RLF_THRESHOLD:
                log(
                    t,
                    "Handover-in-progress",
                    "Radio-Link-Failure",
                    f"Serving RSRP {rsrp[serving, t]:.1f} dBm < {RLF_THRESHOLD:.0f} dBm during handover",
                    serving,
                )
                state_hist[t] = "Radio-Link-Failure"
                state = "Radio-Link-Failure"
                t += 1
                continue

            if t >= pending_at:
                log(
                    t,
                    "Handover-in-progress",
                    "Connected",
                    f"Target {CELL_NAMES[pending_target]} access confirmed (RRC reconfiguration complete)",
                    pending_target,
                )
                serving = pending_target
                state = "Connected"
                streak = 0
                state_hist[t] = "Connected"
                serving_hist[t] = serving
                t += 1
                continue

            t += 1
            continue

        if state == "Radio-Link-Failure":
            log(t, "Radio-Link-Failure", "Idle", "RLF recovery -> release connection and re-select cell", serving)
            state_hist[t] = "Idle"
            state = "Idle"
            t += 1
            continue

    handovers = [e for e in events if e["from"] == "Connected" and e["to"] == "Handover-in-progress"]
    rlfs = [e for e in events if e["to"] == "Radio-Link-Failure"]

    connected_samples = [
        (i, s)
        for i, s in enumerate(serving_hist)
        if s >= 0 and state_hist[i] in ("Connected", "Handover-in-progress")
    ]
    mean_serving = float(np.mean([rsrp[s, t] for t, s in connected_samples])) if connected_samples else float("nan")

    return {
        "events": events,
        "state_hist": state_hist,
        "serving_hist": serving_hist,
        "handovers": len(handovers),
        "handover_events": handovers,
        "rlf": len(rlfs),
        "rlf_events": rlfs,
        "ping_pongs": ping_pongs,
        "mean_serving_rsrp": round(mean_serving, 2),
        "min_serving_rsrp": round(float(min(rsrp[s, t] for t, s in connected_samples)), 2)
        if connected_samples
        else None,
    }


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_state_machine():
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, subtitle="", color="#dbe7f5", edge="#2c4f7c"):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=color,
                edgecolor=edge,
                linewidth=1.6,
            )
        )
        ax.text(x, y + 0.028, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        if subtitle:
            ax.text(x, y - 0.035, subtitle, ha="center", va="center", fontsize=7.0, color="#333333")

    def arrow(p1, p2, rad=0.0, color="#333333", ls="-", label="", lx=0.0, ly=0.0, fs=6.8, ha="center"):
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle="-|>",
                mutation_scale=12,
                connectionstyle=f"arc3,rad={rad}",
                color=color,
                linewidth=1.3,
                linestyle=ls,
                shrinkA=2,
                shrinkB=2,
            )
        )
        if label:
            ax.text(lx, ly, label, ha=ha, va="center", fontsize=fs, color=color, style="italic")

    box(0.10, 0.58, 0.14, 0.14, "Idle", "no connection")
    box(0.34, 0.58, 0.21, 0.20, "Random-Access", "", color="#fdebd0", edge="#a04000")
    box(0.60, 0.58, 0.17, 0.14, "Connected", "active link", color="#d5f5e3", edge="#1e8449")
    box(0.85, 0.82, 0.23, 0.14, "Handover-in-progress", "between two Connected states", color="#e8daef", edge="#6c3483")
    box(0.85, 0.30, 0.23, 0.14, "Radio-Link-Failure", "serving RSRP < -105 dBm", color="#fadbd8", edge="#a93226")

    arrow((0.17, 0.58), (0.245, 0.58), label="data arrival", lx=0.207, ly=0.695, fs=6.3)
    arrow((0.445, 0.58), (0.515, 0.58), label="Msg4 success", lx=0.48, ly=0.695, fs=6.3)
    arrow((0.665, 0.645), (0.775, 0.795), rad=-0.18, label="policy trigger\n(hysteresis + TTT)", lx=0.705, ly=0.775, fs=6.2, ha="right")
    arrow((0.775, 0.775), (0.675, 0.65), rad=-0.30, color="#1e8449", label="target\nconfirmed", lx=0.815, ly=0.69, fs=6.2)
    arrow((0.645, 0.51), (0.75, 0.365), rad=-0.12, color="#a93226", label="RLF", lx=0.665, ly=0.455, fs=6.3, ha="left")
    arrow((0.85, 0.75), (0.85, 0.37), color="#a93226", label="RSRP < -105 dBm\nmid-handover", lx=0.925, ly=0.56, fs=6.2)
    arrow((0.75, 0.25), (0.13, 0.51), rad=-0.18, color="#a93226", ls="--", label="RLF recovery -> re-select", lx=0.42, ly=0.19, fs=6.5)

    ax.text(
        0.34,
        0.91,
        "RACH steps:  (1) preamble  ->  (2) RAR + timing advance  ->  "
        "(3) connection request  ->  (4) contention resolution",
        ha="center",
        va="center",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7", edgecolor="#b7950b", linewidth=0.8),
    )
    ax.set_title("Connection-lifecycle finite-state machine, including the RLF path", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig33_state_machine.png", bbox_inches="tight")
    plt.close(fig)


def plot_scenario(x_km, rsrp, naive, tuned):
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.8), sharex=True)
    t = np.arange(len(x_km))

    for ax, res, title in [
        (axes[0], naive, "Naive policy: handover on any positive margin"),
        (axes[1], tuned, "Hysteresis 3 dB + TTT 5 samples"),
    ]:
        hist = res["serving_hist"]
        for c in range(3):
            ax.fill_between(
                t, -125, -40, where=hist == c, color=CELL_COLORS[c], alpha=0.10, step="post"
            )
        for c in range(3):
            ax.plot(t, rsrp[:, c], color=CELL_COLORS[c], lw=1.5, label=f"Cell {CELL_NAMES[c]}")
        for e in res["handover_events"]:
            ax.axvline(e["time"], color="#1e8449", lw=1.0, ls="--", alpha=0.9)
        for e in res["rlf_events"]:
            ax.axvline(e["time"], color="#c0392b", lw=1.4, ls=":", alpha=0.9)
        ax.axhline(RLF_THRESHOLD, color="grey", lw=1.0, ls=":", label="RLF threshold")
        ax.set_ylim(-125, -45)
        ax.set_xlim(0, len(t) - 1)
        ax.set_ylabel("RSRP (dBm)")
        ax.set_title(
            f"{title} - {res['handovers']} handovers, {res['rlf']} RLF, "
            f"{res['ping_pongs']} ping-pong"
        )
        ax.legend(loc="lower left", fontsize=7.2, ncol=4, framealpha=0.9)

    axes[-1].set_xlabel("Time (samples)")
    axes[-1].set_xlabel("Time (samples)   |   device position: Cell A -> Cell B -> Cell C")
    fig.suptitle("Three-cell mobility scenario under two handover policies", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig33_scenario.png", bbox_inches="tight")
    plt.close(fig)


def plot_fade_scenario(x_km, rsrp, res):
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    t = np.arange(len(x_km))
    hist = res["serving_hist"]
    for c in range(3):
        ax.fill_between(t, -125, -40, where=hist == c, color=CELL_COLORS[c], alpha=0.10, step="post")
    for c in range(3):
        ax.plot(t, rsrp[:, c], color=CELL_COLORS[c], lw=1.5, label=f"Cell {CELL_NAMES[c]}")
    for e in res["handover_events"]:
        ax.axvline(e["time"], color="#1e8449", lw=1.0, ls="--")
    for e in res["rlf_events"]:
        ax.axvline(e["time"], color="#c0392b", lw=1.5, ls=":")
        ax.annotate(
            "RLF",
            (e["time"], -101),
            xytext=(8, 0),
            textcoords="offset points",
            color="#c0392b",
            fontsize=8,
            fontweight="bold",
        )
    ax.axhline(RLF_THRESHOLD, color="grey", lw=1.0, ls=":", label="RLF threshold")
    ax.set_ylim(-125, -45)
    ax.set_xlim(220, 340)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title(
        "Tuned policy under a deep fade: delayed decision ends in radio-link failure "
        f"({res['rlf']} RLF, {res['handovers']} handovers)"
    )
    ax.legend(loc="lower right", fontsize=7.4, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "fig33_fade_rlf.png", bbox_inches="tight")
    plt.close(fig)


def write_log(path, events):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "from", "to", "trigger", "cell", "kind"])
        for e in events:
            writer.writerow([e["time"], e["from"], e["to"], e["trigger"], e["cell"], e["kind"]])


def main():
    x_km, rsrp = make_scenario()
    naive = run_fsm(rsrp.T, "naive")
    tuned = run_fsm(rsrp.T, "hysteresis_ttt")

    plot_state_machine()
    plot_scenario(x_km, rsrp, naive, tuned)
    write_log(RES / "mp33_transitions_naive.csv", naive["events"])
    write_log(RES / "mp33_transitions_tuned.csv", tuned["events"])

    # Stress case: deep fade on the serving cell to exercise the RLF path.
    _, rsrp_fade = make_scenario(fade=True)
    naive_fade = run_fsm(rsrp_fade.T, "naive")
    tuned_fade = run_fsm(rsrp_fade.T, "hysteresis_ttt")
    plot_fade_scenario(np.arange(len(rsrp_fade)), rsrp_fade, tuned_fade)

    def metrics(res):
        return {
            "handovers": res["handovers"],
            "rlf": res["rlf"],
            "ping_pongs": res["ping_pongs"],
            "mean_serving_rsrp": res["mean_serving_rsrp"],
            "min_serving_rsrp": res["min_serving_rsrp"],
        }

    payload = {
        "scenario": {
            "samples": N_SAMPLES,
            "cell_centres_km": CELL_CENTERS_KM.tolist(),
            "rlf_threshold": RLF_THRESHOLD,
        },
        "naive": metrics(naive),
        "tuned": metrics(tuned),
        "naive_handover_events": naive["handover_events"],
        "tuned_handover_events": tuned["handover_events"],
        "naive_first_events": naive["events"][:14],
        "tuned_first_events": tuned["events"][:14],
        "fade": {"naive": metrics(naive_fade), "tuned": metrics(tuned_fade)},
        "fade_rlf_events": {
            "naive": naive_fade["rlf_events"],
            "tuned": tuned_fade["rlf_events"],
        },
    }
    with open(RES / "mp33_results.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("Policy comparison:")
    for name, res in [("naive", naive), ("tuned", tuned), ("naive+fade", naive_fade), ("tuned+fade", tuned_fade)]:
        print(
            f"  {name:11s} handovers={res['handovers']:3d}  RLF={res['rlf']}  "
            f"ping-pong={res['ping_pongs']:2d}  mean_serving={res['mean_serving_rsrp']} dBm"
        )
    print("\nNaive handover times:", [e["time"] for e in naive["handover_events"]])
    print("Tuned handover times:", [e["time"] for e in tuned["handover_events"]])
    print("\nFirst tuned transitions:")
    for e in tuned["events"][:10]:
        print(f"  t={e['time']:3d} {e['from']:>22s} -> {e['to']:<22s} [{e['cell']}] {e['trigger']}")
    print("\nFade RLF times:", [e["time"] for e in naive_fade["rlf_events"]], [e["time"] for e in tuned_fade["rlf_events"]])


if __name__ == "__main__":
    main()
