"""
Vergleichsexperiment: Antizipatorische vs. Reaktive Motivation.

Northoffs Kernthese: Ein System mit temporalem Selbst handelt
BEVOR die Krise eintritt — weil es die Zukunft projiziert.

Zwei Toni-Instanzen, identischer Seed:
  Reaktiv:        future_urgency=None → AIF handelt nur auf Istzustand
  Antizipatorisch: future_urgency aktiv → AIF handelt auf Projektion

Messung:
  - Anzahl Krisen (Zustand < 0.25)
  - Durchschnittliches Wohlbefinden
  - Durchschnittliche Schritte zwischen Krisenankündigung und tatsächlicher Krise
  - Endwerte der Körperzustände
"""
from __future__ import annotations

import argparse
import sys
import time
import textwrap

import numpy as np

from toni import GridWorld, Toni

# ── ANSI ────────────────────────────────────────────────────────────────── #
R  = "\033[31m"   # rot
G  = "\033[32m"   # grün
Y  = "\033[33m"   # gelb
B  = "\033[34m"   # blau
M  = "\033[35m"   # magenta
C  = "\033[36m"   # cyan
W  = "\033[37m"   # weiß
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

CRISIS_THRESHOLD = 0.25


def bar(val: float, width: int = 10) -> str:
    filled = int(round(val * width))
    filled = max(0, min(width, filled))
    color = G if val > 0.5 else (Y if val > 0.3 else R)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"


def count_crises(history: dict) -> int:
    """Anzahl Zeitschritte in denen Energie oder Hydration < Schwellwert."""
    crises = 0
    for e, h in zip(history["energy"], history["hydration"]):
        if e < CRISIS_THRESHOLD or h < CRISIS_THRESHOLD:
            crises += 1
    return crises


def avg_wellbeing(history: dict) -> float:
    return float(np.mean(history["wellbeing"]))


def time_to_first_crisis(history: dict) -> int | None:
    """Erster Schritt mit E oder H < Schwellwert."""
    for t, (e, h) in enumerate(zip(history["energy"], history["hydration"])):
        if e < CRISIS_THRESHOLD or h < CRISIS_THRESHOLD:
            return t
    return None


def anticipation_lead(agent_anti: Toni, steps: int) -> list[int]:
    """
    Für jeden Schritt wo future_urgency > 0.5:
    Wie viele Schritte früher erkannte Toni-A die Krise als sie eintrat?
    Gibt Liste von Vorläufen (in Schritten) zurück.
    """
    leads = []
    h = agent_anti.history
    urgency_series = h["future_urgency"]   # list of np.ndarray[:2]
    energy_series  = h["energy"]
    hydro_series   = h["hydration"]

    crisis_start: int | None = None
    warning_start: int | None = None

    for t in range(steps):
        e, hydro = energy_series[t], hydro_series[t]
        in_crisis = (e < CRISIS_THRESHOLD or hydro < CRISIS_THRESHOLD)

        if t < len(urgency_series):
            urg = urgency_series[t]
            urg_max = float(max(urg))
        else:
            urg_max = 0.0

        # Warnung beginnt
        if urg_max > 0.5 and warning_start is None and not in_crisis:
            warning_start = t

        # Krise beginnt
        if in_crisis and crisis_start is None:
            crisis_start = t
            if warning_start is not None:
                leads.append(crisis_start - warning_start)
            warning_start = None
            crisis_start = None

    return leads


def run_agent(label: str, env_seed: int, steps: int, use_ts: bool,
              verbose: bool = False, policy_len: int = 2) -> Toni:
    env = GridWorld(seed=env_seed)
    np.random.seed(env_seed)
    agent = Toni(env, use_temporal_self=use_ts, policy_len=policy_len)

    if verbose:
        tag = f"{C}[{label}]{RESET}"
        print(f"  {tag} Starte Simulation ({steps} Schritte)...")

    for _ in range(steps):
        agent.step()

    return agent


def print_comparison(r: Toni, a: Toni, steps: int) -> None:
    """Tabellarischer Vergleich."""
    rh, ah = r.history, a.history

    # ── Kopf ── #
    print(f"\n{BOLD}{C}{'═'*70}{RESET}")
    print(f"{BOLD}  VERGLEICH: Reaktiv vs. Antizipatorisch (Northoff Temporal Self){RESET}")
    print(f"{BOLD}{C}{'═'*70}{RESET}\n")

    print(f"  {'Metrik':<35} {'Reaktiv':>14}  {'Antizipatorisch':>14}")
    print(f"  {'-'*35} {'-'*14}  {'-'*14}")

    def row(label: str, rv, av, better: str = "higher"):
        hi = G if better == "higher" else R
        lo = R if better == "higher" else G
        rc = hi if rv >= av else lo
        ac = hi if av >= rv else lo
        delta = av - rv
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<35} {rc}{rv:>14.4f}{RESET}  {ac}{av:>14.4f}{RESET}"
              f"  {DIM}(Δ={sign}{delta:.4f}){RESET}")

    def row_int(label: str, rv, av, better: str = "lower"):
        hi = G if better == "lower" else R
        lo = R if better == "lower" else G
        rc = hi if (rv <= av if better == "lower" else rv >= av) else lo
        ac = hi if (av <= rv if better == "lower" else av >= rv) else lo
        if rv is None: rvs = "    keine Krise"
        else: rvs = f"{rv:>14d}"
        if av is None: avs = "    keine Krise"
        else: avs = f"{av:>14d}"
        print(f"  {label:<35} {rc}{rvs}{RESET}  {ac}{avs}{RESET}")

    avg_r = avg_wellbeing(rh)
    avg_a = avg_wellbeing(ah)
    row("Ø Wohlbefinden", avg_r, avg_a, "higher")

    end_r = rh["wellbeing"][-1] if rh["wellbeing"] else 0.0
    end_a = ah["wellbeing"][-1] if ah["wellbeing"] else 0.0
    row("End-Wohlbefinden", end_r, end_a, "higher")

    cr_r = count_crises(rh)
    cr_a = count_crises(ah)
    rv = cr_r; av = cr_a
    rc = G if rv <= av else R; ac = G if av <= rv else R
    delta = av - rv; sign = "+" if delta >= 0 else ""
    print(f"  {'Krisenschritte (E/H < 0.25)':<35} {rc}{rv:>14d}{RESET}  "
          f"{ac}{av:>14d}{RESET}  {DIM}(Δ={sign}{delta}){RESET}")

    ttf_r = time_to_first_crisis(rh)
    ttf_a = time_to_first_crisis(ah)
    row_int("Erste Krise (Schritt)", ttf_r, ttf_a, "lower")

    # Antizipations-Vorlauf
    leads = anticipation_lead(a, steps)
    if leads:
        avg_lead = float(np.mean(leads))
        print(f"  {'Ø Antizipations-Vorlauf (Schritte)':<35} {'—':>14}  "
              f"{G}{avg_lead:>14.1f}{RESET}  "
              f"{DIM}(vor Krise erkannt){RESET}")
    else:
        print(f"  {'Antizipations-Vorlauf':<35} {'—':>14}  "
              f"{DIM}{'keine Daten':>14}{RESET}")

    # End-Körperzustände
    print(f"\n  {DIM}{'─'*66}{RESET}")
    print(f"  {BOLD}End-Körperzustände:{RESET}")

    names = ["Energie", "Hydration", "Temperatur", "Integrität"]
    keys  = ["energy", "hydration", "temperature", "integrity"]
    for name, key in zip(names, keys):
        rv = rh[key][-1] if rh[key] else 0.0
        av = ah[key][-1] if ah[key] else 0.0
        rb = bar(rv, 8)
        ab = bar(av, 8)
        delta = av - rv
        sign = "+" if delta >= 0 else ""
        color = G if delta >= 0 else R
        print(f"  {name:<14} R: {rb} {rv:.3f}   A: {ab} {av:.3f}"
              f"  {DIM}{color}(Δ={sign}{delta:.3f}){RESET}")

    # Timeline-Vergleich
    print(f"\n  {DIM}{'─'*66}{RESET}")
    print(f"  {BOLD}Verlauf Hydration (alle 10 Schritte):{RESET}")
    print(f"  {'t':>6}  {'Reaktiv':>9}  {'Antizipat.':>10}  {'Δ':>7}")
    for t in range(0, min(steps, len(rh["hydration"])), 10):
        rv_h = rh["hydration"][t]
        av_h = ah["hydration"][t]
        delta = av_h - rv_h
        sign = "+" if delta >= 0 else ""
        color = G if delta >= 0 else R
        rv_u = ""
        av_u = ""
        if t < len(ah["future_urgency"]):
            au = float(max(ah["future_urgency"][t]))
            av_u = f" [{Y}urg={au:.2f}{RESET}]" if au > 0.3 else ""
        crisis_r = f" {R}!{RESET}" if rv_h < CRISIS_THRESHOLD else ""
        crisis_a = f" {R}!{RESET}" if av_h < CRISIS_THRESHOLD else ""
        print(f"  {t:>6}:  {rv_h:>6.3f}{crisis_r}    {av_h:>6.3f}{av_u}{crisis_a}"
              f"    {color}{sign}{delta:.3f}{RESET}")

    # Fazit
    print(f"\n{BOLD}{C}{'═'*70}{RESET}")
    wb_diff = avg_a - avg_r
    crisis_diff = cr_r - cr_a
    print(f"  {BOLD}Fazit:{RESET}")
    if wb_diff > 0 or crisis_diff > 0:
        parts = []
        if wb_diff > 0.001:
            parts.append(f"Ø Wohlbefinden {G}+{wb_diff:.4f}{RESET} höher")
        if crisis_diff > 0:
            parts.append(f"{G}{crisis_diff} weniger{RESET} Krisenschritte")
        if leads:
            avg_lead = float(np.mean(leads))
            parts.append(f"erkannte Krisen {G}{avg_lead:.0f} Schritte früher{RESET}")
        print(f"  Antizipatorisch: " + " | ".join(parts))
        print(f"  → {G}Northoff-These bestätigt:{RESET} "
              f"temporales Selbst verbessert Homöostase.")
    elif wb_diff < -0.001:
        print(f"  Kein Vorteil durch Antizipation in diesem Lauf.")
        print(f"  (Mögliche Ursache: zu wenige Schritte, gleiche Navigation)")
    else:
        print(f"  Kein messbarer Unterschied. Mehr Schritte oder anderen Seed testen.")
    print(f"{BOLD}{C}{'═'*70}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reaktiv vs. Antizipatorisch — Northoff Temporal Self Experiment"
    )
    parser.add_argument("--steps", type=int, default=200,
                        help="Simulationsschritte (default: 200)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Zufalls-Seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Fortschritt anzeigen")
    parser.add_argument("--policy-len", type=int, default=2,
                        help="AIF Planungshorizont in Schritten (default: 2, empfohlen: 4-5)")
    args = parser.parse_args()

    print(f"\n{BOLD}{C}{'═'*70}{RESET}")
    print(f"{BOLD}  NORTHOFF TEMPORAL SELF — Vergleichsexperiment{RESET}")
    print(f"  Seed={args.seed}  Schritte={args.steps}  policy_len={args.policy_len}")
    print(f"{BOLD}{C}{'═'*70}{RESET}")
    print(f"\n  {DIM}Zwei identische Agenten (gleicher Seed, gleiche Welt){RESET}")
    print(f"  {R}Reaktiv:{RESET}         future_urgency=None  → nur Istzustand")
    print(f"  {G}Antizipatorisch:{RESET} future_urgency aktiv → projizierte Zukunft\n")

    t0 = time.time()
    print(f"  Simuliere {R}Reaktiv{RESET}...", end="", flush=True)
    agent_reaktiv = run_agent("R", args.seed, args.steps, use_ts=False,
                              verbose=False, policy_len=args.policy_len)
    print(f" {DIM}({time.time()-t0:.1f}s){RESET}")

    t1 = time.time()
    print(f"  Simuliere {G}Antizipatorisch{RESET}...", end="", flush=True)
    agent_anti = run_agent("A", args.seed, args.steps, use_ts=True,
                           verbose=False, policy_len=args.policy_len)
    print(f" {DIM}({time.time()-t1:.1f}s){RESET}")

    print_comparison(agent_reaktiv, agent_anti, args.steps)


if __name__ == "__main__":
    main()
