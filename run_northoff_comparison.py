"""
Northoff vs. Solms vs. Borderline — Vergleichsexperiment.

Northoffs TTC-Depressionstheorie unterscheidet sich von Solms' SEEKING-Kollaps:

  Solms (SEEKING-Kollaps):
    → Agent verliert Motivation zu suchen (seeking_gain ↓)
    → Niedrige Urgency: spürt die Krise nicht stark
    → Findet Ressourcen (wenn er zufällig drüberläuft) aber WILL nicht konsumieren

  Northoff (Rest-Self-Overlap):
    → Agent verliert Kopplung zur Umwelt (A-Matrix flach, Affordanzen stumm)
    → Hohe Urgency: weiß dass er in Krise ist
    → WILL Ressourcen aber FINDET sie nicht (Wahrnehmung zu unscharf)

  Borderline (Regulationsstörung):
    → Valenz-Rauschen: Körpersignal fluktuiert unvorhersehbar
    → Temporaler Kollaps bei Stress: Zukunftsplanung bricht zusammen → Impulsivität
    → Weder stabil WILL noch stabil FINDET — WEISS NICHT OB ES WILL

Fünf Bedingungen:
  1. Gesund     (d=0.0, nd=0.0, bl=0.0)
  2. Solms      (d=1.0, nd=0.0, bl=0.0) — SEEKING-Kollaps
  3. Northoff   (d=0.0, nd=1.0, bl=0.0) — Rest-Self-Overlap + temporale Stasis
  4. Borderline (d=0.0, nd=0.0, bl=1.0) — Regulationsstörung
  5. Beide      (d=1.0, nd=1.0, bl=0.0) — Depression kombiniert

Ausführen:
  python3 run_northoff_comparison.py
  python3 run_northoff_comparison.py --seed 55 --steps 400
  python3 run_northoff_comparison.py --seeds 42 55 99 17
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from toni import GridWorld, Toni
from toni.environment import FOOD, WATER

warnings.filterwarnings("ignore")

# ── ANSI ──────────────────────────────────────────────────────────────────── #
R  = "\033[31m"
G  = "\033[32m"
Y  = "\033[33m"
B  = "\033[34m"
M  = "\033[35m"
C  = "\033[36m"
W  = "\033[37m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

CRISIS_THRESHOLD = 0.25


def bar(val: float, width: int = 8) -> str:
    filled = max(0, min(width, int(round(val * width))))
    color = G if val > 0.5 else (Y if val > 0.3 else R)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"


def run_condition(label: str, seed: int, steps: int,
                  d: float, nd: float, bl: float = 0.0) -> dict:
    env = GridWorld(seed=seed)
    np.random.seed(seed)
    agent = Toni(env, depression_level=d, northoff_depression_level=nd,
                 borderline_level=bl, use_temporal_self=True, policy_len=2)

    for _ in range(steps):
        agent.step()

    h = agent.history
    crises = sum(1 for e, hy in zip(h["energy"], h["hydration"])
                 if e < CRISIS_THRESHOLD or hy < CRISIS_THRESHOLD)
    consumes = sum(1 for x in h["action"] if x == "consume")
    wb = float(np.mean(h["wellbeing"]))
    avg_urgency = float(np.mean([max(fu[:2]) for fu in h["future_urgency"]]))

    # Verpasste Ressourcen: auf benötigtem Feld stehend ohne zu konsumieren
    missed_direct = 0
    env2 = GridWorld(seed=seed)
    for t in range(steps):
        if h["action"][t] != "consume":
            r, c = h["position"][t]
            dom = h["dominant_need"][t]
            res = FOOD if dom == 0 else WATER
            if env2.grid[r, c] == res:
                missed_direct += 1

    return {
        "label": label,
        "d": d, "nd": nd, "bl": bl,
        "crises": crises,
        "crisis_pct": crises / steps * 100,
        "consumes": consumes,
        "wellbeing": wb,
        "urgency": avg_urgency,
        "missed_direct": missed_direct,
        "history": h,
        "steps": steps,
    }


def dissociation_tag(res: dict, baseline_urgency: float,
                     baseline_consumes: int) -> str:
    """Interpretations-Tag für die Urgency/Konsum-Dissoziation."""
    nd, d, bl = res["nd"], res["d"], res["bl"]
    if d == 0.0 and nd == 0.0 and bl == 0.0:
        return f"{DIM}Baseline{RESET}"
    urg = res["urgency"]
    cons = res["consumes"]
    if bl > 0:
        # Borderline: hohe Urgency-Varianz, unvorhersehbare Konsume
        # Kein konsistentes Muster wie bei den Depressionen
        return f"{C}WEISS NICHT OB ES WILL{RESET} {DIM}(Borderline ✓){RESET}"
    if nd > 0 and d == 0:
        if urg > baseline_urgency * 1.1 and cons <= max(1, baseline_consumes - 1):
            return f"{R}WILL aber FINDET NICHT{RESET} {DIM}(Northoff ✓){RESET}"
        return f"{DIM}Northoff-Signal schwach{RESET}"
    if d > 0 and nd == 0:
        if urg < baseline_urgency * 0.95 and cons >= baseline_consumes:
            return f"{B}FINDET aber WILL NICHT{RESET} {DIM}(Solms ✓){RESET}"
        if urg >= baseline_urgency * 0.9 and cons <= max(1, baseline_consumes - 1):
            return f"{Y}Spät-Solms: weiß es, handelt nicht{RESET}"
        return f"{DIM}Solms-Signal gemischt{RESET}"
    return f"{DIM}beide Mechanismen aktiv{RESET}"


def print_results(results: list[dict], seed: int) -> None:
    steps = results[0]["steps"]
    baseline = next(r for r in results if r["d"] == 0 and r["nd"] == 0)

    print(f"\n{BOLD}{C}{'═'*76}{RESET}")
    print(f"{BOLD}  Seed={seed}  Schritte={steps}{RESET}")
    print(f"{BOLD}{C}{'═'*76}{RESET}")

    # Kopfzeile
    print(f"\n  {'Bedingung':<16}  {'Urgency':>8}  {'Konsume':>8}  "
          f"{'Krisen%':>8}  {'Ø-Wellb.':>9}  Interpretation")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*30}")

    for r in results:
        tag = dissociation_tag(r, baseline["urgency"], baseline["consumes"])
        urg_col = R if r["urgency"] > baseline["urgency"] * 1.05 else (
            G if r["urgency"] < baseline["urgency"] * 0.95 else "")
        con_col = G if r["consumes"] > baseline["consumes"] else (
            R if r["consumes"] < baseline["consumes"] else "")
        print(f"  {r['label']:<16}  "
              f"{urg_col}{r['urgency']:>8.3f}{RESET}  "
              f"{con_col}{r['consumes']:>8}{RESET}  "
              f"{r['crisis_pct']:>7.1f}%  "
              f"{r['wellbeing']:>9.3f}  {tag}")

    # Timeline Hydration
    print(f"\n  {DIM}{'─'*72}{RESET}")
    print(f"  {BOLD}Verlauf Hydration (alle 20 Schritte):{RESET}")
    col_map = {
        "Gesund":      G,
        "Solms":       Y,
        "Northoff":    R,
        "Borderline":  C,
        "Beide":       M,
    }
    labels = [r["label"] for r in results]
    header = f"  {'t':>5}" + "".join(f"  {col_map.get(lb,'')}{lb:<10}{RESET}" for lb in labels)
    print(header)
    for t in range(0, steps, 20):
        row = f"  {t:>5}"
        for r in results:
            h = r["history"]["hydration"]
            v = h[t] if t < len(h) else 0.0
            col = col_map.get(r["label"], "")
            crisis_mark = f"{R}!{RESET}" if v < CRISIS_THRESHOLD else " "
            row += f"  {col}{v:>5.3f}{RESET}{crisis_mark}    "
        print(row)

    # Urgency-Verlauf (Northoff vs. Solms vs. Borderline)
    nh = next((r for r in results if r["nd"] > 0 and r["d"] == 0), None)
    sh = next((r for r in results if r["d"] > 0 and r["nd"] == 0), None)
    bh = next((r for r in results if r["bl"] > 0), None)
    if nh and sh:
        print(f"\n  {DIM}{'─'*72}{RESET}")
        print(f"  {BOLD}Urgency-Verlauf "
              f"Northoff {R}(rot){RESET}{BOLD} / "
              f"Solms {Y}(gelb){RESET}{BOLD} / "
              f"Borderline {C}(cyan){RESET}{BOLD}:{RESET}")
        for t in range(0, steps, 20):
            nu = nh["history"]["future_urgency"]
            su = sh["history"]["future_urgency"]
            nv = float(max(nu[t][:2])) if t < len(nu) else 0.0
            sv = float(max(su[t][:2])) if t < len(su) else 0.0
            nb = bar(nv, 10)
            sb = bar(sv, 10)
            row = f"  t={t:>4}:  N {nb} {nv:.2f}   S {sb} {sv:.2f}"
            if bh:
                bu = bh["history"]["future_urgency"]
                bv = float(max(bu[t][:2])) if t < len(bu) else 0.0
                bb = bar(bv, 10)
                row += f"   {C}BL{RESET} {bb} {bv:.2f}"
            print(row)

    # Kern-Befund
    print(f"\n{BOLD}{C}{'═'*76}{RESET}")
    print(f"  {BOLD}Kernbefund:{RESET}")
    if nh and sh:
        urg_diff = nh["urgency"] - sh["urgency"]
        con_diff = sh["consumes"] - nh["consumes"]
        if urg_diff > 0.05 and con_diff > 0:
            print(f"  {R}Northoff{RESET}: urgency {nh['urgency']:.3f} > "
                  f"Solms {sh['urgency']:.3f} ({urg_diff:+.3f}) "
                  f"→ {R}leidet mehr{RESET}")
            print(f"  {Y}Solms{RESET}:    konsume {sh['consumes']} > "
                  f"Northoff {nh['consumes']} ({con_diff:+d}) "
                  f"→ {Y}findet trotzdem{RESET}")
            print(f"  {BOLD}→ Dissoziation bestätigt: "
                  f"{R}WILL aber FINDET NICHT{RESET} vs. "
                  f"{Y}FINDET aber WILL NICHT{RESET}{BOLD}{RESET}")
        else:
            print(f"  Dissoziation bei diesem Seed schwach — "
                  f"teste andere Seeds (--seed 55)")
    if bh:
        bl_res = bh
        baseline = next(r for r in results if r["d"] == 0 and r["nd"] == 0 and r["bl"] == 0)
        # Urgency-Varianz als BL-Marker berechnen
        bl_urgencies = [float(max(fu[:2])) for fu in bl_res["history"]["future_urgency"]]
        bl_std = float(np.std(bl_urgencies))
        base_urgencies = [float(max(fu[:2])) for fu in baseline["history"]["future_urgency"]]
        base_std = float(np.std(base_urgencies))
        print(f"\n  {C}Borderline{RESET}: urgency-σ {bl_std:.3f} vs. "
              f"Gesund {base_std:.3f} ({bl_std/max(base_std,1e-6):.1f}×) "
              f"→ {C}Valenz instabil{RESET}")
        print(f"  {C}Borderline{RESET}: konsume {bl_res['consumes']}  "
              f"krisen {bl_res['crisis_pct']:.1f}%")
        print(f"  {BOLD}→ Regulationsstörung: "
              f"{C}WEISS NICHT OB ES WILL{RESET} "
              f"{DIM}(kein stabiles SEEKING){RESET}{BOLD}{RESET}")
    print(f"{BOLD}{C}{'═'*76}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Northoff vs. Solms Depressions-Vergleich"
    )
    parser.add_argument("--seed",  type=int, default=55,
                        help="Einzelner Seed (default: 55)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Mehrere Seeds (überschreibt --seed)")
    parser.add_argument("--steps", type=int, default=300,
                        help="Simulationsschritte (default: 300)")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]

    CONDITIONS = [
        ("Gesund",      0.0, 0.0, 0.0),
        ("Solms",       1.0, 0.0, 0.0),
        ("Northoff",    0.0, 1.0, 0.0),
        ("Borderline",  0.0, 0.0, 1.0),
        ("Beide",       1.0, 1.0, 0.0),
    ]

    for seed in seeds:
        print(f"\n{DIM}Simuliere Seed={seed} ({args.steps} Schritte × 5 Bedingungen)...{RESET}")
        results = []
        for label, d, nd, bl in CONDITIONS:
            t0 = time.time()
            print(f"  {label:<12} ", end="", flush=True)
            res = run_condition(label, seed, args.steps, d, nd, bl)
            results.append(res)
            print(f"{DIM}({time.time()-t0:.1f}s){RESET}")
        print_results(results, seed)


if __name__ == "__main__":
    main()
