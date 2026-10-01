"""
Depression-Experiment: Gesund vs. Depressiv.

Solms: Depression ist keine kognitive Störung — es ist ein Kollaps des
SEEKING-Systems. Das Affektantriebssystem, das Organismen motiviert aktiv
zu suchen, ist gedämpft.

Fünf implementierte Mechanismen:
  1. Anhedonie:           Konsum gibt weniger zurück (anhedonia_factor=0.2)
  2. SEEKING-Kollaps:     C-Vektor gedämpft (seeking_gain=0.3)
  3. Zeithorizont:        Zukunftsprojektion kollabiert auf 12 Schritte
  4. Desynchronisation:   Zirkadianer Rhythmus entkoppelt (k_ext_scale≈0)
  5. Rumination:          Negative Erinnerungen fünffach verstärkt

Erwartete Unterschiede:
  - Mehr Krisenschritte beim depressiven Agenten
  - Tiefere Wohlbefindens-Kurve ohne Erholung nach Konsum
  - Langsamere Reaktion auf wachsende Dringlichkeit
  - LLM-Reflexionen ohne Zukunftsbezug, ruminativer Ton
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from toni import GridWorld, Toni, LLMCortex

# ── ANSI ────────────────────────────────────────────────────────────────── #
R    = "\033[31m"
G    = "\033[32m"
Y    = "\033[33m"
B    = "\033[34m"
M    = "\033[35m"
C    = "\033[36m"
W    = "\033[37m"
DIM  = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def bar(val: float, width: int = 7) -> str:
    filled = max(0, min(width, int(round(val * width))))
    color = G if val > 0.5 else (Y if val > 0.3 else R)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"


def agent_line(agent: Toni, label: str, color: str) -> str:
    b = agent.body
    return (
        f"  {color}{BOLD}{label}{RESET} "
        f"E:{bar(b.energy)} {b.energy:.2f}  "
        f"H:{bar(b.hydration)} {b.hydration:.2f}  "
        f"wb={agent.wellbeing:+.4f}  "
        f"val={agent.valence:+.5f}  "
        f"pos=({agent.row},{agent.col})"
    )


def print_reflexion(label: str, color: str, text: str, t: int, elapsed: float) -> None:
    width = 68
    inner_w = width - 2
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        words = para.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > inner_w - 2:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)

    print(f"  {color}┌{'─'*width}┐{RESET}")
    print(f"  {color}│{RESET} {BOLD}LLM-KORTEX {label}{RESET}  "
          f"{DIM}t={t}  {elapsed:.1f}s{RESET}"
          + " " * max(0, inner_w - 22 - len(label))
          + f"{color}│{RESET}")
    print(f"  {color}├{'─'*width}┤{RESET}")
    for line in lines:
        pad = inner_w - len(line)
        print(f"  {color}│{RESET}  {W}{line}{RESET}{' ' * pad}{color}│{RESET}")
    print(f"  {color}└{'─'*width}┘{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toni Depression-Experiment: gesund vs. depressiv"
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reflect-every", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--depression", type=float, default=1.0,
                        help="Depressions-Stärke [0.0=gesund, 1.0=voll depressiv]")
    parser.add_argument("--policy-len", type=int, default=2,
                        help="AIF Planungshorizont in Schritten (default: 2, empfohlen: 4-5)")
    args = parser.parse_args()

    print(f"\n{BOLD}{M}{'═'*72}{RESET}")
    print(f"{BOLD}{M}  TONI — Depression-Experiment{RESET}")
    print(f"  {DIM}Solms: Depression = Kollaps des SEEKING-Systems (kein kognitives Problem){RESET}")
    print(f"{BOLD}{M}{'═'*72}{RESET}")
    print(f"\n  {G}Gesunder Agent{RESET}  vs.  {R}Depressiver Agent{RESET}  (depression_level={args.depression:.1f})")
    print(f"\n  Mechanismen beim depressiven Agenten:")

    d = args.depression
    print(f"    {DIM}① Anhedonie:        anhedonia_factor = {1.0-0.8*d:.2f}  (Konsum-Gain){RESET}")
    print(f"    {DIM}② SEEKING-Kollaps:  seeking_gain     = {1.0-0.7*d:.2f}  (C-Vektor){RESET}")
    print(f"    {DIM}③ Zeithorizont:     horizon          = {max(5,int(80*(1-0.85*d)))} Schritte  (normal: 80){RESET}")
    print(f"    {DIM}④ Desynchron.:      k_ext_scale      = {1.0-0.95*d:.2f}  (zirkadianer Rhythmus){RESET}")
    print(f"    {DIM}⑤ Rumination:       rumination       = {1.0+4.0*d:.1f}×  (neg. Erinnerungen){RESET}")
    print(f"\n  Schritte: {BOLD}{args.steps}{RESET}  Seed: {BOLD}{args.seed}{RESET}\n")
    print(f"  {DIM}{'─'*70}{RESET}\n")

    # ── Agenten ── #
    env_h = GridWorld(seed=args.seed)
    env_d = GridWorld(seed=args.seed)
    np.random.seed(args.seed)

    healthy   = Toni(env_h, start_pos=(4, 4), agent_id="Gesund",
                     depression_level=0.0, policy_len=args.policy_len)
    depressed = Toni(env_d, start_pos=(4, 4), agent_id="Depressiv",
                     depression_level=args.depression, policy_len=args.policy_len)

    # ── LLM-Kortex ── #
    cortex: Optional[LLMCortex] = None
    if not args.dry_run:
        cortex = LLMCortex(model=args.model)
        if cortex.available:
            print(f"  {G}✓{RESET} LLM-Kortex verfügbar\n")
        else:
            print(f"  {Y}⚠{RESET} LLM nicht verfügbar: {cortex.last_error}\n")
            cortex = None

    # ── Metriken-Tracking ── #
    crises_h = crises_d = 0
    first_crisis_h = first_crisis_d = None
    consume_lags_h: list[int] = []  # Schritte zwischen urgency>0.5 und Konsum
    consume_lags_d: list[int] = []
    urgency_start_h = urgency_start_d = None

    api_calls = api_time = 0.0

    print(f"  {DIM}{'─'*70}{RESET}")

    for step in range(args.steps):
        healthy.step()
        depressed.step()
        t = step + 1

        # Krisen zählen
        if healthy.body.energy < 0.25 or healthy.body.hydration < 0.25:
            crises_h += 1
            if first_crisis_h is None:
                first_crisis_h = t
        if depressed.body.energy < 0.25 or depressed.body.hydration < 0.25:
            crises_d += 1
            if first_crisis_d is None:
                first_crisis_d = t

        # Reaktionslatenz: Schritte von urgency>0.5 bis Konsum
        fu_h = healthy.temporal_self.future_urgency()
        fu_d = depressed.temporal_self.future_urgency()

        if max(fu_h[:2]) > 0.5 and urgency_start_h is None:
            urgency_start_h = t
        if healthy.history["action"] and healthy.history["action"][-1] == "consume":
            if urgency_start_h is not None:
                consume_lags_h.append(t - urgency_start_h)
                urgency_start_h = None

        if max(fu_d[:2]) > 0.5 and urgency_start_d is None:
            urgency_start_d = t
        if depressed.history["action"] and depressed.history["action"][-1] == "consume":
            if urgency_start_d is not None:
                consume_lags_d.append(t - urgency_start_d)
                urgency_start_d = None

        # Anzeige
        if t % 10 == 0 or t == args.steps:
            urgency_h = max(fu_h[:2])
            urgency_d = max(fu_d[:2])
            print(f"\n{DIM}t={t:4d}{RESET}")
            print(agent_line(healthy, "Gesund  ", G)
                  + f"  {DIM}urg={urgency_h:.2f}{RESET}")
            print(agent_line(depressed, "Depressiv", R)
                  + f"  {DIM}urg={urgency_d:.2f}{RESET}")

        # ── LLM-Reflexionen ── #
        if cortex and t % args.reflect_every == 0:
            for agent, color, label in [
                (healthy,   G, "Gesund  "),
                (depressed, R, "Depressiv"),
            ]:
                print()
                t0 = time.time()
                text = cortex.reflect_from_agent(agent)
                elapsed = time.time() - t0
                api_calls += 1
                api_time += elapsed
                if text:
                    print_reflexion(label, color, text, t, elapsed)

    # ── Ergebnis ── #
    print(f"\n{BOLD}{M}{'═'*72}{RESET}")
    print(f"{BOLD}ERGEBNIS nach {args.steps} Schritten{RESET}")
    print(f"{DIM}{'─'*70}{RESET}\n")

    def metrics(agent, color, label, crises, first_crisis, lags):
        h = agent.history
        avg_wb  = float(np.mean(h["wellbeing"])) if h["wellbeing"] else 0.0
        end_wb  = h["wellbeing"][-1] if h["wellbeing"] else 0.0
        avg_vib = float(np.mean(h["viability"])) if h["viability"] else 0.0
        avg_mob = float(np.mean(h["mobility"])) if h["mobility"] else 0.0
        consumes = sum(1 for a in h["action"] if a == "consume")
        avg_lag = float(np.mean(lags)) if lags else float("nan")
        print(f"  {color}{BOLD}{label}{RESET}")
        print(f"  Ø Wohlbefinden:       {avg_wb:+.4f}")
        print(f"  End-Wohlbefinden:     {end_wb:+.4f}")
        print(f"  Krisenschritte:       {crises}/{args.steps}  ({100*crises/args.steps:.1f}%)")
        print(f"  Erste Krise:          {first_crisis if first_crisis else '—'}")
        print(f"  Konsum-Aktionen:      {consumes}")
        print(f"  Reaktionslatenz Ø:   {avg_lag:.1f} Schritte  "
              f"{DIM}(urgency>0.5 → Konsum){RESET}")
        print(f"  Ø Viabilität:         {avg_vib:.3f}")
        print(f"  Ø Mobilität:          {avg_mob:.3f}")
        print()

    metrics(healthy,   G, "Gesund  ", crises_h, first_crisis_h, consume_lags_h)
    metrics(depressed, R, "Depressiv", crises_d, first_crisis_d, consume_lags_d)

    # Delta-Zusammenfassung
    h_wb = float(np.mean(healthy.history["wellbeing"]))
    d_wb = float(np.mean(depressed.history["wellbeing"]))
    print(f"  {BOLD}Δ Wohlbefinden:{RESET}  {d_wb - h_wb:+.4f}  "
          f"({R}depressiv{RESET} vs. {G}gesund{RESET})")
    print(f"  {BOLD}Δ Krisen:{RESET}        {crises_d - crises_h:+d} Schritte")

    h_lag = float(np.mean(consume_lags_h)) if consume_lags_h else float("nan")
    d_lag = float(np.mean(consume_lags_d)) if consume_lags_d else float("nan")
    if not np.isnan(h_lag) and not np.isnan(d_lag):
        print(f"  {BOLD}Δ Reaktionslatenz:{RESET} {d_lag - h_lag:+.1f} Schritte  "
              f"{DIM}(positiv = depressiver Agent reagiert später){RESET}")

    if api_calls > 0:
        print(f"\n  LLM-Reflexionen: {int(api_calls)}  "
              f"Ø {api_time/api_calls:.1f}s/Reflexion")
        print(f"  {DIM}(Reflexionen hatten keinen Einfluss auf Tonis Verhalten.){RESET}")

    print(f"\n{BOLD}{M}{'═'*72}{RESET}\n")


if __name__ == "__main__":
    main()
