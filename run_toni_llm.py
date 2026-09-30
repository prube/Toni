"""
run_toni_llm.py – Toni mit LLM-Kortex.

Demonstriert die vollständige Northoff+Solms+LLM-Architektur:

  Körper/Affekt → homöostatischer Fehler → Valenz → AIF-Policy → Aktion
       ↓
  LLM-Kortex (beobachtend, nicht kausal)
       ↓
  Sprachliche Selbstreflexion (für menschliche Beobachter)

Das LLM STEUERT NICHT. Es artikuliert nur den Zustand den der Körper bestimmt.
Architekturprinzip aus der RTF: "Das LLM dürfte nicht die Bedürfnisse bestimmen."

Voraussetzung:
  pip install anthropic

  Optional: ANTHROPIC_API_KEY setzen (oder `ant auth login` nutzen)

Verwendung:
  python3 run_toni_llm.py
  python3 run_toni_llm.py --steps 200 --reflect-every 15
  python3 run_toni_llm.py --model claude-sonnet-5
  python3 run_toni_llm.py --dry-run     # ohne API-Aufruf (Test der Pipeline)
"""
from __future__ import annotations

import argparse
import sys
import time
import textwrap

import numpy as np

from toni import GridWorld, Toni
from toni.body import NEED_NAMES, SETPOINT
from toni.llm_cortex import LLMCortex


# ── ANSI-Farben ──────────────────────────────────────────────────────────── #

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN  = "\033[36m"
RED   = "\033[31m"
MAGENTA = "\033[35m"
BLUE  = "\033[34m"
WHITE = "\033[37m"


def _bar(val: float, width: int = 12) -> str:
    filled = int(round(val * width))
    filled = max(0, min(width, filled))
    if val > 0.6:
        color = GREEN
    elif val > 0.3:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET} {val:.2f}"


def _valence_str(v: float) -> str:
    if v > 0.005:
        return f"{GREEN}▲ {v:+.4f}{RESET}"
    if v < -0.005:
        return f"{RED}▼ {v:+.4f}{RESET}"
    return f"{DIM}● {v:+.4f}{RESET}"


def print_header() -> None:
    print()
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  TONI – Northoff + Solms + LLM-Kortex{RESET}")
    print(f"{DIM}  Körper/Affekt treibt Verhalten. LLM artikuliert nur.{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
    print()


def print_state(agent: Toni, action: str, t: int) -> None:
    s = agent.body.state()
    err = SETPOINT - s
    need = NEED_NAMES[agent.dominant_need]

    print(f"{DIM}t={t:4d}{RESET}  "
          f"pos=({agent.row},{agent.col})  "
          f"aktion={action:<10}  "
          f"bedürfnis={YELLOW}{need}{RESET}")

    print(f"  E {_bar(s[0])}  H {_bar(s[1])}  "
          f"T {_bar(s[2])}  I {_bar(s[3])}")

    print(f"  wellbeing={agent.wellbeing:+.4f}  valenz={_valence_str(agent.valence)}")


def print_cortex_reflection(text: str, t: int, elapsed: float) -> None:
    """Gibt die LLM-Reflexion formatiert aus – klar als Beobachtungsschicht markiert."""
    width = 66
    print()
    print(f"  {MAGENTA}┌{'─' * (width - 2)}┐{RESET}")
    print(f"  {MAGENTA}│{RESET} {BOLD}LLM-KORTEX{RESET} {DIM}(t={t}, {elapsed:.1f}s){RESET}"
          + f"{' ' * (width - 22 - len(str(t)) - len(f'{elapsed:.1f}'))}  "
          + f"{MAGENTA}│{RESET}")
    print(f"  {MAGENTA}│{RESET} {DIM}Beobachtend · nicht kausal · kein Einfluss auf Body{RESET}"
          + f"{' ' * 8}{MAGENTA}│{RESET}")
    print(f"  {MAGENTA}├{'─' * (width - 2)}┤{RESET}")

    wrapped = textwrap.wrap(text, width=width - 4)
    for line in wrapped:
        padding = width - 4 - len(line)
        print(f"  {MAGENTA}│{RESET}  {WHITE}{line}{RESET}{' ' * padding}  {MAGENTA}│{RESET}")

    print(f"  {MAGENTA}└{'─' * (width - 2)}┘{RESET}")
    print()


def print_architecture_note() -> None:
    """Erklärt die Architektur beim Start."""
    print(f"{DIM}  Architektur-Schichten:{RESET}")
    print(f"  {GREEN}①{RESET} Körper   → Energie/Hydration/Temperatur/Integrität")
    print(f"  {GREEN}②{RESET} Affekt   → Homöostatischer Fehler → Valenz (Solms)")
    print(f"  {GREEN}③{RESET} Zeitdyn. → Northoff-Oszillatoren → Präzision/Arousal")
    print(f"  {GREEN}④{RESET} AIF      → pymdp Expected Free Energy → Aktion")
    print(f"  {MAGENTA}⑤{RESET} {MAGENTA}LLM      → Kortex artikuliert Zustand (KEIN Einfluss){RESET}")
    print()


def run_with_cortex(
    steps: int = 100,
    reflect_every: int = 10,
    model: str = "claude-sonnet-4-6",
    dry_run: bool = False,
    seed: int = 42,
) -> None:

    print_header()

    # ── Cortex initialisieren ───────────────────────────────────────────── #
    if dry_run:
        cortex = None
        print(f"  {YELLOW}[dry-run]{RESET} LLM-Kortex deaktiviert (--dry-run)")
    else:
        cortex = LLMCortex(model=model)
        if cortex.available:
            print(f"  {GREEN}✓{RESET} LLM-Kortex verfügbar  "
                  f"{DIM}(Modell: {model}){RESET}")
        else:
            print(f"  {YELLOW}⚠{RESET}  LLM-Kortex nicht verfügbar:")
            print(f"     {DIM}{cortex.last_error}{RESET}")
            print(f"     {DIM}Simulation läuft ohne Kortex-Reflexionen.{RESET}")

    print()
    print_architecture_note()

    # ── Agent und Umwelt ────────────────────────────────────────────────── #
    env = GridWorld(size=9, seed=seed)
    agent = Toni(env, start_pos=(4, 4), use_pymdp=True, use_northoff=True)

    print(f"  Simulation: {BOLD}{steps} Schritte{RESET}  "
          f"Reflexion alle {BOLD}{reflect_every}{RESET} Schritte")
    print(f"  {BOLD}{'─' * 68}{RESET}")
    print()

    reflection_count = 0
    total_api_time = 0.0

    for step in range(steps):
        result = agent.step()
        t = result["t"]
        action = result["action"]

        if not result["alive"]:
            print(f"\n{RED}  Toni ist nicht mehr am Leben (Schritt {t}).{RESET}\n")
            break

        # Statuszeile
        print_state(agent, action, t)

        # LLM-Reflexion alle N Schritte
        if (t % reflect_every == 0) and (cortex is not None) and cortex.available:
            t0 = time.time()
            reflection = cortex.reflect_from_agent(agent)
            elapsed = time.time() - t0
            total_api_time += elapsed
            reflection_count += 1

            if reflection:
                print_cortex_reflection(reflection, t, elapsed)

    # ── Zusammenfassung ─────────────────────────────────────────────────── #
    print(f"\n  {BOLD}{'═' * 68}{RESET}")
    print(f"  {BOLD}ERGEBNIS nach {agent.t} Schritten{RESET}")
    print(f"  {'─' * 68}")

    h = agent.history
    wbs = h["wellbeing"]
    if wbs:
        print(f"  Ø Wohlbefinden:    {np.mean(wbs):.4f}")
        print(f"  End-Wohlbefinden:  {wbs[-1]:.4f}")
        crises = sum(1 for e, hyd in zip(h["energy"], h["hydration"])
                     if e < 0.25 or hyd < 0.25)
        print(f"  Krisen (E/H<0.25): {crises}/{len(wbs)}  "
              f"({100*crises/len(wbs):.1f}%)")

    if reflection_count > 0:
        print(f"\n  LLM-Kortex:")
        print(f"  Reflexionen:       {reflection_count}")
        print(f"  Ø API-Zeit:        {total_api_time/reflection_count:.2f}s/Reflexion")
        print(f"  {DIM}(Reflexionen hatten keinen Einfluss auf Tonis Verhalten.){RESET}")

    print(f"\n  {BOLD}{'═' * 68}{RESET}\n")


# ── Einstiegspunkt ────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toni mit LLM-Kortex: Northoff+Solms+Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Beispiele:
              python3 run_toni_llm.py
              python3 run_toni_llm.py --steps 200 --reflect-every 15
              python3 run_toni_llm.py --model claude-sonnet-5
              python3 run_toni_llm.py --dry-run     # Test ohne API-Aufruf
        """),
    )
    parser.add_argument("--steps", type=int, default=100,
                        help="Simulationsschritte (default: 100)")
    parser.add_argument("--reflect-every", type=int, default=10,
                        help="Kortex-Reflexion alle N Schritte (default: 10)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Claude-Modell für den Kortex (default: claude-sonnet-4-6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Keine API-Aufrufe (testet nur die Simulation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Zufalls-Seed für die GridWorld (default: 42)")

    args = parser.parse_args()

    run_with_cortex(
        steps=args.steps,
        reflect_every=args.reflect_every,
        model=args.model,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
