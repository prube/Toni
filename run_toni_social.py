"""
Zwei-Agenten-Simulation: Soziale Positionierung nach Northoff.

Northoffs dritte Achse: Das Selbst positioniert sich nicht nur temporal,
sondern relational — in Bezug auf andere. Das Selbst existiert als
soziales Selbst.

Zwei Toni-Instanzen in derselben Welt:
  - Jeder beobachtet den anderen (Position, Aktion, Bedürfnis)
  - Soziale Signale fließen in AIF ein (D-Prior, C-Konkurrenz)
  - LLM-Kortex bekommt beide Zeit- und Sozialkontext

Ausgabe:
  - Nebeneinander-Display beider Agenten
  - LLM-Reflexionen zeigen soziale Dimension
  - Am Ende: Vergleich Wohlbefinden Allein vs. Zusammen
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np

from toni import GridWorld, Toni, LLMCortex

# ── ANSI ────────────────────────────────────────────────────────────────── #
R   = "\033[31m"
G   = "\033[32m"
Y   = "\033[33m"
B   = "\033[34m"
M   = "\033[35m"
C   = "\033[36m"
W   = "\033[37m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

NEED_COLORS = {"energy": Y, "hydration": C, "temperature": B, "integrity": R}


def bar(val: float, width: int = 8) -> str:
    filled = int(round(val * width))
    filled = max(0, min(width, filled))
    color = G if val > 0.5 else (Y if val > 0.3 else R)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"


def agent_line(agent: Toni, label: str, color: str) -> str:
    b = agent.body
    need = agent.history["action"][-1] if agent.history["action"] else "?"
    dn = ["E", "H", "T", "I"][agent.dominant_need]
    return (
        f"  {color}{BOLD}{label}{RESET} "
        f"pos=({agent.row},{agent.col})  "
        f"E:{bar(b.energy,6)} {b.energy:.2f}  "
        f"H:{bar(b.hydration,6)} {b.hydration:.2f}  "
        f"T:{bar(b.temperature,6)} {b.temperature:.2f}  "
        f"wb={agent.wellbeing:+.3f}  "
        f"need={color}{dn}{RESET}  "
        f"→{DIM}{need}{RESET}"
    )


def print_reflexion(label: str, color: str, text: str, t: int, elapsed: float) -> None:
    width = 68
    label_str = f"{BOLD}LLM-KORTEX {label}{RESET}"
    time_str = f"{DIM}(t={t}, {elapsed:.1f}s){RESET}"
    inner_w = width - 2
    lines = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        words = paragraph.split()
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
    print(f"  {color}│{RESET} {label_str} {time_str:<{inner_w-len(label)-20}}{color}│{RESET}")
    print(f"  {color}│{RESET} {DIM}Beobachtend · kein Einfluss auf Body{RESET}"
          + " " * (inner_w - 36) + f"{color}│{RESET}")
    print(f"  {color}├{'─'*width}┤{RESET}")
    for line in lines:
        pad = inner_w - len(line)
        print(f"  {color}│{RESET}  {W}{line}{RESET}{' ' * pad}{color}│{RESET}")
    print(f"  {color}└{'─'*width}┘{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toni×2 — Soziale Positionierung (Northoff)"
    )
    parser.add_argument("--steps", type=int, default=150,
                        help="Simulationsschritte (default: 150)")
    parser.add_argument("--reflect-every", type=int, default=25,
                        help="LLM-Reflexionsintervall (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Zufalls-Seed (default: 42)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--dry-run", action="store_true",
                        help="Kein LLM-Aufruf")
    parser.add_argument("--policy-len", type=int, default=2,
                        help="AIF Planungshorizont in Schritten (default: 2, empfohlen: 4-5)")
    args = parser.parse_args()

    print(f"\n{BOLD}{C}{'═'*72}{RESET}")
    print(f"{BOLD}{C}  TONI×2 — Soziale Positionierung (Northoff){RESET}")
    print(f"  {DIM}Northoff: das Selbst positioniert sich temporal, räumlich UND sozial.{RESET}")
    print(f"{BOLD}{C}{'═'*72}{RESET}")
    print(f"\n  {G}①{RESET} Körper/Affekt        → Homöostase (Solms)")
    print(f"  {G}②{RESET} Zeitdynamik           → Northoff-Oszillatoren")
    print(f"  {G}③{RESET} Temporales Selbst     → Vergangenheit/Gegenwart/Zukunft")
    print(f"  {G}④{RESET} Soziales Selbst       → Relationale Positionierung")
    print(f"  {M}⑤{RESET} {M}LLM-Kortex           → artikuliert alle vier Schichten{RESET}\n")
    print(f"  Schritte: {BOLD}{args.steps}{RESET}  "
          f"Reflexion alle {BOLD}{args.reflect_every}{RESET}  "
          f"Seed: {BOLD}{args.seed}{RESET}")
    print(f"  {DIM}{'─'*70}{RESET}\n")

    # ── Welt und Agenten ── #
    env = GridWorld(seed=args.seed)
    np.random.seed(args.seed)

    # Zwei Agenten mit verschiedenen Startpositionen
    size = env.size
    toni_a = Toni(env, start_pos=(1, 1), agent_id="Toni-A",
                  policy_len=args.policy_len)
    toni_b = Toni(env, start_pos=(size - 2, size - 2), agent_id="Toni-B",
                  policy_len=args.policy_len)

    # LLM-Kortex (optional)
    cortex: Optional[LLMCortex] = None
    if not args.dry_run:
        cortex = LLMCortex(model=args.model)
        if cortex.available:
            print(f"  {G}✓{RESET} LLM-Kortex verfügbar  {DIM}(Modell: {args.model}){RESET}\n")
        else:
            print(f"  {Y}⚠{RESET} LLM nicht verfügbar: {cortex.last_error}\n")
            cortex = None

    api_calls = 0
    api_time_total = 0.0
    COLORS = {toni_a: C, toni_b: M}
    LABELS = {toni_a: "Toni-A", toni_b: "Toni-B"}

    print(f"  {DIM}{'─'*70}{RESET}")

    for step in range(args.steps):
        # ── Jeder Agent beobachtet den anderen ── #
        obs_a = toni_a.social_observation()
        obs_b = toni_b.social_observation()

        # A beobachtet B, B beobachtet A
        toni_a.receive_social_observations([obs_b])
        toni_b.receive_social_observations([obs_a])

        # ── Beide treten einen Schritt ── #
        toni_a.step()
        toni_b.step()

        t_disp = step + 1

        # ── Anzeige ── #
        if step % 5 == 0 or t_disp == args.steps:
            print(f"\n{DIM}t={t_disp:4d}{RESET}")
            print(agent_line(toni_a, "A", C))
            print(agent_line(toni_b, "B", M))

            # Sozialer Abstand
            dist = abs(toni_a.row - toni_b.row) + abs(toni_a.col - toni_b.col)
            dist_str = f"{G}nah ({dist}){RESET}" if dist <= 3 else f"{DIM}weit ({dist}){RESET}"
            print(f"  {DIM}Abstand A↔B:{RESET} {dist_str}")

        # ── LLM-Reflexionen ── #
        if (cortex is not None and
                t_disp % args.reflect_every == 0 and t_disp > 0):

            for agent, color, label in [
                (toni_a, C, "Toni-A"),
                (toni_b, M, "Toni-B"),
            ]:
                print()
                t0 = time.time()
                text = cortex.reflect_from_agent(agent)
                elapsed = time.time() - t0
                api_calls += 1
                api_time_total += elapsed

                if text:
                    print_reflexion(label, color, text, t_disp, elapsed)

    # ── Zusammenfassung ── #
    print(f"\n{BOLD}{C}{'═'*72}{RESET}")
    print(f"{BOLD}ERGEBNIS nach {args.steps} Schritten{RESET}")
    print(f"{DIM}{'─'*70}{RESET}")

    for agent, color, label in [
        (toni_a, C, "Toni-A"),
        (toni_b, M, "Toni-B"),
    ]:
        h = agent.history
        avg_wb = float(np.mean(h["wellbeing"])) if h["wellbeing"] else 0.0
        end_wb = h["wellbeing"][-1] if h["wellbeing"] else 0.0
        crises = sum(
            1 for e, hy in zip(h["energy"], h["hydration"])
            if e < 0.25 or hy < 0.25
        )
        print(f"\n  {color}{BOLD}{label}{RESET}")
        print(f"  Ø Wohlbefinden:  {avg_wb:+.4f}")
        print(f"  End-Wohlbefinden:{end_wb:+.4f}")
        print(f"  Krisenschritte:  {crises}")

        # Soziale Statistik
        sc = agent.social_self
        consumption_events = len(sc._consumption_log)
        print(f"  Soziale Ereignisse beobachtet: {consumption_events}")

    if api_calls > 0:
        print(f"\n  LLM-Reflexionen: {api_calls}  "
              f"Ø {api_time_total/api_calls:.1f}s/Reflexion")
        print(f"  {DIM}(Reflexionen hatten keinen Einfluss auf Tonis Verhalten.){RESET}")

    print(f"\n{BOLD}{C}{'═'*72}{RESET}\n")


if __name__ == "__main__":
    main()
