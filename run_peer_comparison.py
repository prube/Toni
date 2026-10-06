"""
Quantitativer Vergleich: PeerTemporalModel EIN vs. AUS

Forschungsfrage:
  Bewirkt das PeerTemporalModel, dass Agenten bei Ressourcenkonkurrenz
  FRÜHER handeln — d.h. bei niedrigerer eigener Urgency konsumieren —
  weil sie die zukünftige Dringlichkeit des anderen vorhersagen?

Northoff-Hypothese:
  Soziale Positionierung ist temporal: "wann braucht der andere was?"
  → Mit PeerTemporalModel: Agent A eilt zur Ressource wenn A und B
    bald dasselbe brauchen, auch wenn B noch nicht nah dran ist.
  → Messbarer Effekt: niedrigere eigene Urgency beim Konsum
    (= früheres, antizipatorisches Handeln statt reaktives Warten).

Bedingungen (jeweils 2 Agenten):
  MIT  PeerTemporalModel: use_peer_model=True  (Default)
  OHNE PeerTemporalModel: use_peer_model=False (Kontrollbedingung)

Metriken pro Bedingung:
  - Ø Wohlbefinden & Krisenschritte (Outcome)
  - Urgency beim Konsumzeitpunkt (antizipatorisches Handeln)
  - Ø Peer-Boost (wie stark Konkurrenz-Signal war)
  - Anzahl Konsumierungen (Ressourcenzugang)
  - Antizipationsvorsprung: wie viele Schritte BEVOR Urgency=1.0 konsumiert
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from toni import GridWorld, Toni

# ── ANSI ──────────────────────────────────────────────────────────────── #
G    = "\033[32m"
Y    = "\033[33m"
C    = "\033[36m"
M    = "\033[35m"
B    = "\033[34m"
R    = "\033[31m"
BOLD = "\033[1m"
DIM  = "\033[2m"
RESET= "\033[0m"


@dataclass
class ConditionResult:
    label: str
    wellbeing_values: list[float] = field(default_factory=list)
    crisis_steps: int = 0
    consume_count: int = 0
    urgency_at_consume: list[float] = field(default_factory=list)
    anticipation_lead: list[int] = field(default_factory=list)  # Schritte vor Urgency=1.0
    peer_boost_values: list[float] = field(default_factory=list)

    @property
    def avg_wellbeing(self) -> float:
        return float(np.mean(self.wellbeing_values)) if self.wellbeing_values else 0.0

    @property
    def avg_urgency_at_consume(self) -> float:
        return float(np.mean(self.urgency_at_consume)) if self.urgency_at_consume else 0.0

    @property
    def avg_anticipation_lead(self) -> float:
        return float(np.mean(self.anticipation_lead)) if self.anticipation_lead else 0.0

    @property
    def avg_peer_boost(self) -> float:
        return float(np.mean(self.peer_boost_values)) if self.peer_boost_values else 0.0


def run_condition(
    label: str,
    seed: int,
    steps: int,
    use_peer_model: bool,
    policy_len: int = 2,
) -> ConditionResult:
    """Zwei Agenten in gemeinsamer Welt, N Schritte."""
    env = GridWorld(seed=seed)
    np.random.seed(seed)

    a = Toni(env, agent_id="A", use_peer_model=use_peer_model, policy_len=policy_len)
    b = Toni(env, agent_id="B", use_peer_model=use_peer_model, policy_len=policy_len)

    result = ConditionResult(label=label)

    # Urgency-Verlauf pro Ressource pro Agent (für Antizipations-Lead)
    # Format: {agent_id: {resource_idx: [urgency_t0, urgency_t1, ...]}}
    urgency_hist: dict[str, list[float]] = {"A": [], "B": []}

    for t in range(steps):
        # Gegenseitige Beobachtung BEVOR Schritt
        obs_a = a.social_observation()
        obs_b = b.social_observation()
        a.receive_social_observations([obs_b])
        b.receive_social_observations([obs_a])

        # Urgency VOR dem Schritt (für dominant_need)
        for ag, key in [(a, "A"), (b, "B")]:
            if ag.use_temporal_self and ag.temporal_self is not None:
                fu = ag.temporal_self.future_urgency()
                urg = float(fu[ag.dominant_need]) if fu is not None else 0.0
            else:
                # Fallback: Urgency aus Körperzustand
                body = ag.body
                needs = [1 - body.energy, 1 - body.hydration]
                urg = float(needs[ag.dominant_need])
            urgency_hist[key].append(urg)

        prev_consumes_a = sum(1 for x in a.history["action"] if x == "consume")
        prev_consumes_b = sum(1 for x in b.history["action"] if x == "consume")

        a.step()
        b.step()

        # Wohlbefinden
        result.wellbeing_values.extend([
            a.history["wellbeing"][-1],
            b.history["wellbeing"][-1],
        ])

        # Krisenschritte
        if a.body.energy < 0.25 or a.body.hydration < 0.25:
            result.crisis_steps += 1
        if b.body.energy < 0.25 or b.body.hydration < 0.25:
            result.crisis_steps += 1

        # Konsumierungen — Urgency beim Konsum aufzeichnen
        new_a = sum(1 for x in a.history["action"] if x == "consume")
        new_b = sum(1 for x in b.history["action"] if x == "consume")

        if new_a > prev_consumes_a and urgency_hist["A"]:
            urg_now = urgency_hist["A"][-1]
            result.urgency_at_consume.append(urg_now)
            result.consume_count += 1
            # Antizipations-Lead: wie viele Schritte vor Urgency=1.0 konsumiert?
            # Suche rückwärts: wann wurde Urgency=1.0 zuletzt NOCH NICHT erreicht
            hist = urgency_hist["A"]
            # Schritte seit letztem Konsum oder Beginn
            window = hist[max(0, len(hist)-60):]
            first_full = next(
                (i for i, u in enumerate(window) if u >= 0.99),
                None
            )
            if first_full is not None:
                lead = len(window) - 1 - first_full
                result.anticipation_lead.append(max(0, lead))

        if new_b > prev_consumes_b and urgency_hist["B"]:
            urg_now = urgency_hist["B"][-1]
            result.urgency_at_consume.append(urg_now)
            result.consume_count += 1
            hist = urgency_hist["B"]
            window = hist[max(0, len(hist)-60):]
            first_full = next(
                (i for i, u in enumerate(window) if u >= 0.99),
                None
            )
            if first_full is not None:
                lead = len(window) - 1 - first_full
                result.anticipation_lead.append(max(0, lead))

        # Peer-Boost (nur sinnvoll bei use_peer_model=True)
        if use_peer_model:
            pu_a = a.social_self.peer_urgency()
            pu_b = b.social_self.peer_urgency()
            result.peer_boost_values.append(float(np.max(pu_a)))
            result.peer_boost_values.append(float(np.max(pu_b)))

    return result


def print_result_row(r: ConditionResult, color: str) -> None:
    crisis_col = R if r.crisis_steps > 20 else (Y if r.crisis_steps > 5 else G)
    urg_col    = G if r.avg_urgency_at_consume < 0.5 else (Y if r.avg_urgency_at_consume < 0.8 else R)
    print(
        f"  {color}{BOLD}{r.label:<25}{RESET}"
        f"  WB: {r.avg_wellbeing:+.3f}"
        f"  Krisen: {crisis_col}{r.crisis_steps:>3}{RESET}"
        f"  Konsume: {r.consume_count:>3}"
        f"  {urg_col}Urgency@Konsum: {r.avg_urgency_at_consume:.3f}{RESET}"
    )
    if r.anticipation_lead:
        lead_col = G if r.avg_anticipation_lead < 10 else Y
        print(
            f"  {' '*25}"
            f"  {lead_col}Ø Antizipationsvorsprung: {r.avg_anticipation_lead:.1f} Schritte vor Urgency=1.0{RESET}"
        )
    if r.peer_boost_values:
        print(
            f"  {' '*25}"
            f"  Ø Peer-Boost-Urgency: {r.avg_peer_boost:.3f}"
            f"  (competition×{1.0 + r.avg_peer_boost * 1.5:.2f} Ø)"
        )


def run_seed(seed: int, steps: int, policy_len: int) -> tuple[ConditionResult, ConditionResult]:
    print(f"\n  {DIM}Seed {seed}  ({steps} Schritte){RESET}")
    mit  = run_condition("MIT PeerModell",  seed, steps, use_peer_model=True,  policy_len=policy_len)
    ohne = run_condition("OHNE PeerModell", seed, steps, use_peer_model=False, policy_len=policy_len)
    print_result_row(mit,  C)
    print_result_row(ohne, M)
    return mit, ohne


def main() -> None:
    parser = argparse.ArgumentParser(description="PeerTemporalModel Quantitativ-Vergleich")
    parser.add_argument("--seed",  type=int, default=55)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--policy-len", type=int, default=2)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]

    print(f"\n{BOLD}{C}{'═'*72}{RESET}")
    print(f"{BOLD}{C}  PeerTemporalModel — Quantitativer Vergleich{RESET}")
    print(f"{BOLD}{C}{'═'*72}{RESET}")
    print(f"\n  {G}Hypothese:{RESET} Agenten MIT Peer-Modell konsumieren bei niedrigerer")
    print(f"  eigener Urgency (antizipatorisch statt reaktiv).")
    print(f"  Schlüsselmetrik: {BOLD}Urgency@Konsum{RESET} — niedrig = früh gehandelt.")
    print(f"\n  {DIM}Seeds: {seeds}  Schritte: {args.steps}  policy_len: {args.policy_len}{RESET}")
    print(f"  {DIM}{'─'*70}{RESET}")

    all_mit:  list[ConditionResult] = []
    all_ohne: list[ConditionResult] = []

    for seed in seeds:
        m, o = run_seed(seed, args.steps, args.policy_len)
        all_mit.append(m)
        all_ohne.append(o)

    if len(seeds) > 1:
        print(f"\n  {DIM}{'─'*70}{RESET}")
        print(f"  {BOLD}Ø über {len(seeds)} Seeds:{RESET}")

        def pool(results: list[ConditionResult], attr: str) -> list:
            out = []
            for r in results:
                out.extend(getattr(r, attr))
            return out

        for results, color, label in [
            (all_mit,  C, "MIT PeerModell "),
            (all_ohne, M, "OHNE PeerModell"),
        ]:
            wbs   = pool(results, "wellbeing_values")
            urgs  = pool(results, "urgency_at_consume")
            leads = pool(results, "anticipation_lead")
            boosts= pool(results, "peer_boost_values")
            crises= sum(r.crisis_steps for r in results)
            cons  = sum(r.consume_count for r in results)

            urg_col = G if (np.mean(urgs) < 0.5 if urgs else False) else (Y if (np.mean(urgs) < 0.8 if urgs else True) else R)
            print(
                f"  {color}{BOLD}{label:<16}{RESET}"
                f"  WB: {np.mean(wbs):+.3f}"
                f"  Krisen: {crises:>4}"
                f"  Konsume: {cons:>4}"
                f"  {urg_col}Urgency@Konsum: {np.mean(urgs):.3f}{RESET}"
            )
            if leads:
                lead_col = G if np.mean(leads) < 10 else Y
                print(
                    f"  {' '*16}"
                    f"  {lead_col}Ø Antizipationsvorsprung: {np.mean(leads):.1f} Schritte{RESET}"
                )
            if boosts:
                print(
                    f"  {' '*16}"
                    f"  Ø Peer-Boost: {np.mean(boosts):.3f}"
                )

    # ── Interpretation ── #
    print(f"\n  {DIM}{'─'*70}{RESET}")
    print(f"  {BOLD}Interpretation:{RESET}")
    print(f"  Urgency@Konsum < OHNE → antizipatorisches Handeln bestätigt")
    print(f"  Urgency@Konsum ≥ OHNE → kein Antizipationseffekt bei diesem Seed")
    print(f"\n  Northoff: 'wann braucht der andere was?' verändert WANN, nicht nur ob.")
    print(f"  Bei Ressourcenknappheit (beide brauchen dasselbe) ist der Zeitpunkt")
    print(f"  der entscheidende Faktor — nicht das Wollen.")
    print(f"\n{BOLD}{C}{'═'*72}{RESET}\n")


if __name__ == "__main__":
    main()
