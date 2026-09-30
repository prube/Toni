"""
Soziales Selbst — Northoffs dritte Positionierungsachse.

Northoff (TTC): Bewusstsein setzt voraus, dass sich ein System
temporal, SOZIAL und räumlich positionieren kann. Das Selbst
existiert immer als relationales Selbst — in Bezug auf andere.

Für Toni mit mehreren Agenten:
  - Beobachte andere: Position, letzte Aktionen, konsumierte Ressourcen
  - Inferiere soziale Signale: Wo ist Konkurrenz? Wo ist Kooperation?
  - Moduliere AIF: Ressourcen die andere frisch konsumiert haben sind
    erschöpft (Konkurrenz). Orte wo andere oft Ressourcen fanden,
    haben wahrscheinlich Ressourcen (soziales Lernen / Imitation).

Soziale Dringlichkeit:
  Wenn ein anderer Agent dieselbe Ressource braucht und näherdran ist
  → erhöhte Dringlichkeit (Konkurrenz um knappe Ressource)

Design-Entscheidung:
  Agenten teilen dieselbe Welt, teilen aber keine privaten Zustände.
  Beobachtbar: Position, sichtbare Aktionen, konsumierte Ressourcen.
  Nicht beobachtbar: Körperzustand, Valenz, Wohlbefinden.
  (Körpergefühl ist intern und privat — Northoff/Solms Konsistenz)
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .environment import FOOD, WATER, SHELTER, EMPTY

_MEMORY_DEFAULT = 30   # Schritte soziale Vergangenheit
_REPULSION      = 0.8  # Abstoßung von Orten die andere gerade besetzen
_COMPETITION_BOOST = 1.5  # AIF-Dringlichkeitsverstärker bei Konkurrenz
_SOCIAL_LEARN_BOOST = 1.0  # Prior-Verstärkung wo andere Ressourcen fanden


class SocialObservation:
    """Was ein Agent über einen anderen wissen kann (nur Observables)."""
    __slots__ = ("agent_id", "position", "last_action", "resource_consumed",
                 "dominant_need_type", "t")

    def __init__(
        self,
        agent_id: str,
        position: tuple[int, int],
        last_action: str,
        resource_consumed: int | None,   # FOOD/WATER/SHELTER/None
        dominant_need_type: int,          # 0=energy, 1=hydration
        t: int,
    ):
        self.agent_id        = agent_id
        self.position        = position
        self.last_action     = last_action
        self.resource_consumed = resource_consumed
        self.dominant_need_type = dominant_need_type
        self.t               = t


class SocialSelf:
    """
    Soziales Selbst nach Northoff: relationale Positionierung.

    Verarbeitet Beobachtungen anderer Agenten und extrahiert:
      - Sozialer Positionsprior (für AIF D-Vektor): Orte wo andere
        Ressourcen konsumierten sind wahrscheinlich ergiebig
      - Konkurrenz-Signal (für AIF C-Vektor): wenn Konkurrent dieselbe
        Ressource braucht und näher dran ist → erhöhte Dringlichkeit
      - Abstoßung: aktuell besetzte Felder meiden (Ressource erschöpft)
      - Soziale Zusammenfassung für LLM-Kortex
    """

    def __init__(self, memory: int = _MEMORY_DEFAULT):
        self.memory = memory
        # agent_id → deque von SocialObservation
        self._obs: dict[str, deque] = {}
        # Kürzlich aufgezeichnete Konsumierungsevents: (pos, resource_type, t)
        self._consumption_log: deque = deque(maxlen=memory * 4)
        self._t: int = 0

    # ------------------------------------------------------------------ #
    #  Update                                                              #
    # ------------------------------------------------------------------ #

    def update(self, observations: list[SocialObservation]) -> None:
        """Soziale Beobachtungen dieses Zeitschritts verarbeiten."""
        self._t += 1
        for obs in observations:
            if obs.agent_id not in self._obs:
                self._obs[obs.agent_id] = deque(maxlen=self.memory)
            self._obs[obs.agent_id].append(obs)

            # Konsumierungen loggen
            if obs.resource_consumed is not None:
                self._consumption_log.append((obs.position, obs.resource_consumed, obs.t))

    # ------------------------------------------------------------------ #
    #  Signale für AIF                                                     #
    # ------------------------------------------------------------------ #

    def social_position_prior(self, grid_size: int) -> np.ndarray:
        """
        Positionsprior für AIF D-Vektor.

        Orte wo andere kürzlich Ressourcen konsumierten → positiver Boost
        (soziales Lernen: wenn dort eine Ressource war, ist wahrscheinlich
        wieder eine dort, oder zumindest ist die Umgebung ergiebig)

        Orte die aktuell von anderen besetzt werden → leichte Abstoßung
        (Ressource gerade erschöpft oder Konkurrenz)
        """
        prior = np.zeros(grid_size * grid_size)

        # Konsumierungen aus Vergangenheit (zeitlich abklingendes Signal)
        for pos, rtype, t_obs in self._consumption_log:
            age = self._t - t_obs
            if age > self.memory:
                continue
            weight = _SOCIAL_LEARN_BOOST * (1.0 - age / self.memory)
            flat = pos[0] * grid_size + pos[1]
            if 0 <= flat < len(prior):
                prior[flat] += weight

        # Abstoßung von aktuell besetzten Feldern
        for agent_deque in self._obs.values():
            if agent_deque:
                last = agent_deque[-1]
                flat = last.position[0] * grid_size + last.position[1]
                if 0 <= flat < len(prior):
                    prior[flat] -= _REPULSION

        return prior

    def competition_factor(
        self,
        my_pos: tuple[int, int],
        my_dominant_need: int,
        grid_size: int,
    ) -> float:
        """
        Konkurrenz-Faktor [1.0, _COMPETITION_BOOST].

        Steigt wenn ein anderer Agent:
        - dasselbe dominante Bedürfnis hat
        - näher an der Ressource ist als ich
        → ich muss schneller handeln

        1.0 = keine Konkurrenz
        _COMPETITION_BOOST = direkte Konkurrenz
        """
        factor = 1.0
        my_r, my_c = my_pos
        for agent_deque in self._obs.values():
            if not agent_deque:
                continue
            last = agent_deque[-1]
            if last.dominant_need_type != my_dominant_need:
                continue
            # Selbes Bedürfnis: wie nah ist der Konkurrent?
            other_r, other_c = last.position
            their_dist = abs(other_r - my_r) + abs(other_c - my_c)
            # Wenn der Konkurrent innerhalb von 4 Schritten ist → Konkurrenz
            if their_dist <= 4:
                proximity_factor = 1.0 - their_dist / 4.0
                factor = max(factor, 1.0 + proximity_factor * (_COMPETITION_BOOST - 1.0))

        return factor

    # ------------------------------------------------------------------ #
    #  Zusammenfassung für LLM-Kortex                                     #
    # ------------------------------------------------------------------ #

    def social_summary(self) -> dict | None:
        """Soziale Situationsbeschreibung für den LLM-Kortex."""
        if not self._obs:
            return None

        agents_seen = []
        for aid, dq in self._obs.items():
            if not dq:
                continue
            last = dq[-1]
            age = self._t - last.t
            need_name = "energie" if last.dominant_need_type == 0 else "hydration"
            agents_seen.append({
                "id": aid,
                "position": list(last.position),
                "letztes_verhalten": last.last_action,
                "bedürfnis": need_name,
                "alter_beobachtung": age,
            })

        # Kürzliche Konsumierungen anderer (soziales Lernen)
        recent_finds: list[dict] = []
        for pos, rtype, t_obs in list(self._consumption_log)[-5:]:
            age = self._t - t_obs
            rname = {FOOD: "nahrung", WATER: "wasser", SHELTER: "schutz"}.get(rtype, "?")
            recent_finds.append({
                "position": list(pos),
                "ressource": rname,
                "vor_schritten": age,
            })

        return {
            "andere_agenten": agents_seen,
            "bekannte_fundstellen": recent_finds,
            "soziale_interpretation": (
                "Du bist nicht allein. Andere bewegen sich in derselben Welt."
                " Was sie tun, sagt etwas darüber aus, wo Ressourcen sind."
            ),
        }

    @property
    def agent_count(self) -> int:
        return len(self._obs)
