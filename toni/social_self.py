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

Soziale Dringlichkeit (reaktiv):
  Wenn ein anderer Agent dieselbe Ressource braucht und näherdran ist
  → erhöhte Dringlichkeit (Konkurrenz um knappe Ressource)

Soziale Dringlichkeit (antizipatorisch — PeerTemporalModel):
  Aus den Konsum-Intervallen des anderen Agenten wird vorhergesagt,
  wann er als nächstes eine Ressource benötigen wird.
  → "In 12 Schritten braucht der andere Wasser — ich auch. Jetzt handeln."

Design-Entscheidung:
  Agenten teilen dieselbe Welt, teilen aber keine privaten Zustände.
  Beobachtbar: Position, sichtbare Aktionen, konsumierte Ressourcen.
  Nicht beobachtbar: Körperzustand, Valenz, Wohlbefinden.
  (Körpergefühl ist intern und privat — Northoff/Solms Konsistenz)
  Das PeerTemporalModel rekonstruiert das temporale Selbst des anderen
  aus behavioralen Beobachtungen — keine direkte Zustandsübertragung.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .environment import FOOD, WATER, SHELTER, EMPTY

_MEMORY_DEFAULT = 30   # Schritte soziale Vergangenheit
_REPULSION      = 0.8  # Abstoßung von Orten die andere gerade besetzen
_COMPETITION_BOOST = 1.5  # AIF-Dringlichkeitsverstärker bei Konkurrenz
_SOCIAL_LEARN_BOOST = 1.0  # Prior-Verstärkung wo andere Ressourcen fanden
_PEER_HORIZON   = 40   # Schritte für Peer-Urgency-Vorhersage


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


class PeerTemporalModel:
    """
    Modell des temporalen Selbst eines anderen Agenten.

    Northoff: Bewusstsein positioniert sich sozial indem es nicht nur
    fragt "wo ist der andere?" sondern "wann braucht der andere was?".
    Dieses Modell rekonstruiert die Zeitstruktur des anderen Agenten
    aus beobachtbarem Verhalten — ohne Zugriff auf seinen Körperzustand.

    Mechanismus:
      1. Konsum-Events loggen: (t, ressource_typ)
      2. Inter-Konsum-Intervalle berechnen → geschätzte Depletionsrate
      3. Vorwärts projizieren: "letzter Wasser-Konsum vor X Schritten,
         typisches Interval Y → nächste Wassernot in Z Schritten"
      4. future_urgency[ressource] ∈ [0,1] — Urgency-Format kompatibel
         mit TemporalSelf.future_urgency()

    Kalibriert nach 2+ Konsum-Events pro Ressourcentyp.
    Davor: Fallback auf dominant_need_type der letzten Beobachtung.
    """

    def __init__(self, agent_id: str, horizon: int = _PEER_HORIZON):
        self.agent_id = agent_id
        self.horizon  = horizon
        # (t, resource_type) — Konsum-Zeitreihe
        self._consume_log: list[tuple[int, int]] = []
        self._last_obs: SocialObservation | None = None
        self._t: int = 0

    def update(self, obs: SocialObservation) -> None:
        """Beobachtung verarbeiten. Konsum-Events ins Modell aufnehmen."""
        self._last_obs = obs
        self._t = obs.t
        if obs.resource_consumed is not None:
            self._consume_log.append((obs.t, obs.resource_consumed))
            if len(self._consume_log) > 30:
                self._consume_log = self._consume_log[-30:]

    def future_urgency(self) -> np.ndarray:
        """
        Vorhergesagte Dringlichkeit des anderen Agenten.
        Shape: (2,) für [energie_urgency, hydration_urgency].

        Formel (analog zu TemporalSelf.future_urgency):
          time_to_next = avg_interval - time_since_last_consume
          urgency = max(0, 1 - time_to_next / horizon)

        Bei ttc=0 (überfällig): urgency=1.0
        Bei ttc=horizon: urgency=0.0
        """
        urgency = np.zeros(2)

        if not self._consume_log:
            # Keine Konsum-Events: nutze letztes bekanntes Bedürfnis als schwaches Signal
            if self._last_obs is not None:
                urgency[self._last_obs.dominant_need_type] = 0.25
            return urgency

        for resource_idx in range(2):
            resource_type = FOOD if resource_idx == 0 else WATER
            events = [(t, r) for t, r in self._consume_log if r == resource_type]

            if not events:
                continue

            last_t = events[-1][0]
            time_since = self._t - last_t

            if len(events) < 2:
                # Nur 1 Event bekannt: generische Depletion (~60 Schritte)
                typical_interval = 60
            else:
                times = [t for t, _ in events]
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                # Neuere Intervalle stärker gewichten
                w = np.linspace(0.5, 1.0, len(intervals))
                typical_interval = float(np.average(intervals, weights=w))
                typical_interval = max(typical_interval, 1.0)  # div-by-zero-Schutz

            time_to_next = max(0.0, typical_interval - time_since)
            urgency[resource_idx] = max(0.0, 1.0 - time_to_next / self.horizon)

        return urgency

    @property
    def is_calibrated(self) -> bool:
        """True wenn ≥2 Konsum-Events für mindestens eine Ressource bekannt."""
        for resource_type in (FOOD, WATER):
            if sum(1 for _, r in self._consume_log if r == resource_type) >= 2:
                return True
        return False

    def summary(self) -> dict:
        pu = self.future_urgency()
        return {
            "kalibriert": self.is_calibrated,
            "konsum_events": len(self._consume_log),
            "vorhergesagte_urgency": {
                "energie":   round(float(pu[0]), 3),
                "hydration": round(float(pu[1]), 3),
            },
        }


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

    def __init__(self, memory: int = _MEMORY_DEFAULT, use_peer_model: bool = True):
        self.memory = memory
        self._use_peer_model = use_peer_model
        # agent_id → deque von SocialObservation
        self._obs: dict[str, deque] = {}
        # Kürzlich aufgezeichnete Konsumierungsevents: (pos, resource_type, t)
        self._consumption_log: deque = deque(maxlen=memory * 4)
        # Temporales Selbst-Modell pro bekanntem Agenten (nur wenn use_peer_model)
        self._peer_models: dict[str, PeerTemporalModel] = {}
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

            # Peer-Temporal-Modell aktualisieren (nur wenn aktiviert)
            if self._use_peer_model:
                if obs.agent_id not in self._peer_models:
                    self._peer_models[obs.agent_id] = PeerTemporalModel(obs.agent_id)
                self._peer_models[obs.agent_id].update(obs)

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

    def peer_urgency(self) -> np.ndarray:
        """
        Vorhergesagte maximale Dringlichkeit aller bekannten anderen Agenten.
        Shape: (2,) für [energie_urgency, hydration_urgency].

        Gibt das Maximum über alle Peer-Modelle zurück — konservativ:
        der Agent handelt als wäre die dringendste Bedrohung durch den
        anderen Agenten real.

        Antizipatorische Konkurrenz:
          peer_urgency[0] = 0.8 bedeutet: "In ca. 8 Schritten braucht der
          andere Agent Nahrung dringend" — wenn ich auch Nahrung brauche,
          sollte ich jetzt handeln, bevor der andere die Ressource nimmt.

        Nur kalibrierte Modelle werden genutzt (≥2 Konsum-Events pro
        Ressourcentyp). Unkalibrierte Modelle liefern zu viele false positives
        und stören die AIF-Policy.
        """
        if not self._peer_models:
            return np.zeros(2)
        calibrated = [m for m in self._peer_models.values() if m.is_calibrated]
        if not calibrated:
            return np.zeros(2)
        all_urgencies = np.array([m.future_urgency() for m in calibrated])
        return np.max(all_urgencies, axis=0)

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

        # Peer-Temporal-Modelle: vorhergesagte Dringlichkeit der anderen
        peer_temporal: dict = {}
        for aid, model in self._peer_models.items():
            peer_temporal[aid] = model.summary()

        # Antizipatorische Konkurrenz-Einschätzung
        pu = self.peer_urgency()
        competition_outlook = "keine Konkurrenz erwartet"
        if pu.max() > 0.6:
            need_name = "nahrung" if float(pu[0]) > float(pu[1]) else "wasser"
            competition_outlook = (
                f"hohe Konkurrenz für {need_name} in Kürze erwartet "
                f"(peer_urgency={pu.max():.2f})"
            )
        elif pu.max() > 0.3:
            competition_outlook = (
                f"moderate Konkurrenz möglich (peer_urgency={pu.max():.2f})"
            )

        return {
            "andere_agenten": agents_seen,
            "bekannte_fundstellen": recent_finds,
            "peer_temporal_modell": peer_temporal,
            "konkurrenz_prognose": competition_outlook,
            "soziale_interpretation": (
                "Du bist nicht allein. Andere bewegen sich in derselben Welt."
                " Was sie tun, sagt etwas darüber aus, wo Ressourcen sind."
                " Was sie bald brauchen werden, sagt etwas darüber aus,"
                " wann du handeln musst."
            ),
        }

    @property
    def agent_count(self) -> int:
        return len(self._obs)
