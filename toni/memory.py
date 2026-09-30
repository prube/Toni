"""
Stufe 9: Autobiografisches Gedächtnis.

Vergangene Erfahrungen müssen die aktuelle Dynamik verändern – nicht bloß
gespeichert werden. Orte mit positiver Valenz erhalten höhere Prior-Wahrscheinlichkeit.
Die individuelle Geschichte formt die Gegenwart.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Any


class Episode:
    """Ein autobiografischer Erfahrungsmoment."""

    __slots__ = ("body_state", "valence", "wellbeing", "position",
                 "action", "dominant_need", "temporal_state", "t")

    def __init__(self, body_state: np.ndarray, valence: float,
                 wellbeing: float, position: tuple[int, int],
                 action: str, dominant_need: int,
                 temporal_state: np.ndarray, t: int):
        self.body_state = body_state
        self.valence = valence
        self.wellbeing = wellbeing
        self.position = position
        self.action = action
        self.dominant_need = dominant_need
        self.temporal_state = temporal_state
        self.t = t


class AutobiographicalMemory:
    """
    Episodisches Gedächtnis: Vergangenheit verändert aktuelle Dynamik.

    Orte mit positiv erlebter Valenz erhalten höhere Prior-Gewichte,
    d.h. das System "erinnert sich" an gute Plätze und bevorzugt sie.
    """

    def __init__(self, capacity: int = 2000):
        self.episodes: deque[Episode] = deque(maxlen=capacity)
        self._location_valence_cache: dict[tuple[int, int], list[float]] = {}
        self.t: int = 0

    def store(self, body_state: np.ndarray, valence: float,
              wellbeing: float, position: tuple[int, int],
              action: str, dominant_need: int,
              temporal_state: np.ndarray,
              rumination_factor: float = 1.0) -> None:
        ep = Episode(
            body_state=body_state.copy(),
            valence=valence,
            wellbeing=wellbeing,
            position=position,
            action=action,
            dominant_need=dominant_need,
            temporal_state=temporal_state.copy(),
            t=self.t,
        )
        self.episodes.append(ep)

        # Cache für Orts-Valenz aktualisieren
        # Rumination (Depression): negative Erfahrungen brennen sich stärker ein
        stored_valence = valence * rumination_factor if valence < 0 else valence
        pos_key = position
        if pos_key not in self._location_valence_cache:
            self._location_valence_cache[pos_key] = []
        self._location_valence_cache[pos_key].append(stored_valence)
        # Nur letzte 50 Besuche behalten (Verblassen alter Erinnerungen)
        if len(self._location_valence_cache[pos_key]) > 50:
            self._location_valence_cache[pos_key].pop(0)

        self.t += 1

    def location_prior(self, position: tuple[int, int]) -> float:
        """
        Erwartete Valenz an diesem Ort – formt Prior für Entscheidung.
        Positive Werte: Ort wurde als gut erlebt (Nahrung/Wasser gefunden).
        Negative Werte: Ort war mit Notlagen assoziiert.
        """
        vals = self._location_valence_cache.get(position)
        if not vals:
            return 0.0
        # Neuere Erfahrungen zählen mehr (exponentielles Abklingen)
        weights = np.exp(np.linspace(-2.0, 0.0, len(vals)))
        return float(np.average(vals, weights=weights))

    def recent_wellbeing_trend(self, n: int = 30) -> float:
        """
        Trend im Wohlbefinden über die letzten n Schritte.
        Positiv: Erholung. Negativ: Verschlechterung.
        """
        recent = list(self.episodes)[-n:]
        if len(recent) < 2:
            return 0.0
        wbs = [e.wellbeing for e in recent]
        return float(np.mean(np.diff(wbs)))

    def dominant_need_history(self, n: int = 20) -> int:
        """Welches Bedürfnis dominierte zuletzt am häufigsten?"""
        recent = list(self.episodes)[-n:]
        if not recent:
            return 0
        needs = [e.dominant_need for e in recent]
        return int(np.bincount(needs).argmax())

    def __len__(self) -> int:
        return len(self.episodes)
