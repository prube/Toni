"""
Temporales Selbstmodell – drei Zeitfenster nach Northoff.

Northoff (TTC): Bewusstsein entsteht als Schnittstelle von
Vergangenheit, Gegenwart und Zukunft. Das Selbst ist keine Momentaufnahme,
sondern eine Trajektorie — ein System das sich zeitlich positionieren kann.

  Vergangenheit → Trend (lineare Regression über Zustandsverlauf)
  Gegenwart     → kommt von Interozeption (außerhalb dieses Moduls)
  Zukunft       → Projektion + Zeit-bis-Krise + antizipatorische Dringlichkeit

Der entscheidende Unterschied zu reiner Homöostase:
  Homöostase reagiert wenn die Not da ist.
  Temporales Selbst reagiert, weil es die Not kommen sieht.

Die Zukunfts-Dringlichkeit (future_urgency) fließt in den C-Vektor der
pymdp-Präferenzen — der Agent sucht Ressourcen BEVOR er sie akut braucht.

Vorhersagemodell:
  use_rnn=True  → Echo State Network (nichtlinear, lernt Konsum-Muster)
  use_rnn=False → linearer polyfit (schnell, kein Lernen)
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .body import NEED_NAMES, SETPOINT
from .world_model import WorldModelRNN

CRISIS_THRESHOLD = 0.25
_WINDOW_DEFAULT  = 50   # Schritte Vergangenheitsfenster
_HORIZON_DEFAULT = 80   # Schritte Zukunftshorizont


class TemporalSelf:
    """
    Drei-Fenster-Selbstmodell nach Northoff.

    Vergangenheit: gleitender Verlauf + Trendschätzung per linearer Regression
    Gegenwart:     letzter bekannter Körperzustand (extern über update() gesetzt)
    Zukunft:       Projektion, Zeit-bis-Krise, antizipatorische Dringlichkeit

    Der Agent weiß nicht nur wie es ihm jetzt geht — er weiß wohin er sich bewegt.
    """

    def __init__(self, window: int = _WINDOW_DEFAULT, horizon: int = _HORIZON_DEFAULT,
                 use_rnn: bool = True, past_bias: float = 0.0):
        self.window  = window
        self.horizon = horizon
        self._history: deque = deque(maxlen=window)

        # Neuronales Weltmodell (ESN) — optional, ersetzt linearen polyfit
        self._rnn: WorldModelRNN | None = WorldModelRNN() if use_rnn else None
        self._fit_every: int = 5       # ESN alle 5 Schritte neu trainieren
        self._steps_since_fit: int = 0
        # Northoff temporale Stasis: past_bias > 0 lässt ältere Samples dominieren
        self._past_bias: float = past_bias

    # ------------------------------------------------------------------ #
    #  Zustandsupdate                                                      #
    # ------------------------------------------------------------------ #

    def update(
        self,
        body_state: np.ndarray,
        wellbeing:  float,
        valence:    float,
        position:   tuple,
        t:          int,
    ) -> None:
        """Aktuellen Zustand ins Vergangenheitsfenster aufnehmen."""
        self._history.append({
            "t":         t,
            "body":      body_state.copy(),
            "wellbeing": wellbeing,
            "valence":   valence,
            "position":  position,
        })

        # ESN neu trainieren wenn genug Verlauf vorhanden
        if self._rnn is not None:
            self._steps_since_fit += 1
            if self._steps_since_fit >= self._fit_every and len(self._history) >= 10:
                mat = self._body_matrix()
                if mat is not None:
                    self._rnn.fit(mat, past_bias=self._past_bias)
                    self._steps_since_fit = 0

    # ------------------------------------------------------------------ #
    #  Vergangenheit → Trend                                              #
    # ------------------------------------------------------------------ #

    def trend(self) -> np.ndarray:
        """
        Linearer Trend jedes Körperbedürfnisses (Einheiten/Schritt).

        Positiv = verbessernd, negativ = verschlechternd.
        Gibt np.zeros(4) zurück wenn nicht genug Verlauf vorhanden.
        """
        mat = self._body_matrix()
        if mat is None:
            return np.zeros(4)
        x = np.arange(len(mat), dtype=float)
        return np.array([np.polyfit(x, mat[:, i], 1)[0] for i in range(4)])

    def wellbeing_trend(self) -> float:
        """Steigung des Wohlbefindens (Einheiten/Schritt)."""
        if len(self._history) < 3:
            return 0.0
        wbs = np.array([e["wellbeing"] for e in self._history])
        x   = np.arange(len(wbs), dtype=float)
        return float(np.polyfit(x, wbs, 1)[0])

    # ------------------------------------------------------------------ #
    #  Zukunft → Projektion und Dringlichkeit                             #
    # ------------------------------------------------------------------ #

    def project(self, steps: int) -> np.ndarray:
        """
        Körperzustand in `steps` Schritten.

        RNN-Pfad (wenn trainiert): autoregressive ESN-Rollout
        Linear-Fallback:           aktuell + trend * steps, geclippt [0,1]
        """
        if not self._history:
            return SETPOINT.copy()
        current = self._history[-1]["body"].copy()
        if self._rnn is not None and self._rnn.is_trained:
            traj = self._rnn.predict(current, steps)
            return traj[-1]
        return np.clip(current + self.trend() * steps, 0.0, 1.0)

    def project_trajectory(self, steps: int) -> np.ndarray:
        """
        Vollständige projizierte Trajektorie der nächsten `steps` Schritte.

        RNN-Pfad: ESN-Rollout → nichtlineare Kurve
        Linear-Fallback: lineare Interpolation

        Gibt (steps, 4) zurück.
        """
        if not self._history:
            return np.tile(SETPOINT, (steps, 1))
        current = self._history[-1]["body"].copy()
        if self._rnn is not None and self._rnn.is_trained:
            return self._rnn.predict(current, steps)
        # Linear: aktuell + trend * t für t in [1..steps]
        slopes = self.trend()
        ts = np.arange(1, steps + 1, dtype=float)
        return np.clip(current[None, :] + slopes[None, :] * ts[:, None], 0.0, 1.0)

    def time_to_crisis(self) -> list:
        """
        Pro Bedürfnis: Schritte bis zur Krise (Zustand < CRISIS_THRESHOLD).

        RNN-Pfad (wenn trainiert): nutzt ESN-Trajektorie (nichtlinear!)
        Linear-Fallback:           Steigung × Istzustand

        None  = keine Krise projiziert
        0     = bereits in Krise
        int   = geschätzte Schritte bis Krise
        """
        if not self._history:
            return [None] * 4
        current = self._history[-1]["body"]

        # ── RNN-Pfad: Trajektorie scannen ── #
        if self._rnn is not None and self._rnn.is_trained:
            traj = self.project_trajectory(self.horizon)   # (horizon, 4)
            result = []
            for i in range(4):
                v = float(current[i])
                if v <= CRISIS_THRESHOLD:
                    result.append(0)
                else:
                    below = np.where(traj[:, i] <= CRISIS_THRESHOLD)[0]
                    result.append(int(below[0]) if len(below) > 0 else None)
            return result

        # ── Linearer Fallback ── #
        slopes = self.trend()
        result = []
        for i in range(4):
            v = float(current[i])
            s = float(slopes[i])
            if v <= CRISIS_THRESHOLD:
                result.append(0)
            elif s < -1e-6:
                ttc = (v - CRISIS_THRESHOLD) / abs(s)
                result.append(int(ttc))
            else:
                result.append(None)
        return result

    def future_urgency(self) -> np.ndarray:
        """
        Antizipatorische Dringlichkeit pro Bedürfnis [0, 1].

        Formel:
          ttc = None  →  0.0   (keine Krise projiziert)
          ttc = 0     →  1.0   (bereits in Krise)
          ttc > 0     →  max(0, 1 - ttc/horizon)

        Beispiel (horizon=80):
          ttc=80  →  urgency=0.0   (Krise am Horizont, kein Handlungsdruck)
          ttc=40  →  urgency=0.5   (halber Weg, erhöhte Motivation)
          ttc=10  →  urgency=0.875 (unmittelbar, starker Drang)

        Das ist Northoffs Kern-Unterschied zu reiner Homöostase:
        Der Agent handelt antizipatorisch, nicht erst reaktiv.
        """
        ttcs    = self.time_to_crisis()
        urgency = np.zeros(4)
        for i, ttc in enumerate(ttcs):
            if ttc is None:
                urgency[i] = 0.0
            elif ttc <= 0:
                urgency[i] = 1.0
            else:
                urgency[i] = max(0.0, 1.0 - ttc / self.horizon)
        return urgency

    # ------------------------------------------------------------------ #
    #  Zusammenfassung für LLM-Kortex und Visualisierung                  #
    # ------------------------------------------------------------------ #

    def temporal_summary(self) -> dict:
        """
        Drei-Fenster-Zusammenfassung.

        Vergangenheit: Trend pro Bedürfnis (Richtung + Stärke)
        Gegenwart:     Projektion in 50 Schritten
        Zukunft:       Zeit bis Krise + antizipatorische Dringlichkeit
        """
        trends  = self.trend()
        ttcs    = self.time_to_crisis()
        urgency = self.future_urgency()
        proj50  = self.project(50)

        def _trend_str(s: float) -> str:
            if   s >  0.0005: return f"↑ +{s:.4f}/Schritt"
            elif s < -0.0005: return f"↓ {s:.4f}/Schritt"
            return "→ stabil"

        def _ttc_str(ttc) -> str:
            if ttc is None: return "keine Krise projiziert"
            if ttc == 0:    return "⚠ BEREITS IN KRISE"
            return f"~{ttc} Schritte"

        dominant_threat = int(np.argmax(urgency[:2]))

        rnn_info = (
            self._rnn.summary()
            if self._rnn is not None
            else {"trainiert": False}
        )

        return {
            "weltmodell": (
                "RNN (Echo State)" if (self._rnn and self._rnn.is_trained)
                else "linear (polyfit)"
            ),
            "vergangenheit_trend": {
                NEED_NAMES[i]: _trend_str(float(trends[i])) for i in range(4)
            },
            "gegenwart_projektion_in_50_schritten": {
                NEED_NAMES[i]: round(float(proj50[i]), 3) for i in range(4)
            },
            "zukunft_krise_in_schritten": {
                NEED_NAMES[i]: _ttc_str(ttcs[i]) for i in range(4)
            },
            "zukunft_dringlichkeit": {
                NEED_NAMES[i]: round(float(urgency[i]), 3) for i in range(4)
            },
            "dringlichste_bedrohung": NEED_NAMES[dominant_threat],
            "wohlbefinden_trend": _trend_str(self.wellbeing_trend()),
            "rnn": rnn_info,
        }

    # ------------------------------------------------------------------ #
    #  Hilfsmethoden                                                       #
    # ------------------------------------------------------------------ #

    def _body_matrix(self):
        """Körperzustands-Matrix aus dem Verlauf. None wenn < 3 Einträge."""
        if len(self._history) < 3:
            return None
        return np.array([e["body"] for e in self._history])
