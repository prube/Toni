"""
Stufe 7+8: Northoff – intrinsische Zeitdynamik und temporo-spatial alignment.

Kernideen:
  - Das System hat eine permanente Eigenaktivität, auch ohne Prompt/Stimulus.
  - Diese Aktivität besitzt verschachtelte Zeitskalen (nestedness):
    langsame Oszillatoren modulieren die Amplitude schnellerer.
  - Temporo-spatial alignment: interne Phasen koppeln an externe Rhythmen
    (Tagesrhythmus der Umwelt) via Kuramoto-artiger Phasenkopplung.
  - Der langsame Zustand moduliert die Präzision der Interozeption:
    Gleiche Wahrnehmung wirkt je nach intrinsischem Zustand anders.
"""
from __future__ import annotations
import numpy as np


class Oscillator:
    """Ein einzelner Sinusoszillator mit Phase."""

    def __init__(self, frequency: float, phase: float | None = None):
        self.frequency = frequency
        self.phase = (np.random.uniform(0.0, 2 * np.pi)
                      if phase is None else float(phase))
        self.value: float = np.sin(self.phase)

    def step(self, dt: float = 1.0, phase_push: float = 0.0) -> float:
        """Einen Schritt; phase_push erlaubt Modulation durch andere Oszillatoren."""
        self.phase += 2.0 * np.pi * self.frequency * dt + phase_push
        self.value = float(np.sin(self.phase))
        return self.value


class TemporalDynamics:
    """
    Northoffs verschachtelte Zeitdynamik (temporo-spatial nestedness).

    Fünf Ebenen:
      0 – sehr langsam  (~Stimmungsbaseline, Minuten-Skala in Simulation)
      1 – langsam       (~Triebrhythmus)
      2 – mittel        (~Aufmerksamkeitsrhythmus)
      3 – schnell       (~Wahrnehmungstakt)
      4 – sehr schnell  (~Reaktions-Takt)

    Nesting: Ebene i+1 wird von Ebene i amplitudenmoduliert.
    Alignment: Ebene 0 koppelt an externen Umweltrhythmus (Kuramoto).
    """

    FREQUENCIES = [0.005, 0.02, 0.1, 0.8, 4.0]
    NESTING_STRENGTH = 0.15   # wie stark höhere Frequenzen moduliert werden
    COUPLING_STRENGTH = 0.08  # Kuramoto-Kopplungsstärke an Umwelt

    def __init__(self, k_ext_scale: float = 1.0):
        self.oscillators = [Oscillator(f) for f in self.FREQUENCIES]
        self._k_ext_scale = float(np.clip(k_ext_scale, 0.0, 1.0))
        # Kopplungsmatrix zwischen internen Oszillatoren
        # (langsame phasen-koppeln an nächst-schnellere)
        self._history: list[np.ndarray] = []

    def step(self, dt: float = 1.0, env_phase: float = 0.0) -> np.ndarray:
        """
        Einen Zeitschritt der intrinsischen Dynamik.

        env_phase: aktuelle Phase des Umweltrhythmus (z.B. Tageszeit 0–2π).
        k_ext_scale=0: zirkadianer Desynchronisation (Depression) —
          interne Rhythmen entkoppeln vom Tag/Nacht-Rhythmus.
        """
        vals = []

        # Ebene 0: sehr langsam, mit Kuramoto-Kopplung an Umwelt
        # Depression: k_ext_scale → 0 = Verlust der zirkadianen Synchronisation
        osc0 = self.oscillators[0]
        kuramoto_push = (self.COUPLING_STRENGTH * self._k_ext_scale
                         * np.sin(env_phase - osc0.phase) * dt)
        v0 = osc0.step(dt, phase_push=kuramoto_push)
        vals.append(v0)

        # Ebenen 1–4: jede wird von der nächst-langsameren moduliert (nestedness)
        for i in range(1, len(self.oscillators)):
            parent_val = vals[i - 1]
            # Amplitudenmodulation: schnellere Ebene bekommt phasenschub
            modulation = self.NESTING_STRENGTH * parent_val * dt
            v = self.oscillators[i].step(dt, phase_push=modulation)
            vals.append(v)

        state = np.array(vals)
        self._history.append(state.copy())
        return state

    # --- Auslese-Methoden (werden vom Agent genutzt) ---

    def precision(self) -> float:
        """
        Northoff: Die langsame Eigenaktivität moduliert die Präzision
        (Genauigkeit) der Interozeption.
        Dieselbe Körperwahrnehmung wirkt je nach intrinsischem Zustand anders.
        """
        slow = self.oscillators[0].value
        return max(0.3, 1.0 + 0.4 * slow)  # [0.6, 1.4]

    def arousal(self) -> float:
        """
        Schnelle Oszillatoren treiben das Arousal-Niveau.
        Hohes Arousal → mehr Exploration.
        """
        fast = self.oscillators[2].value  # mittlere Ebene
        return 0.5 + 0.3 * fast  # [0.2, 0.8]

    def mood_baseline(self) -> float:
        """
        Sehr langsamer Zustand als Stimmungs-Baseline.
        Moduliert Erwartungen und Priors des Agenten.
        """
        return self.oscillators[0].value  # [-1, 1]

    def get_state(self) -> np.ndarray:
        return np.array([o.value for o in self.oscillators])

    @property
    def history(self) -> list[np.ndarray]:
        return self._history
