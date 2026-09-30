"""
Stufe 4: Interozeption – verrauschte Wahrnehmung des eigenen Körperzustands.

Der Agent darf seinen Körperzustand NICHT perfekt kennen.
Er muss inferieren: "Wie geht es meinem Körper wahrscheinlich?"
Die Präzision dieser Wahrnehmung wird von der langsamen Eigenaktivität
(Northoff) moduliert – gleiche Reize wirken je nach intrinsischem Zustand anders.
"""
from __future__ import annotations
import numpy as np
from .body import Body

BASE_NOISE_STD = 0.04  # Grundrauschen der Interozeption


def interoceptive_sample(body: Body, precision: float = 1.0) -> np.ndarray:
    """
    Verrauschte Stichprobe des Körperzustands.

    precision: von TemporalDynamics.precision() – moduliert Rauschstärke.
    Höhere Präzision → weniger Rauschen, genauere Selbstwahrnehmung.
    """
    noise_std = BASE_NOISE_STD / max(0.1, precision)
    noise = np.random.normal(0.0, noise_std, size=4)
    return np.clip(body.state() + noise, 0.0, 1.0)


def bayesian_update(prior: np.ndarray, observation: np.ndarray,
                    precision: float = 1.0,
                    prior_weight: float = 0.7) -> np.ndarray:
    """
    Vereinfachter Bayesscher Update: gewichteter Mittelwert aus Prior und
    aktuellem interozeptivem Signal.

    Entspricht dem Active-Inference-Prinzip der Belief-Propagation:
    Neue Schätzung = λ·Prior + (1-λ)·Beobachtung
    wobei λ von der Präzision abhängt (höhere Präzision → mehr Gewicht auf Daten).
    """
    obs_weight = min(0.9, (1.0 - prior_weight) * precision)
    p = prior_weight - obs_weight * 0.3
    p = max(0.1, min(0.9, p))
    return p * prior + (1.0 - p) * observation
