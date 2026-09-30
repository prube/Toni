"""
Stufe 1+2: Künstlicher Körper und Homöostase (Solms).

Der Körper verändert sich *auch ohne äußeren Input* – das ist entscheidend.
Zustände haben Sollwerte; Abweichungen konstituieren funktionale Bedürfnisse.
"""
from __future__ import annotations
import numpy as np

NEED_NAMES = ["energy", "hydration", "temperature", "integrity"]

# Homöostatische Sollwerte
SETPOINT = np.array([0.75, 0.75, 0.50, 1.00])

# Gewichte der Bedürfnisse (Integrität ist kritischer als Temperatur)
NEED_WEIGHTS = np.array([1.0, 1.0, 0.5, 2.0])


class Body:
    """
    Vier homöostatische Variablen nach dem Solms-Modell.

    Energie & Hydration sinken kontinuierlich.
    Temperatur driftet zum Umgebungswert.
    Integrität nimmt ab, wenn die Temperatur extrem ist.

    anhedonia_factor [0,1]: Konsum gibt weniger zurück (Depression).
    Solms: nicht "ich will nicht" — sondern "ich tue es, aber es hilft nicht mehr."
    """

    def __init__(self, energy: float = 0.8, hydration: float = 0.8,
                 temperature: float = 0.5, integrity: float = 1.0,
                 anhedonia_factor: float = 1.0):
        self.energy = energy
        self.hydration = hydration
        self.temperature = temperature
        self.integrity = integrity
        self.anhedonia_factor = float(np.clip(anhedonia_factor, 0.0, 1.0))

    def step(self, env_temp: float = 0.5, activity: float = 1.0) -> None:
        """Einen Zeitschritt simulieren."""
        # Metabolischer Verbrauch (Aktivität erhöht Verbrauch)
        self.energy -= 0.003 * activity
        self.hydration -= 0.005 * activity

        # Temperatur driftet zur Umgebungstemperatur (Wärmetausch)
        self.temperature += 0.02 * (env_temp - self.temperature)

        # Integrität leidet bei extremer Temperatur (Über- oder Unterkühlung)
        temp_stress = max(0.0, abs(self.temperature - 0.5) - 0.25)
        self.integrity -= 0.002 * temp_stress

        # Sehr niedrige Energie schadet der Integrität
        if self.energy < 0.1:
            self.integrity -= 0.001

        self._clip()

    def consume(self, resource_type: str) -> float:
        """Ressource aufnehmen; gibt die tatsächliche Aufnahme zurück."""
        if resource_type == "food":
            gain = min(0.35, 1.0 - self.energy) * self.anhedonia_factor
            self.energy += gain
            return gain
        if resource_type == "water":
            gain = min(0.35, 1.0 - self.hydration) * self.anhedonia_factor
            self.hydration += gain
            return gain
        if resource_type == "shelter":
            # Shelter normalisiert Temperatur und stärkt Integrität
            self.temperature += 0.1 * (0.5 - self.temperature)
            self.integrity = min(1.0, self.integrity + 0.005)
            return 0.005
        return 0.0

    def state(self) -> np.ndarray:
        """Aktueller Zustandsvektor [energy, hydration, temperature, integrity]."""
        return np.array([self.energy, self.hydration, self.temperature, self.integrity])

    def homeostatic_error(self) -> np.ndarray:
        """Abweichung vom Sollzustand (positiv = unterversorgt)."""
        return SETPOINT - self.state()

    def wellbeing(self) -> float:
        """
        Solms: wie gut/schlecht geht es dem System?
        Skala: 0 (perfekt) bis negativ (Not).
        """
        err = self.homeostatic_error()
        return -float(np.sum(NEED_WEIGHTS * err ** 2))

    def is_alive(self) -> bool:
        return self.energy > 0.0 and self.integrity > 0.0

    def _clip(self) -> None:
        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        self.hydration = float(np.clip(self.hydration, 0.0, 1.0))
        self.temperature = float(np.clip(self.temperature, 0.0, 1.0))
        self.integrity = float(np.clip(self.integrity, 0.0, 1.0))

    @property
    def mobility(self) -> float:
        """
        Operative Kapazität (Enaktivismus): Handlungsspielraum.

        sqrt(Energie × Hydration) × Integrität — kollabiert wenn eine der drei
        Dimensionen kollabiert. Nicht "wie weit vom Sollwert?" sondern
        "was kann dieser Körper noch tun?" (Maturana/Varela).
        """
        return float(
            np.sqrt(max(1e-6, self.energy) * max(1e-6, self.hydration))
            * max(1e-6, self.integrity)
        )

    def __repr__(self) -> str:
        return (f"Body(E={self.energy:.2f} H={self.hydration:.2f} "
                f"T={self.temperature:.2f} I={self.integrity:.2f})")
