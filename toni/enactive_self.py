"""
Enaktivistisches Selbst — Autopoiese, Affordanzen, Viabilität.

Philosophische Grundlage:
  Maturana/Varela (1980): Autopoiesis — der Organismus produziert und
  erhält die Bedingungen seiner eigenen Organisation. Das Ziel ist nicht
  Reward-Maximierung, sondern Aufrechterhaltung des Lebensprozesses.

  Thompson (2007): "Mind in Life" — Bewusstsein entsteht nicht durch
  interne Repräsentationen, sondern durch die Kopplung von Körper und Umwelt.

  Gibson (1979): Affordanzen — Umwelteigenschaften sind nicht neutral.
  Wasser "bietet" Trinken an — aber nur für ein durstiges Wesen. Bedeutung
  entsteht durch die Beziehung zwischen Körperzustand und Umwelt.

Enaktivistisches Kernprinzip:
  NICHT: "brain in a box" — internes Modell, das Welt repräsentiert
  SONDERN: Körperzustand ↔ Umwelt ↔ Wahrnehmung ↔ Handlung ↔ Bedeutung

Drei Konzepte für Toni:
  Mobilität:   Operative Kapazität (Handlungsspielraum)
               = sqrt(Energie × Hydration) × Integrität
  Viabilität:  Abstand vom Rand des lebensfähigen Raums
               = min(homöostatische Erfüllung) × Mobilität
  Affordanzen: Bedeutung von Ressourcen relativ zum Körperzustand
               = Defizit × Nähe (exponentiell abklingend)

Diese Signale fließen in den AIF C-Vektor (verändert Verhalten) und
den LLM-Kontext (artikuliert die enaktivistische Dimension).

Verhältnis zu Solms/Northoff:
  Solms:        Was fehlt jetzt? (homöostatischer Fehler, reaktiv)
  Northoff TTC: Wann fehlt es? (temporale Projektion, antizipatorisch)
  Enaktivismus: Kann ich noch handeln? (Viabilität, autopoietisch)
  Active Inf.:  Was ist wahrscheinlich/nützlich? (EFE-Minimierung)
"""
from __future__ import annotations

from collections import deque

import numpy as np


class EnactiveSelf:
    """
    Autopoiese und Affordanz-Wahrnehmung nach Maturana/Varela + Gibson.

    Ergänzt Northoffs temporale und soziale Positionierung um die
    fundamentalste Frage: Kann der Agent die Bedingungen seiner eigenen
    Organisation aufrechterhalten?

    Viabilität ≠ Wohlbefinden (Solms):
      Wohlbefinden: aktuelle Abweichung vom Sollzustand (reaktiv)
      Viabilität:   kann der Agent überhaupt noch handeln? (autopoietisch)
      Ein Agent mit gutem Wohlbefinden aber sinkender Mobilität verliert
      bereits seinen Handlungsspielraum — Autopoiese wird gefährdet.
    """

    def __init__(self) -> None:
        self._last_affordances: dict = {}
        self._viability_history: deque = deque(maxlen=20)

    # ------------------------------------------------------------------ #
    #  Hauptmethode                                                        #
    # ------------------------------------------------------------------ #

    def update(
        self,
        body_state: np.ndarray,
        env_grid: np.ndarray,
        row: int,
        col: int,
        grid_size: int,
        t: int,
    ) -> dict:
        """Affordanzen und Viabilität neu berechnen. Gibt AIF-Signal-Dict zurück."""
        affordances = self._compute_affordances(body_state, env_grid, row, col, grid_size)
        self._last_affordances = affordances
        self._viability_history.append(affordances["viability"])
        return affordances

    # ------------------------------------------------------------------ #
    #  Kernmetriken (statisch, auch direkt aufrufbar)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def mobility(body_state: np.ndarray) -> float:
        """
        Operative Kapazität: Wie viel Handlungsspielraum hat der Agent noch?

        Geometrisches Mittel aus Energie und Hydration, skaliert durch Integrität.
        Wenn Energie oder Hydration kollabiert → Mobilität kollabiert.
        Wenn Integrität kollabiert → Mobilität kollabiert.

        Enaktivistisch: nicht "wie weit ist er vom Sollzustand entfernt?",
        sondern "was kann er überhaupt noch tun?"
        """
        e, h, _t, i = body_state
        return float(np.sqrt(max(1e-6, e) * max(1e-6, h)) * max(1e-6, i))

    @staticmethod
    def viability(body_state: np.ndarray) -> float:
        """
        Lebensfähigkeitsmaß: Wie weit ist der Agent vom Rand des vitalen Raums?

        [0.0 = nicht lebensfähig (Autopoiese kollabiert)]
        [1.0 = vollständig lebensfähig (alle Ressourcen optimal)]

        Kombiniert homöostatische Erfüllung + Mobilität:
          homeostasis = min(e/0.75, h/0.75, i)   ← wie gut erfüllt?
          viability   = homeostasis × (0.5 + 0.5×mob)  ← kann ich es halten?

        Ein Agent kann aktuell "über Wasser" sein, aber wenn seine Mobilität
        bereits sinkt, sinkt die Viabilität schneller als das Wohlbefinden.
        """
        e, h, _t, i = body_state
        mob = EnactiveSelf.mobility(body_state)
        homeostasis = min(e / 0.75, h / 0.75, i)
        return float(np.clip(homeostasis * (0.5 + 0.5 * mob), 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Affordanz-Berechnung                                                #
    # ------------------------------------------------------------------ #

    def _compute_affordances(
        self,
        body_state: np.ndarray,
        env_grid: np.ndarray,
        row: int,
        col: int,
        grid_size: int,
    ) -> dict:
        """
        Affordanzen: Was bietet die Umwelt gerade an — relativ zu diesem Körper?

        Gibson (1979): Affordanzen existieren nicht in der Umwelt allein,
        nicht im Organismus allein — sie entstehen in der Kopplung.

        Formel: Affordanz = Defizit × Nähe
          Defizit:  wie dringend ist das Bedürfnis?
          Nähe:     wie zugänglich ist die Ressource? (exp(-d/4))
          Produkt:  BEIDE müssen groß sein für hohe Affordanz

        Beispiel:
          Hydration=0.3 (Defizit=0.45), Wasser 2 Felder weg:
            water_afford = 0.45 × exp(-2/4) = 0.45 × 0.61 = 0.27  [mittel]
          Hydration=0.7 (Defizit=0.05), Wasser 2 Felder weg:
            water_afford = 0.05 × exp(-2/4) = 0.05 × 0.61 = 0.03  [niedrig]
        """
        e, h, t_body, i = body_state
        mob = self.mobility(body_state)

        # Defizite (Abweichung unterhalb Sollwert)
        e_deficit  = max(0.0, 0.75 - e)
        h_deficit  = max(0.0, 0.75 - h)
        t_stress   = max(0.0, abs(t_body - 0.5) - 0.1)  # Toleranzzone ±0.1
        i_deficit  = max(0.0, 1.0 - i)

        # Ressourcendistanzen
        from .environment import FOOD, WATER, SHELTER
        food_dist    = self._nearest_dist(env_grid, row, col, FOOD, grid_size)
        water_dist   = self._nearest_dist(env_grid, row, col, WATER, grid_size)
        shelter_dist = self._nearest_dist(env_grid, row, col, SHELTER, grid_size)

        def prox(d: int) -> float:
            return float(np.exp(-d / 4.0))

        return {
            "food":    float(e_deficit * prox(food_dist)),
            "water":   float(h_deficit * prox(water_dist)),
            "shelter": float((t_stress + i_deficit * 0.5) * prox(shelter_dist)),
            "explore": float(mob * 0.3),
            "mobility":  mob,
            "viability": self.viability(body_state),
            "_distances": {
                "food": food_dist, "water": water_dist, "shelter": shelter_dist,
            },
        }

    @staticmethod
    def _nearest_dist(
        grid: np.ndarray, row: int, col: int,
        resource_type: int, grid_size: int,
    ) -> int:
        min_d = grid_size * 2
        for r in range(grid_size):
            for c in range(grid_size):
                if int(grid[r, c]) == resource_type:
                    d = abs(r - row) + abs(c - col)
                    if d < min_d:
                        min_d = d
        return min_d

    # ------------------------------------------------------------------ #
    #  Signale für AIF und LLM                                            #
    # ------------------------------------------------------------------ #

    def aif_signal(self) -> dict[str, float]:
        """Affordanz-Stärken für den AIF C-Vektor (nur resource-Typen)."""
        return {
            "food":    float(self._last_affordances.get("food", 0.0)),
            "water":   float(self._last_affordances.get("water", 0.0)),
            "shelter": float(self._last_affordances.get("shelter", 0.0)),
        }

    def enactive_summary(self) -> dict:
        """Zusammenfassung für LLM-Kortex und Visualisierung."""
        a = self._last_affordances
        if not a:
            return {}

        dists = a.get("_distances", {})

        def interpret(val: float, dist: int) -> str:
            if val < 0.03:
                return "niedrig — kein Bedarf oder zu weit"
            if val < 0.15:
                return f"gering (Distanz ~{dist})"
            if val < 0.35:
                return f"mittel (Distanz ~{dist}, Bedarf wächst)"
            return f"hoch (Distanz ~{dist}, dringend relevant)"

        dominant = max(
            [("nahrung", a.get("food", 0.0)),
             ("wasser",  a.get("water", 0.0)),
             ("schutz",  a.get("shelter", 0.0))],
            key=lambda x: x[1],
        )[0]

        vib_trend = "stabil"
        if len(self._viability_history) >= 5:
            hist = list(self._viability_history)[-5:]
            slope = (hist[-1] - hist[0]) / 4.0
            if slope < -0.01:
                vib_trend = "sinkend ⚠"
            elif slope > 0.01:
                vib_trend = "steigend"

        mob = a.get("mobility", 0.0)
        vib = a.get("viability", 0.0)

        return {
            "enaktivismus": "Maturana/Varela Autopoiese + Gibson Affordanzen",
            "viabilität": round(vib, 3),
            "viabilität_trend": vib_trend,
            "mobilität": round(mob, 3),
            "affordanzen": {
                "nahrung": interpret(a.get("food", 0.0), dists.get("food", 0)),
                "wasser":  interpret(a.get("water", 0.0), dists.get("water", 0)),
                "schutz":  interpret(a.get("shelter", 0.0), dists.get("shelter", 0)),
                "exploration": (
                    "möglich" if mob > 0.3 else "eingeschränkt"
                ) + f" (Mobilität={mob:.2f})",
            },
            "bedeutsamste_affordanz": dominant,
            "autopoiese": (
                "aktiv"
                if vib > 0.4
                else "⚠ gefährdet — Handlungsspielraum schwindet"
            ),
        }
