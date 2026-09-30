"""
Stufe 1 (Umwelt): GridWorld mit Ressourcen und externem Zeitrhythmus.

Die Umwelt hat eigene Rhythmen (Tag/Nacht, Temperatur, Ressourcenverfügbarkeit).
Diese liefern die externe Phase für Northoffs temporo-spatial alignment.
"""
from __future__ import annotations
import numpy as np


# Ressourcentypen
EMPTY = 0
FOOD = 1
WATER = 2
SHELTER = 3

RESOURCE_CHARS = {EMPTY: ".", FOOD: "F", WATER: "W", SHELTER: "S"}
RESOURCE_NAMES = {EMPTY: "empty", FOOD: "food", WATER: "water", SHELTER: "shelter"}


class GridWorld:
    """
    Diskrete 2D-Welt mit Nahrung, Wasser, Schutz und Tagesrhythmus.

    Der Tagesrhythmus (day_phase) ist der externe Zeitgeber für
    Northoffs temporo-spatial alignment.
    """

    DAY_LENGTH = 200  # Schritte pro Tag

    def __init__(self, size: int = 9, seed: int | None = None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.time: int = 0

        self.grid = np.zeros((size, size), dtype=int)
        self._food_respawn_timer: dict[tuple[int, int], int] = {}
        self._initial_food: list[tuple[int, int]] = []
        self._initial_water: list[tuple[int, int]] = []

        self._place_resources()

    def _place_resources(self) -> None:
        s = self.size
        # Nahrungsquellen (verteilt über die Welt)
        food_pos = [
            (1, 1), (1, s - 2),
            (s // 2, 2), (s // 2, s - 3),
            (s - 2, 1), (s - 2, s - 2),
            (s // 3, s // 2),
        ]
        # Wasserquellen (weniger, zentraler)
        water_pos = [
            (0, s // 2),
            (s - 1, s // 2),
            (s // 2, 0),
            (s // 2, s - 1),
        ]
        # Schutz/Shelter
        shelter_pos = [(s // 2, s // 2)]

        for r, c in food_pos:
            self.grid[r, c] = FOOD
            self._initial_food.append((r, c))
        for r, c in water_pos:
            self.grid[r, c] = WATER
            self._initial_water.append((r, c))
        for r, c in shelter_pos:
            self.grid[r, c] = SHELTER

    def step(self) -> None:
        self.time += 1
        self._respawn_resources()

    def _respawn_resources(self) -> None:
        """Aufgebrauchte Nahrung respawnt nach einer Weile."""
        for pos, timer in list(self._food_respawn_timer.items()):
            if self.time >= timer:
                r, c = pos
                if self.grid[r, c] == EMPTY:
                    self.grid[r, c] = FOOD
                del self._food_respawn_timer[pos]

    # --- Phasensignale für Northoff-Alignment ---

    def day_phase(self) -> float:
        """Externe Tagesphase [0, 2π] – Zeitgeber für Northoffs Alignment."""
        return 2.0 * np.pi * (self.time % self.DAY_LENGTH) / self.DAY_LENGTH

    def environment_temperature(self) -> float:
        """
        Umgebungstemperatur schwankt mit dem Tagesrhythmus.
        Mittags (phase ≈ π) am wärmsten.
        """
        return 0.5 + 0.25 * np.sin(self.day_phase())

    def light_level(self) -> float:
        """Helligkeit 0..1 (Tag/Nacht)."""
        return 0.5 + 0.5 * np.sin(self.day_phase())

    # --- Interaktion ---

    def get_at(self, row: int, col: int) -> int:
        if 0 <= row < self.size and 0 <= col < self.size:
            return int(self.grid[row, col])
        return -1  # außerhalb

    def consume_at(self, row: int, col: int) -> int:
        """Ressource aufnehmen; FOOD-Tiles verschwinden und respawnen später."""
        resource = self.grid[row, col]
        if resource == FOOD:
            self.grid[row, col] = EMPTY
            self._food_respawn_timer[(row, col)] = self.time + 80
        # Wasser und Shelter bleiben dauerhaft
        return int(resource)

    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def resource_positions(self, resource_type: int) -> list[tuple[int, int]]:
        positions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == resource_type:
                    positions.append((r, c))
        return positions

    def render(self, agent_pos: tuple[int, int] | None = None) -> str:
        lines = []
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if agent_pos and (r, c) == agent_pos:
                    row_str += "A "
                else:
                    row_str += RESOURCE_CHARS[self.grid[r, c]] + " "
            lines.append(row_str.rstrip())
        return "\n".join(lines)
