"""
Echte Active Inference via pymdp (Friston's Free Energy Principle).

Navigation-POMDP für Toni:
  Zustand:      Position im Grid (size×size diskrete Positionen)
  Beobachtung:  Ressourcentyp auf aktuellem Feld (5 Kategorien)
  Aktionen:     hoch/runter/links/rechts/konsumieren

Northoff-Integration:
  Die Präzision des Wahrnehmungsmodells (A-Matrix) wird von der langsamen
  Eigenaktivität moduliert:
    - hohe Präzision  → scharfes A  → exploitieren (gezielte Navigation)
    - niedrige Präzision → flaches A → explorieren (Neugierde)

Solms-Integration:
  Der C-Vektor (logarithmische Präferenzen) wird von der Valenz /
  dem dominanten Bedürfnis des Körperzustands gesetzt.
  Das System hat keine hartkodierte Belohnung – die "Bedeutung"
  der Beobachtungen entsteht aus dem homöostatischen Fehler heraus.
"""
from __future__ import annotations
import numpy as np
from pymdp.agent import Agent as PyMDPAgent
from pymdp import utils

from .environment import EMPTY, FOOD, WATER, SHELTER

# Aktionsindex ↔ Name (muss mit DELTA-Reihenfolge übereinstimmen)
_ACTIONS = ["up", "down", "left", "right", "consume"]
_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

# Beobachtungskategorien: gleich wie Ressourcentypen (0–3) + 4 = "außerhalb"
N_OBS = 5
N_ACTIONS = 5


class NavigationAIF:
    """
    Active-Inference-Navigationsmodul basierend auf pymdp.

    Erstellt beim ersten Aufruf ein POMDP und aktualisiert A (Wahrnehmungs-
    modell) und C (Präferenzen) bei jedem Schritt aus dem aktuellen
    Körperzustand und der Northoff-Präzision.
    """

    def __init__(self, env_size: int, policy_len: int = 2):
        self.size = env_size
        self.n_states = env_size * env_size
        self.policy_len = policy_len

        self._B = self._build_B()
        self._aif_agent: PyMDPAgent | None = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  Öffentliche Methode                                                 #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        row: int, col: int,
        grid: np.ndarray,
        dominant_need: int,
        wellbeing: float,
        northoff_precision: float,
        memory_priors: dict[tuple[int, int], float] | None = None,
        future_urgency: np.ndarray | None = None,
        social_prior: np.ndarray | None = None,
        competition_factor: float = 1.0,
        affordances: dict | None = None,
        seeking_gain: float = 1.0,
    ) -> str:
        """
        Aktion über Expected Free Energy (EFE) auswählen.

        Northoff: precision moduliert Schärfe der A-Matrix.
        Solms:    dominant_need und wellbeing setzen den C-Vektor.
        Gedächtnis: memory_priors verschieben den Prior D.
        Enaktivismus: affordances boosten C relativ zu Körper-Umwelt-Kopplung.
        Depression: seeking_gain skaliert C-Vektor herunter (SEEKING-Kollaps).
        """
        pos_flat = row * self.size + col

        A = self._build_A(grid, northoff_precision)
        C = self._build_C(dominant_need, wellbeing, future_urgency,
                          competition_factor=competition_factor,
                          affordances=affordances,
                          seeking_gain=seeking_gain)
        D = self._build_D(memory_priors, social_prior)

        # pymdp-Agent bei jedem Schritt neu initialisieren.
        # policy_len=3 → Agent plant 3 Schritte voraus → findet Ressourcen auch
        # wenn sie nicht direkt benachbart sind.
        agent = PyMDPAgent(
            A=A, B=self._B, C=C, D=D,
            policy_len=self.policy_len,
            inference_algo="VANILLA",
            use_utility=True,
            use_states_info_gain=True,
        )

        # Belief auf bekannte Position setzen (wir kennen unsere Position exakt)
        agent.qs = utils.obj_array(1)
        agent.qs[0] = np.zeros(self.n_states)
        agent.qs[0][pos_flat] = 1.0

        # Aktuelle Beobachtung: Ressource auf diesem Feld
        obs_here = min(int(grid[row, col]), N_OBS - 1)
        agent.infer_states([obs_here])

        # Policy-Inferenz via EFE
        agent.infer_policies()

        # sample_action() gibt den ersten Schritt der besten Policy zurück:
        # Array der Form [action_idx] (ein Eintrag pro Kontrollfaktor)
        action_raw = agent.sample_action()
        action_idx = int(action_raw[0])
        return _ACTIONS[action_idx]

    # ------------------------------------------------------------------ #
    #  Matrixkonstruktion                                                  #
    # ------------------------------------------------------------------ #

    def _build_B(self) -> np.ndarray:
        """
        Transitionsmodell P(s' | s, a) – deterministisches Grid-Movement.
        Shape: (n_states, n_states, n_actions)
        """
        B_np = np.zeros((self.n_states, self.n_states, N_ACTIONS))
        for pos in range(self.n_states):
            r, c = divmod(pos, self.size)
            for a_idx, (dr, dc) in enumerate(_DELTAS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    next_pos = nr * self.size + nc
                else:
                    next_pos = pos  # Wand → bleibe stehen
                B_np[next_pos, pos, a_idx] = 1.0

        B = utils.obj_array(1)
        B[0] = B_np
        return B

    def _build_A(self, grid: np.ndarray, precision: float) -> np.ndarray:
        """
        Wahrnehmungsmodell P(o | s) – basierend auf aktuellem Grid.

        Northoff-Präzision moduliert die Schärfe:
          precision ≈ 1.4  → sehr scharfe Likelihood (Exploitation)
          precision ≈ 0.6  → flache Likelihood (Exploration)
        """
        # Basislikelihood: welche Ressource ist an welcher Position?
        A_np = np.ones((N_OBS, self.n_states)) * 0.02  # Grundrauschen

        for pos in range(self.n_states):
            r, c = divmod(pos, self.size)
            resource = min(int(grid[r, c]), N_OBS - 1)
            A_np[resource, pos] += precision * 10.0  # Northoff-Gewichtung

        # Spalten normieren → gültige Wahrscheinlichkeitsverteilung
        A_np /= A_np.sum(axis=0, keepdims=True)

        A = utils.obj_array(1)
        A[0] = A_np
        return A

    def _build_C(self, dominant_need: int, wellbeing: float,
                 future_urgency: np.ndarray | None = None,
                 competition_factor: float = 1.0,
                 affordances: dict | None = None,
                 seeking_gain: float = 1.0) -> np.ndarray:
        """
        Log-Präferenzen über Beobachtungen (Solms + Northoff + Sozial + Enaktivismus).

        Dringlichkeit kommt aus vier Quellen:
          current_urgency:    reaktiv (aus aktuellem Wohlbefinden)
          future_urgency:     antizipatorisch (aus projizierter Zukunft)
          competition_factor: sozial (Konkurrenz um knappe Ressource)
          affordances:        enaktivistisch (Körper-Umwelt-Kopplung)

        Affordanz-Boost: Ressourcen werden zusätzlich attraktiv, wenn sie
        sowohl nahbar als auch für den aktuellen Körperzustand relevant sind.
        Das bricht die strikte Dominanz eines einzelnen Bedürfnisses auf —
        enaktivistisch: nicht "eine Ressource oder die andere", sondern
        "was bietet die Umwelt gerade an, gegeben wer ich bin?"
        """
        # Reaktive Dringlichkeit aus aktuellem Wohlbefinden
        current_urgency = max(0.0, -wellbeing * 2.0)

        # Antizipatorische Dringlichkeit aus Zukunftsprojektion
        if future_urgency is not None and len(future_urgency) > dominant_need:
            future_boost = float(future_urgency[dominant_need]) * 3.0  # [0..3]
            urgency = max(current_urgency, future_boost)
        else:
            urgency = current_urgency

        # Soziale Dringlichkeit: Konkurrenz erhöht Motivation
        urgency *= competition_factor

        C_vec = np.zeros(N_OBS)
        # [0]=EMPTY [1]=FOOD [2]=WATER [3]=SHELTER [4]=außerhalb
        if dominant_need == 0:  # Energie
            C_vec[FOOD] = (3.0 + urgency * 1.5) * seeking_gain
            C_vec[WATER] = 0.5 * seeking_gain
        else:  # Hydration
            C_vec[WATER] = (3.0 + urgency * 1.5) * seeking_gain
            C_vec[FOOD] = 0.5 * seeking_gain
        # Leerfelder sind positiv bewertet (Bewegung führt zu Ressourcen)
        C_vec[EMPTY] = 0.3
        # Shelter nur attraktiv wenn explizit benötigt (Integrity niedrig)
        C_vec[SHELTER] = 0.05

        # Enaktivistischer Affordanz-Boost (Gibson: state-relative Bedeutung)
        if affordances is not None:
            C_vec[FOOD]    += affordances.get("food", 0.0) * 2.5
            C_vec[WATER]   += affordances.get("water", 0.0) * 2.5
            C_vec[SHELTER] += affordances.get("shelter", 0.0) * 1.5

        C = utils.obj_array(1)
        C[0] = C_vec
        return C

    def _build_D(
        self,
        memory_priors: dict[tuple[int, int], float] | None,
        social_prior: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Prior-Überzeugungen über Positionen.

        Autobiografisches Gedächtnis (Northoff+Damasio) erhöht Prior
        für Orte mit positiver Valenz-Geschichte.

        Soziales Selbst (Northoff Sozial-Achse) erhöht Prior
        für Orte wo andere Agenten Ressourcen gefunden haben.
        """
        D_vec = np.ones(self.n_states) / self.n_states

        if memory_priors:
            for (r, c), val in memory_priors.items():
                pos = r * self.size + c
                if 0 <= pos < self.n_states:
                    boost = max(0.0, val) * 5.0
                    D_vec[pos] += boost

        if social_prior is not None and len(social_prior) == self.n_states:
            D_vec += social_prior  # kann auch negativ sein (Abstoßung)
            D_vec = np.maximum(D_vec, 1e-6)  # keine negativen Priors

        D_vec /= D_vec.sum()

        D = utils.obj_array(1)
        D[0] = D_vec
        return D
