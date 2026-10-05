"""
Toni – der integrierte Agent.

Verbindet alle Module nach der Northoff+Solms-Architektur:

  Umwelt → Körper → Interozeption (verrauscht, präzisionsmoduliert)
         → Valenz (Δwellbeing = "gut/schlecht für mich")
         → Aufmerksamkeit (Solms: dringendstes Bedürfnis)
         → Intrinsische Zeitdynamik (Northoff: Eigenaktivität, nestedness)
         → Temporo-spatial alignment (Kuramoto an Tagesrhythmus)
         → Autobiografisches Gedächtnis (Vergangenheit formt Gegenwart)
         → Active Inference via pymdp (EFE-Minimierung, Northoff-moduliert)
         → Aktion → Umwelt (geschlossener Regelkreis)

Das LLM kommt – falls gewünscht – ganz oben drauf, nicht als Kern.

Parameter use_pymdp:  True  → echte Active Inference via pymdp
                      False → heuristische Fallback-Policy (schneller)
Parameter use_northoff: True  → verschachtelte Zeitdynamik aktiv
                         False → flache Präzision=1.0 (für Experiment)
"""
from __future__ import annotations
import numpy as np
from .body import Body, SETPOINT, NEED_NAMES, NEED_WEIGHTS
from .oscillators import TemporalDynamics
from .interoception import interoceptive_sample, bayesian_update
from .memory import AutobiographicalMemory
from .environment import GridWorld, FOOD, WATER, SHELTER, EMPTY
from .temporal_self import TemporalSelf
from .social_self import SocialSelf, SocialObservation
from .enactive_self import EnactiveSelf

ACTIONS = ["up", "down", "left", "right", "consume", "rest"]
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class Toni:
    """
    Toni: KI-Agent nach Northoff + Solms.

    Kein LLM-Kern. Kein externer Reward. Kein Programmierer, der sagt
    'battery < 10% → go_to_charger'. Stattdessen:
    - Interne Zustände verändern sich kontinuierlich (auch ohne Prompt).
    - Abweichungen vom Sollzustand erzeugen Bedürfnisse und Valenz.
    - Die intrinsische Zeitdynamik moduliert Wahrnehmung und Entscheidung.
    - Vergangenheit wirkt auf Gegenwart durch autobiografisches Gedächtnis.
    - Active Inference (pymdp) minimiert Expected Free Energy.
    """

    def __init__(self, env: GridWorld,
                 start_pos: tuple[int, int] | None = None,
                 use_pymdp: bool = True,
                 use_northoff: bool = True,
                 use_temporal_self: bool = True,
                 agent_id: str = "toni",
                 depression_level: float = 0.0,
                 northoff_depression_level: float = 0.0,
                 borderline_level: float = 0.0,
                 policy_len: int = 2):
        self.env = env
        self.depression_level = float(np.clip(depression_level, 0.0, 1.0))
        d = self.depression_level

        # Solms-Mechanismen (SEEKING-Kollaps, Anhedonie, Rumination)
        anhedonia    = 1.0 - 0.8 * d   # Konsum-Gain reduziert
        k_ext_scale  = 1.0 - 0.95 * d  # zirkadianer Desynchronisation
        horizon      = max(5, int(80 * (1.0 - 0.85 * d)))  # Zeithorizont kollabiert
        self._seeking_gain   = 1.0 - 0.7 * d   # SEEKING-Antrieb gedämpft
        self._rumination     = 1.0 + 4.0 * d   # negative Erinnerungen verstärkt

        # Northoff-spezifische Mechanismen (Rest-Self-Overlap, temporale Stasis)
        nd = float(np.clip(northoff_depression_level, 0.0, 1.0))
        self._env_coupling  = 1.0 - 0.9 * nd   # Affordanz-Blindheit (0.1 bei nd=1)
        past_bias_val       = nd * 3.0           # Vergangenheit dominiert ESN (0→3)

        # Borderline-spezifische Mechanismen (Valenz-Dysregulation, temporaler Kollaps)
        # Regulationsstörung: kein stabiler Setpoint, Selbst-Welt-Grenze instabil.
        # Drei Mechanismen:
        #   1. Valenz-Rauschen: Körpersignal schwankt unvorhersehbar → unstabiles C
        #   2. Temporaler Kollaps: bei Stress bricht Zukunftsplanung zusammen (Impulsivität)
        #   3. Soziale Hypersensitivität: andere Agenten dominieren den Prior
        bl = float(np.clip(borderline_level, 0.0, 1.0))
        self._bl_noise              = 0.15 * bl   # Valenz-Rauschen: 0 normal → 0.15 bei bl=1
        self._bl_collapse_threshold = 1.0 - 0.7 * bl  # Kollaps ab: 1.0 (nie) → 0.30 (bl=1)
        self._social_weight         = 1.0 + 4.0 * bl  # Soziales Gewicht: 1× → 5×

        self.body = Body(anhedonia_factor=anhedonia)
        self.temporal = TemporalDynamics(k_ext_scale=k_ext_scale)
        self.memory = AutobiographicalMemory()
        self.temporal_self = TemporalSelf(horizon=horizon, past_bias=past_bias_val)
        self.social_self = SocialSelf()
        self.enactive_self = EnactiveSelf()
        self.use_northoff = use_northoff
        self.use_temporal_self = use_temporal_self
        self.agent_id = agent_id
        self._aif: object | None = None
        if use_pymdp:
            from .active_inference import NavigationAIF
            self._aif = NavigationAIF(env_size=env.size, policy_len=policy_len)

        # Startposition
        mid = env.size // 2
        self.row, self.col = start_pos if start_pos else (mid, mid)

        # Zustandsschätzung (wird per Bayesscher Update aktualisiert)
        self._intero_estimate: np.ndarray = self.body.state().copy()

        # Zustandsvariablen für aktuellen Schritt
        self.valence: float = 0.0
        self.wellbeing: float = self.body.wellbeing()
        self.dominant_need: int = 0
        self.temporal_state: np.ndarray = self.temporal.get_state()

        # Zeitschrittzähler
        self.t: int = 0

        # Verlaufsaufzeichnung für Visualisierung
        self.history: dict[str, list] = {
            "wellbeing": [], "valence": [], "energy": [], "hydration": [],
            "temperature": [], "integrity": [], "dominant_need": [],
            "oscillators": [], "position": [], "action": [],
            "precision": [], "arousal": [], "future_urgency": [],
            "viability": [], "mobility": [],
        }

    # ------------------------------------------------------------------ #
    #  Hauptschleife                                                       #
    # ------------------------------------------------------------------ #

    def step(self) -> dict:
        """Einen vollständigen Zeitschritt ausführen."""

        # ── 1. Umwelt tickt (Tag/Nacht, Ressourcen-Respawn) ──────────────
        self.env.step()
        env_temp = self.env.environment_temperature()
        env_phase = self.env.day_phase()

        # ── 2. Körper verändert sich (auch ohne äußeren Input) ────────────
        old_wellbeing = self.body.wellbeing()
        activity = 1.0 + 0.2 * self.temporal.arousal()  # Arousal → mehr Verbrauch
        self.body.step(env_temp=env_temp, activity=activity)

        # ── 3. Northoff: intrinsische Zeitdynamik ─────────────────────────
        self.temporal_state = self.temporal.step(dt=1.0, env_phase=env_phase)
        # use_northoff=False → flache Präzision=1.0 (Kontrollbedingung)
        precision = self.temporal.precision() if self.use_northoff else 1.0
        arousal = self.temporal.arousal() if self.use_northoff else 0.5

        # ── 4. Interozeption: verrauschte, präzisionsmodulierte Wahrnehmung
        raw_intero = interoceptive_sample(self.body, precision)
        self._intero_estimate = bayesian_update(
            self._intero_estimate, raw_intero, precision
        )

        # ── 5. Homöostatischer Fehler aus Schätzung (nicht Ground Truth!) ─
        error = SETPOINT - self._intero_estimate

        # ── 6. Aufmerksamkeit: Solms – dringendstes Bedürfnis ─────────────
        # Nur Defizite (unterhalb Sollwert) zählen als echte Bedürfnisse.
        # Überschüsse (über Sollwert) werden ignoriert – übervolle Hydration
        # ist kein Bedürfnis, auch wenn der Abstand vom Sollwert groß ist.
        deficits = np.maximum(0.0, error[:2])  # nur Shortfalls für E/H
        need_urgency = NEED_WEIGHTS[:2] * deficits
        if need_urgency.sum() > 0.0:
            self.dominant_need = int(np.argmax(need_urgency))
        else:
            # Kein akutes Defizit → vorausschauend: welches Bedürfnis wächst am schnellsten?
            self.dominant_need = int(np.argmin(self._intero_estimate[:2]))

        # ── 7. Valenz: Δwellbeing = "gut oder schlecht für dieses System" ─
        new_wellbeing = self.body.wellbeing()
        self.valence = new_wellbeing - old_wellbeing
        self.wellbeing = new_wellbeing

        # ── 7b. Temporales Selbst: drei Zeitfenster (Northoff) ───────────
        # Vergangenheit: Verlauf aufzeichnen
        # Zukunft: Projektion → antizipatorische Dringlichkeit für AIF
        self.temporal_self.update(
            body_state=self.body.state(),
            wellbeing=self.wellbeing,
            valence=self.valence,
            position=(self.row, self.col),
            t=self.t,
        )

        # ── 7c. Enaktivistisches Selbst: Affordanzen + Viabilität ────────
        # Maturana/Varela: Kann ich die Bedingungen meiner Organisation halten?
        # Gibson: Was bietet die Umwelt gerade an — gegeben wer ich bin?
        self.enactive_self.update(
            body_state=self.body.state(),
            env_grid=self.env.grid,
            row=self.row, col=self.col,
            grid_size=self.env.size,
            t=self.t,
        )

        # ── 8. Active-Inference-Policy (vereinfacht) ──────────────────────
        action = self._select_action(error, arousal)

        # ── 9. Aktion ausführen ───────────────────────────────────────────
        self._execute_action(action)

        # ── 10. Autobiografisches Gedächtnis ──────────────────────────────
        self.memory.store(
            body_state=self.body.state(),
            valence=self.valence,
            wellbeing=self.wellbeing,
            position=(self.row, self.col),
            action=action,
            dominant_need=self.dominant_need,
            temporal_state=self.temporal_state,
            rumination_factor=self._rumination,
        )

        # ── 11. Verlauf aufzeichnen ───────────────────────────────────────
        self._record(action, precision, arousal)
        self.t += 1

        return {
            "t": self.t,
            "wellbeing": self.wellbeing,
            "valence": self.valence,
            "dominant_need": NEED_NAMES[self.dominant_need],
            "position": (self.row, self.col),
            "action": action,
            "temporal_state": self.temporal_state,
            "body": self.body.state(),
            "alive": self.body.is_alive(),
        }

    # ------------------------------------------------------------------ #
    #  Policy: vereinfachte Expected-Free-Energy-Minimierung              #
    # ------------------------------------------------------------------ #

    def _select_action(self, error: np.ndarray, arousal: float) -> str:
        """
        Policy: Active Inference via pymdp (wenn verfügbar) oder Fallback.

        pymdp wählt Aktion via Expected Free Energy (EFE):
          EFE = erwarteter Überraschung + Informationsgewinn (Epistemic value)
        C (Präferenzen) kommen von Solms (dominantes Bedürfnis).
        A (Präzision) kommt von Northoff (Zeitdynamik).
        D (Prior) kommt vom autobiografischen Gedächtnis.

        PANIK-OVERRIDE (Northoff: Gegenwart schlägt Zukunft bei existentieller Dringlichkeit)
        Wenn der Agent bereits auf der benötigten Ressource steht und die Krise
        unmittelbar ist (future_urgency > 0.9), wird sofort konsumiert — ohne
        EFE-Planung. Verhindert Paralyse durch Überoptimierung: der D-Prior
        auf ein "besseres" Wasserfeld weiter weg darf nicht das Trinken des
        aktuellen Wassers blockieren.
        """
        # PANIK-OVERRIDE: aktuelle Krise + Ressource direkt verfügbar → sofort konsumieren
        # Reagiert auf IST-Zustand, nicht auf Projektion — verhindert dass D-Prior auf
        # ein "besseres" Feld den Agenten von der Ressource direkt unter ihm wegnavigiert.
        current_crisis = (
            self.body.energy < 0.25 or
            self.body.hydration < 0.25
        )
        if current_crisis:
            resource = self.env.get_at(self.row, self.col)
            if self._should_consume(resource):
                return "consume"

        if self._aif is not None:
            # Gedächtniskarte als Positionsprior
            mem_priors = {
                pos: self.memory.location_prior(pos)
                for pos in self.memory._location_valence_cache
            }
            # Sozialer Positionsprior (anderen Agenten beobachtet)
            social_prior = (
                self.social_self.social_position_prior(self.env.size)
                if self.social_self.agent_count > 0 else None
            )
            # Konkurrenz-Faktor
            competition = (
                self.social_self.competition_factor(
                    (self.row, self.col), self.dominant_need, self.env.size
                )
                if self.social_self.agent_count > 0 else 1.0
            )

            # ── Borderline-Modifikationen ────────────────────────────────
            # ① Valenz-Rauschen: emotionale Dysregulation
            #    Der Agent "weiß nicht ob er will" — Körpersignal fluktuiert,
            #    C-Vektor ist von Schritt zu Schritt instabil.
            _wellbeing = self.wellbeing
            if self._bl_noise > 0.0:
                _wellbeing = float(np.clip(
                    self.wellbeing + np.random.normal(0.0, self._bl_noise),
                    -2.0, 1.0
                ))

            # ② Temporaler Kollaps bei Stress: wenn Urgency die Schwelle
            #    überschreitet, bricht Zukunftsplanung zusammen → Impulsivität.
            #    (Future_urgency=None schaltet antizipatorischen Modus aus)
            _fu = (self.temporal_self.future_urgency()
                   if self.use_temporal_self else None)
            if _fu is not None and self._bl_collapse_threshold < 1.0:
                if float(np.max(_fu[:2])) > self._bl_collapse_threshold:
                    _fu = None  # Zukunft kollabiert → pure Gegenwart

            # ③ Soziale Hypersensitivität: Gewicht des sozialen Priors erhöhen
            _social = social_prior
            if _social is not None and self._social_weight > 1.0:
                _social = _social * self._social_weight
            # ─────────────────────────────────────────────────────────────

            try:
                action = self._aif.select_action(
                    row=self.row, col=self.col,
                    grid=self.env.grid,
                    dominant_need=self.dominant_need,
                    wellbeing=_wellbeing,
                    northoff_precision=(
                        self.temporal.precision() if self.use_northoff else 1.0
                    ),
                    memory_priors=mem_priors if mem_priors else None,
                    future_urgency=_fu,
                    social_prior=_social,
                    competition_factor=competition,
                    affordances=self.enactive_self.aif_signal(),
                    seeking_gain=self._seeking_gain,
                    env_coupling_scale=self._env_coupling,
                )
                # Post-Processing: "consume" nur wenn passende Ressource vorhanden
                if action == "consume":
                    resource = self.env.get_at(self.row, self.col)
                    if not self._should_consume(resource):
                        # Keine passende Ressource → Heuristik für Navigation
                        action = self._heuristic_action(error, arousal)
                return action
            except Exception:
                pass  # bei pymdp-Fehler → Fallback

        # Fallback: heuristische Policy (original)
        return self._heuristic_action(error, arousal)

    def _heuristic_action(self, error: np.ndarray, arousal: float) -> str:
        """Heuristische Fallback-Policy (schnell, ohne pymdp)."""
        resource = self.env.get_at(self.row, self.col)
        if self._should_consume(resource):
            return "consume"
        if resource == SHELTER and self.body.integrity < 0.5:
            return "consume"

        target_resource = FOOD if self.dominant_need == 0 else WATER
        target = self._find_nearest(target_resource)

        explore_prob = 0.1 + 0.3 * max(0.0, arousal - 0.4)
        if np.random.random() < explore_prob:
            return self._memory_guided_action()

        return self._navigate_toward(target) if target else self._memory_guided_action()

    def _should_consume(self, resource: int) -> bool:
        if resource == FOOD and self.dominant_need == 0 and self.body.energy < 0.9:
            return True
        if resource == WATER and self.dominant_need == 1 and self.body.hydration < 0.9:
            return True
        if resource == FOOD and self.body.energy < 0.5:
            return True
        if resource == WATER and self.body.hydration < 0.5:
            return True
        return False

    def _find_nearest(self, resource_type: int) -> tuple[int, int] | None:
        """Nächste sichtbare Ressource finden (Manhattan-Distanz)."""
        positions = self.env.resource_positions(resource_type)
        if not positions:
            return None
        dists = [(abs(r - self.row) + abs(c - self.col), (r, c))
                 for r, c in positions]
        dists.sort()
        return dists[0][1]

    def _navigate_toward(self, target: tuple[int, int]) -> str:
        """Einen Schritt Richtung Ziel (greedy Manhattan)."""
        tr, tc = target
        dr = tr - self.row
        dc = tc - self.col

        # Wähle Richtung mit größerem Fehler, tie-break zufällig
        if abs(dr) == abs(dc):
            return np.random.choice(
                [("down" if dr > 0 else "up"), ("right" if dc > 0 else "left")]
            )
        if abs(dr) > abs(dc):
            return "down" if dr > 0 else "up"
        return "right" if dc > 0 else "left"

    def _memory_guided_action(self) -> str:
        """
        Autobiografisches Gedächtnis beeinflusst Entscheidung:
        Bevorzuge Nachbarfelder mit positivem Valenz-Prior.
        """
        candidates = {}
        for action, (dr, dc) in DELTAS.items():
            nr, nc = self.row + dr, self.col + dc
            if self.env.is_valid(nr, nc):
                prior = self.memory.location_prior((nr, nc))
                # Kleines Rauschen für Exploration
                candidates[action] = prior + np.random.normal(0, 0.01)

        if not candidates:
            return np.random.choice(ACTIONS[:4])

        return max(candidates, key=candidates.get)

    # ------------------------------------------------------------------ #
    #  Aktionsausführung                                                   #
    # ------------------------------------------------------------------ #

    def _execute_action(self, action: str) -> None:
        self._last_consumed: int | None = None
        if action == "consume":
            resource = self.env.consume_at(self.row, self.col)
            if resource == FOOD:
                self.body.consume("food")
                self._last_consumed = FOOD
            elif resource == WATER:
                self.body.consume("water")
                self._last_consumed = WATER
            elif resource == SHELTER:
                self.body.consume("shelter")
                self._last_consumed = SHELTER
        elif action == "rest":
            self.body.integrity = min(1.0, self.body.integrity + 0.005)
        elif action in DELTAS:
            dr, dc = DELTAS[action]
            nr, nc = self.row + dr, self.col + dc
            if self.env.is_valid(nr, nc):
                self.row, self.col = nr, nc

    # ------------------------------------------------------------------ #
    #  Soziale Schnittstelle                                               #
    # ------------------------------------------------------------------ #

    def social_observation(self) -> SocialObservation:
        """Erzeugt eine beobachtbare Snapshot dieses Agenten für andere."""
        last_action = (self.history["action"][-1]
                       if self.history["action"] else "none")
        return SocialObservation(
            agent_id=self.agent_id,
            position=(self.row, self.col),
            last_action=last_action,
            resource_consumed=getattr(self, "_last_consumed", None),
            dominant_need_type=self.dominant_need,
            t=self.t,
        )

    def receive_social_observations(
        self, observations: list[SocialObservation]
    ) -> None:
        """Soziale Beobachtungen anderer Agenten verarbeiten."""
        self.social_self.update(observations)

    # ------------------------------------------------------------------ #
    #  Hilfsmethoden                                                       #
    # ------------------------------------------------------------------ #

    def _record(self, action: str, precision: float, arousal: float) -> None:
        h = self.history
        h["wellbeing"].append(self.wellbeing)
        h["valence"].append(self.valence)
        h["energy"].append(self.body.energy)
        h["hydration"].append(self.body.hydration)
        h["temperature"].append(self.body.temperature)
        h["integrity"].append(self.body.integrity)
        h["dominant_need"].append(self.dominant_need)
        h["oscillators"].append(self.temporal_state.copy())
        h["position"].append((self.row, self.col))
        h["action"].append(action)
        h["precision"].append(precision)
        h["arousal"].append(arousal)
        h["future_urgency"].append(self.temporal_self.future_urgency()[:2].copy())
        ea = self.enactive_self._last_affordances
        h["viability"].append(ea.get("viability", 0.0))
        h["mobility"].append(ea.get("mobility", 0.0))

    @property
    def status(self) -> str:
        need = NEED_NAMES[self.dominant_need]
        urgency = self.temporal_self.future_urgency()
        max_fut = float(max(urgency[:2]))
        fut_str = f" | fut={max_fut:.2f}↑" if max_fut > 0.3 else ""
        return (f"t={self.t:4d} | {self.body} | "
                f"valence={self.valence:+.3f} | need={need}{fut_str} | "
                f"pos=({self.row},{self.col})")
