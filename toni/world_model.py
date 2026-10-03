"""
Neuronales Weltmodell — Echo State Network (ESN).

Ersetzt den linearen polyfit in TemporalSelf durch ein nichtlineares
Zeitreihenmodell, das dynamische Muster im Körperzustand lernt:

  - Hydration springt nach Konsum (kein linearer Abfall)
  - Temperatur pendelt mit Tag/Nacht-Rhythmus
  - Energie: Plateau → Krise → schnelle Erholung

ESN-Prinzip (Echo State Network / Reservoir Computing):
  - Zufälliges festes Reservoir W_r (nie trainiert) mit Spektralradius < 1
  - W_in: feste zufällige Eingangsgewichte
  - W_out: EINZIGE trainierbare Schicht — via Ridge Regression (geschlossene Form)

Training: O(T·D²) statt O(T·D³) Backprop — für T=50, D=20 < 1ms.
Kein Gradientenabstieg, kein vanishing gradient, keine Hyperparameter-Suche.

Referenz: Jaeger (2001) "The echo state approach to analysing and training
recurrent neural networks"; Lukoševičius (2012) "A Practical Guide to ESN".
"""
from __future__ import annotations

import numpy as np


class WorldModelRNN:
    """
    Echo State Network für Körperzustandsvorhersage.

    Trainiert online auf dem gleitenden Verlaufsfenster von TemporalSelf.
    Gibt nichtlineare Mehrsschrittvorhersagen zurück, die in project()
    und time_to_crisis() eingesetzt werden.

    Verwendung:
        rnn = WorldModelRNN(seed=42)
        rnn.fit(sequence)          # sequence: (T, 4) body states
        traj = rnn.predict(x0, 50) # (50, 4) projected trajectory
    """

    def __init__(
        self,
        input_dim: int = 4,
        reservoir_dim: int = 20,
        spectral_radius: float = 0.95,
        ridge: float = 1e-3,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)

        # Festes Reservoir — zufällige Gewichte, nie geändert
        W = rng.standard_normal((reservoir_dim, reservoir_dim))
        eigs = np.max(np.abs(np.linalg.eigvals(W)))
        if eigs > 1e-10:
            self.W_r = W / eigs * spectral_radius
        else:
            self.W_r = W * spectral_radius

        # Eingangsgewichte: kleine Skalierung damit Reservoir nicht sättigt
        self.W_in = rng.standard_normal((reservoir_dim, input_dim)) * 0.1

        self.reservoir_dim = reservoir_dim
        self.input_dim     = input_dim
        self.ridge         = ridge

        # Trainierbare Ausgabegewichte (None = nicht trainiert)
        self.W_out: np.ndarray | None = None

        # Warmup-Zustand: Reservoir-State am Ende der letzten Trainingssequenz
        # Wird als Startpunkt für predict() genutzt
        self._h_warm: np.ndarray | None = None

        # Metriken für temporal_summary
        self.last_train_error: float | None = None
        self.train_count: int = 0

    # ------------------------------------------------------------------ #
    #  Reservoir-Schritt                                                   #
    # ------------------------------------------------------------------ #

    def _step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Einen Reservoir-Schritt berechnen."""
        return np.tanh(self.W_r @ h + self.W_in @ x)

    # ------------------------------------------------------------------ #
    #  Training (Ridge Regression)                                         #
    # ------------------------------------------------------------------ #

    def fit(self, sequence: np.ndarray, past_bias: float = 0.0) -> bool:
        """
        W_out auf der gegebenen Sequenz trainieren.

        Sequenz shape: (T, input_dim)
        Mindestlänge: 6 Schritte (3 Spin-up + 2 Training + 1 Target)

        past_bias > 0: ältere Samples erhalten mehr Gewicht (temporale Stasis,
          Northoff: Vergangenheit überschwemmt die Gegenwart).
          past_bias=0: uniform (normal), past_bias=3: alte Samples ~20× stärker.

        Gibt True zurück bei Erfolg, False bei Fehler (zu kurz, numerisch).
        """
        T = len(sequence)
        if T < 6:
            return False

        # Spin-up: erstes Drittel wegwerfen (eliminiert Anfangstransienten)
        spinup = max(2, T // 3)

        h = np.zeros(self.reservoir_dim)

        # Spin-up phase — Reservoir auf aktuelle Dynamik einschwingen lassen
        for x in sequence[:spinup]:
            h = self._step(x, h)

        # Kollektionsphase — Reservoir-States und Zielwerte sammeln
        H_list = []
        for x in sequence[spinup:-1]:
            h = self._step(x, h)
            H_list.append(h.copy())

        if len(H_list) < 2:
            return False

        H = np.array(H_list)             # (T', reservoir_dim)
        Y = sequence[spinup + 1:]        # (T', input_dim) — Ein-Schritt-Ziele

        # NaN/Inf-Check — kann bei schlechter Skalierung entstehen
        if not np.isfinite(H).all():
            H = np.where(np.isfinite(H), H, 0.0)
        if len(H) < 2:
            return False

        # Ridge Regression mit optionaler temporaler Gewichtung
        # past_bias > 0: ältere Samples dominieren (Northoff temporale Stasis)
        if past_bias > 0.0:
            T_h = len(H)
            idx = np.arange(T_h)
            # w[0] (ältestes) = exp(past_bias), w[T-1] (neuestest) = 1.0
            # Capped bei past_bias=5 um Overflow zu verhindern
            safe_bias = min(float(past_bias), 5.0)
            raw_w = np.exp(safe_bias * (T_h - 1 - idx) / max(T_h, 1))
            weights = raw_w * (T_h / raw_w.sum())  # normalisiert: Summe = T_h
            W_sq = np.diag(weights)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                A = H.T @ W_sq @ H + self.ridge * np.eye(self.reservoir_dim)
                rhs = H.T @ W_sq @ Y
            # NaN/Inf-Absicherung: falls Numerik versagt → normaler (ungewichteter) Fallback
            if not (np.isfinite(A).all() and np.isfinite(rhs).all()):
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    A   = H.T @ H + self.ridge * np.eye(self.reservoir_dim)
                    rhs = H.T @ Y
            try:
                W_out = np.linalg.solve(A, rhs).T
            except np.linalg.LinAlgError:
                return False
        else:
            # W_out = Y.T @ H @ (H.T @ H + λI)^{-1}
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                A = H.T @ H + self.ridge * np.eye(self.reservoir_dim)
            try:
                W_out = np.linalg.solve(A, H.T @ Y).T
            except np.linalg.LinAlgError:
                return False

        # NaN im W_out → Trainingsfehler, nicht speichern
        if not np.isfinite(W_out).all():
            return False

        self.W_out = W_out
        self.train_count += 1

        # Trainings-Rekonstruktionsfehler
        Y_pred = H @ W_out.T
        self.last_train_error = float(np.mean((Y - Y_pred) ** 2))

        # Warmup-State: Reservoir nach der letzten Eingabe der Sequenz
        # (Startpunkt für autoregressive Vorhersage)
        self._h_warm = self._step(sequence[-1], h)

        return True

    # ------------------------------------------------------------------ #
    #  Mehrsschrittvorhersage                                              #
    # ------------------------------------------------------------------ #

    def predict(self, x0: np.ndarray, steps: int) -> np.ndarray:
        """
        Autoregressive Mehrsschrittvorhersage.

        Startet vom Warmup-Zustand nach dem letzten fit()-Aufruf.
        Gibt (steps, input_dim) zurück, geclippt auf [0, 1].

        Wenn nicht trainiert: gibt lineare Extrapolation von x0 zurück.
        """
        if self.W_out is None:
            # Fallback: konstante Projektion
            return np.tile(x0.copy(), (steps, 1))

        h = self._h_warm.copy() if self._h_warm is not None else np.zeros(self.reservoir_dim)
        x = x0.copy()
        trajectory = []

        for _ in range(steps):
            h = self._step(x, h)
            x = np.clip(self.W_out @ h, 0.0, 1.0)
            trajectory.append(x.copy())

        return np.array(trajectory)  # (steps, input_dim)

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_trained(self) -> bool:
        return self.W_out is not None

    def summary(self) -> dict:
        return {
            "trainiert": self.is_trained,
            "trainings_runs": self.train_count,
            "letzter_fehler": (
                round(self.last_train_error, 6)
                if self.last_train_error is not None else None
            ),
            "reservoir_dim": self.reservoir_dim,
        }
