"""
Wissenschaftliches Vergleichsexperiment: Northoff-Zeitdynamik – ja oder nein?

Forschungsfrage (aus der RTF):
  "Verbessert eine intrinsische, verschachtelte zeitliche Dynamik die
   adaptive Selbstregulation eines Active-Inference-Agenten gegenüber
   demselben Agenten ohne Eigenzeit?"

Vier Bedingungen:
  A) Toni-Full     – Northoff + Solms + pymdp  (vollständige Architektur)
  B) Toni-NoNorth  – Solms + pymdp, aber flache Zeitdynamik (precision=1.0)
  C) Toni-NoMemory – Northoff + pymdp, aber kein autobiografisches Gedächtnis
  D) Toni-Random   – Zufällige Aktionen (Baseline)

Metriken:
  - Ø Wohlbefinden über Zeit
  - Anzahl homöostatischer Krisen (body state < 0.3)
  - Variabilität des Wohlbefindens (Stabilität)
  - Präzisions-Arousal-Kopplung (Northoff-Signatur)

Verwendung:
  python experiment_northoff.py
  python experiment_northoff.py --runs 5 --steps 800
"""
import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from toni import GridWorld, Toni
from toni.body import NEED_NAMES, SETPOINT
from toni.environment import FOOD, WATER, SHELTER


# ── Zufälliger Baseline-Agent ────────────────────────────────────────── #

class RandomAgent:
    """Baseline: zufällige Aktionen, kein Körper, kein Gedächtnis."""

    ACTIONS = ["up", "down", "left", "right", "consume"]

    def __init__(self, env: GridWorld, start_pos=None):
        mid = env.size // 2
        self.row, self.col = start_pos or (mid, mid)
        self.env = env
        from toni.body import Body
        self.body = Body()
        self.t = 0
        self.wellbeing = self.body.wellbeing()
        self.valence = 0.0
        self.history = {"wellbeing": [], "valence": [], "energy": [],
                        "hydration": [], "crises": []}

    def step(self):
        self.env.step()
        old_wb = self.body.wellbeing()
        self.body.step(env_temp=self.env.environment_temperature())

        action = np.random.choice(self.ACTIONS)
        self._execute(action)

        self.wellbeing = self.body.wellbeing()
        self.valence = self.wellbeing - old_wb
        self._record()
        self.t += 1
        return {"alive": self.body.is_alive()}

    def _execute(self, action):
        if action == "consume":
            res = self.env.consume_at(self.row, self.col)
            self.body.consume(
                "food" if res == FOOD else "water" if res == WATER else "shelter"
            )
        else:
            deltas = {"up": (-1,0), "down": (1,0), "left": (0,-1), "right": (0,1)}
            dr, dc = deltas.get(action, (0,0))
            nr, nc = self.row + dr, self.col + dc
            if self.env.is_valid(nr, nc):
                self.row, self.col = nr, nc

    def _record(self):
        h = self.history
        h["wellbeing"].append(self.wellbeing)
        h["valence"].append(self.valence)
        h["energy"].append(self.body.energy)
        h["hydration"].append(self.body.hydration)
        crisis = int(self.body.energy < 0.25 or self.body.hydration < 0.25)
        h["crises"].append(crisis)


# ── Simulation einer Bedingung ───────────────────────────────────────── #

def run_condition(condition: str, n_runs: int, n_steps: int,
                  seed_offset: int = 0) -> dict:
    """
    Führt eine Bedingung mit n_runs Wiederholungen aus.
    Gibt aggregierte Statistiken zurück.
    """
    all_wellbeing = []
    all_crises = []
    all_precision = []
    all_valences = []

    for run_i in range(n_runs):
        seed = 42 + run_i + seed_offset * 100
        env = GridWorld(size=9, seed=seed)

        if condition == "random":
            agent = RandomAgent(env, start_pos=(4, 4))
            for _ in range(n_steps):
                result = agent.step()
                if not result["alive"]:
                    break
            wbs = agent.history["wellbeing"]
            crises = agent.history["crises"]
            # Auffüllen wenn vorzeitig gestorben
            while len(wbs) < n_steps:
                wbs.append(wbs[-1] if wbs else -2.0)
                crises.append(1)
            all_wellbeing.append(wbs[:n_steps])
            all_crises.append(crises[:n_steps])
            all_valences.append(agent.history["valence"][:n_steps])
            all_precision.append([1.0] * n_steps)

        else:
            use_northoff = condition in ("full", "no_memory")
            use_pymdp = condition != "no_pymdp"

            # Für no_memory: Gedächtnis nach jedem Schritt leeren
            agent = Toni(env, start_pos=(4, 4),
                         use_pymdp=use_pymdp,
                         use_northoff=use_northoff)

            for _ in range(n_steps):
                result = agent.step()

                if condition == "no_memory":
                    # Gedächtnis deaktivieren: Priors immer uniform
                    agent.memory._location_valence_cache.clear()

                if not result["alive"]:
                    break

            h = agent.history
            wbs = h["wellbeing"]
            while len(wbs) < n_steps:
                wbs.append(wbs[-1] if wbs else -2.0)

            crises_list = [
                int(e < 0.25 or hyd < 0.25)
                for e, hyd in zip(h["energy"], h["hydration"])
            ]
            while len(crises_list) < n_steps:
                crises_list.append(1)

            all_wellbeing.append(wbs[:n_steps])
            all_crises.append(crises_list[:n_steps])
            all_valences.append((h["valence"] + [0.0] * n_steps)[:n_steps])
            prec = (h.get("precision", []) + [1.0] * n_steps)[:n_steps]
            all_precision.append(prec)

    wb_arr = np.array(all_wellbeing)   # (runs, steps)
    cr_arr = np.array(all_crises)
    vl_arr = np.array(all_valences)
    pr_arr = np.array(all_precision)

    return {
        "wellbeing_mean": wb_arr.mean(axis=0),
        "wellbeing_std": wb_arr.std(axis=0),
        "crisis_rate": cr_arr.mean(axis=0),
        "mean_wellbeing": float(wb_arr.mean()),
        "std_wellbeing": float(wb_arr.std()),
        "mean_crisis_rate": float(cr_arr.mean()),
        "mean_precision": float(pr_arr.mean()),
        "precision_variance": float(pr_arr.var()),
        "valence_positive_ratio": float((vl_arr > 0).mean()),
    }


# ── Visualisierung ───────────────────────────────────────────────────── #

CONDITIONS = {
    "full":       ("Toni-Full\n(Northoff+Solms+pymdp)", "#4caf50"),
    "no_northoff": ("Toni-NoNorthoff\n(Solms+pymdp, flat precision)", "#2196f3"),
    "no_memory":  ("Toni-NoMemory\n(Northoff+pymdp, kein Gedächtnis)", "#ff9800"),
    "random":     ("Toni-Random\n(Baseline)", "#f44336"),
}


def plot_results(results: dict, n_steps: int, save_path=None):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#0d0d1a")
    fig.suptitle(
        "Vergleichsexperiment: Northoff-Zeitdynamik und Active Inference\n"
        "Forschungsfrage: Verbessert intrinsische Eigenzeit die Selbstregulation?",
        color="white", fontsize=12, y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    ax_wb = fig.add_subplot(gs[0, :2])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_cr = fig.add_subplot(gs[1, 0])
    ax_val = fig.add_subplot(gs[1, 1])
    ax_sum = fig.add_subplot(gs[1, 2])

    ts = np.arange(n_steps)
    smooth = lambda x, w=30: np.convolve(x, np.ones(w)/w, mode="valid")

    # ── 1. Wohlbefinden über Zeit ────────────────────────────────────── #
    ax_wb.set_title("Ø Wohlbefinden über Zeit (geglättet)", color="#aaa", fontsize=9)
    for cond, (label, color) in CONDITIONS.items():
        if cond not in results:
            continue
        r = results[cond]
        wbm = r["wellbeing_mean"]
        wbs = r["wellbeing_std"]
        sm = smooth(wbm)
        ts_sm = ts[len(ts)-len(sm):]
        ax_wb.plot(ts_sm, sm, color=color, linewidth=1.8,
                   label=label.replace("\n", " "))
        ax_wb.fill_between(
            ts_sm,
            smooth(wbm - 0.5 * wbs),
            smooth(wbm + 0.5 * wbs),
            alpha=0.15, color=color
        )
    ax_wb.axhline(0, color="#ffffff30", linewidth=0.5, linestyle="--")
    ax_wb.legend(fontsize=7, framealpha=0.3, labelcolor="white", loc="lower right")
    ax_wb.set_xlabel("Simulationsschritte", color="#aaa", fontsize=8)
    ax_wb.set_ylabel("Wohlbefinden", color="#aaa", fontsize=8)
    ax_wb.tick_params(colors="#666", labelsize=7)
    for spine in ax_wb.spines.values():
        spine.set_color("#333")

    # ── 2. Balkendiagramm: Ø Wohlbefinden ───────────────────────────── #
    ax_bar.set_title("Ø Gesamtwohlbefinden", color="#aaa", fontsize=9)
    cond_keys = [c for c in CONDITIONS if c in results]
    means = [results[c]["mean_wellbeing"] for c in cond_keys]
    stds = [results[c]["std_wellbeing"] * 0.3 for c in cond_keys]
    colors = [CONDITIONS[c][1] for c in cond_keys]

    bars = ax_bar.barh(range(len(cond_keys)), means, xerr=stds,
                        color=colors, alpha=0.85, height=0.6)
    ax_bar.set_yticks(range(len(cond_keys)))
    ax_bar.set_yticklabels(
        [CONDITIONS[c][0].split("\n")[0] for c in cond_keys],
        color="white", fontsize=7
    )
    ax_bar.axvline(0, color="#ffffff50", linewidth=0.7)
    for i, (m, c) in enumerate(zip(means, cond_keys)):
        ax_bar.text(m + 0.01, i, f"{m:.3f}", va="center",
                    color="white", fontsize=7)
    ax_bar.tick_params(colors="#666", labelsize=7)
    for spine in ax_bar.spines.values():
        spine.set_color("#333")

    # ── 3. Krisenrate ────────────────────────────────────────────────── #
    ax_cr.set_title("Krisenrate (E/H < 0.25)", color="#aaa", fontsize=9)
    for cond, (label, color) in CONDITIONS.items():
        if cond not in results:
            continue
        cr = results[cond]["crisis_rate"]
        sm = smooth(cr, 50)
        ts_sm = ts[len(ts)-len(sm):]
        ax_cr.plot(ts_sm, sm * 100, color=color, linewidth=1.5,
                   label=CONDITIONS[cond][0].split("\n")[0])
    ax_cr.set_ylabel("Krisen (%)", color="#aaa", fontsize=8)
    ax_cr.set_xlabel("Schritte", color="#aaa", fontsize=8)
    ax_cr.legend(fontsize=6.5, framealpha=0.3, labelcolor="white")
    ax_cr.tick_params(colors="#666", labelsize=7)
    for spine in ax_cr.spines.values():
        spine.set_color("#333")

    # ── 4. Valenz-Ratio ──────────────────────────────────────────────── #
    ax_val.set_title("Anteil positiver Valenz", color="#aaa", fontsize=9)
    cond_vr = {c: results[c]["valence_positive_ratio"]
               for c in cond_keys if c in results}
    labels_v = [CONDITIONS[c][0].split("\n")[0] for c in cond_vr]
    vals_v = list(cond_vr.values())
    bar_colors_v = [CONDITIONS[c][1] for c in cond_vr]
    ax_val.bar(labels_v, vals_v, color=bar_colors_v, alpha=0.85)
    ax_val.set_ylim(0, 1)
    ax_val.axhline(0.5, color="#ffffff40", linewidth=0.7, linestyle="--",
                   label="Zufallsniveau")
    ax_val.set_ylabel("Anteil (0–1)", color="#aaa", fontsize=8)
    ax_val.tick_params(colors="#666", labelsize=7, axis='x', rotation=25)
    for spine in ax_val.spines.values():
        spine.set_color("#333")
    ax_val.legend(fontsize=6.5, framealpha=0.3, labelcolor="white")

    # ── 5. Zusammenfassungstabelle ───────────────────────────────────── #
    ax_sum.axis("off")
    ax_sum.set_title("Zusammenfassung", color="#aaa", fontsize=9)

    metrics = ["Ø Wohlbefinden", "Ø Krisenrate", "Valenz+", "Ø Präzision-Var."]
    table_data = []
    for c in cond_keys:
        r = results[c]
        row = [
            f"{r['mean_wellbeing']:.3f}",
            f"{r['mean_crisis_rate']*100:.1f}%",
            f"{r['valence_positive_ratio']*100:.0f}%",
            f"{r['precision_variance']:.4f}",
        ]
        table_data.append(row)

    col_labels = [CONDITIONS[c][0].split("\n")[0] for c in cond_keys]
    table = ax_sum.table(
        cellText=table_data,
        rowLabels=metrics,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.1, 1.6)

    # Farbe der Header
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor("#1a1a2e")
        cell.set_edgecolor("#333")
        cell.set_text_props(color="white")
        if i == 0 and j >= 0:
            cell.set_text_props(color=colors[j] if j < len(colors) else "white",
                                fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d0d1a")
        print(f"\n  Ergebnis gespeichert: {save_path}")

    plt.show()


# ── Hauptfunktion ─────────────────────────────────────────────────────── #

def main(n_runs: int = 3, n_steps: int = 600,
         conditions=None, save: bool = True):

    run_conditions = conditions or list(CONDITIONS.keys())

    print("═" * 65)
    print(" Northoff-Experiment  –  Active Inference Vergleich")
    print("═" * 65)
    print(f"  Läufe pro Bedingung: {n_runs}  |  Schritte: {n_steps}")
    print(f"  Bedingungen: {', '.join(run_conditions)}")
    print()

    results = {}
    for i, cond in enumerate(run_conditions):
        label = CONDITIONS[cond][0].split('\n')[0]
        print(f"  [{i+1}/{len(run_conditions)}] {label} ...", end="", flush=True)
        results[cond] = run_condition(
            condition=cond, n_runs=n_runs, n_steps=n_steps, seed_offset=i
        )
        r = results[cond]
        print(f"  Ø WB={r['mean_wellbeing']:.3f}  "
              f"Krisen={r['mean_crisis_rate']*100:.1f}%  "
              f"Valenz+={r['valence_positive_ratio']*100:.0f}%")

    print()
    print("═" * 65)
    print("  ERGEBNISSE")
    print("═" * 65)
    sorted_res = sorted(results.items(), key=lambda x: -x[1]["mean_wellbeing"])
    for rank, (cond, r) in enumerate(sorted_res, 1):
        label = CONDITIONS[cond][0].replace("\n", " ")
        print(f"  {rank}. {label}")
        print(f"     Ø Wohlbefinden:   {r['mean_wellbeing']:.4f}")
        print(f"     Krisenrate:       {r['mean_crisis_rate']*100:.2f}%")
        print(f"     Valenz positiv:   {r['valence_positive_ratio']*100:.1f}%")
        print(f"     Präzisions-Var.:  {r['precision_variance']:.5f}  "
              f"(Northoff-Signatur)")
        print()

    save_path = "northoff_experiment.png" if save else None
    try:
        plot_results(results, n_steps, save_path=save_path)
    except Exception as e:
        print(f"  Visualisierung nicht möglich ({e})")
        print("  Numerische Ergebnisse wurden ausgegeben.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Northoff-Experiment: mit vs. ohne Zeitdynamik"
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="Wiederholungen pro Bedingung (default: 3)")
    parser.add_argument("--steps", type=int, default=600,
                        help="Schritte pro Lauf (default: 600)")
    parser.add_argument("--conditions", nargs="+",
                        choices=list(CONDITIONS.keys()),
                        help="Welche Bedingungen (default: alle)")
    parser.add_argument("--no-save", action="store_true",
                        help="Plot nicht speichern")
    args = parser.parse_args()

    main(
        n_runs=args.runs,
        n_steps=args.steps,
        conditions=args.conditions,
        save=not args.no_save,
    )
