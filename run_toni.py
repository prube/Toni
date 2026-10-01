"""
Toni läuft.

Startet die Simulation und zeigt ein Live-Dashboard:
  - GridWorld mit Agent-Position und Ressourcen
  - Körperzustandsbalken (Energie, Hydration, Temperatur, Integrität)
  - Wohlbefinden & Valenz über Zeit
  - Northoff-Oszillatoren (5 Zeitskalen)
  - Aufmerksamkeits-Indikator (dominantes Bedürfnis)
  - Gedächtniskarte (Orts-Valenz-Prior)

Verwendung:
  python run_toni.py              # Live-Visualisierung
  python run_toni.py --headless   # Ohne Fenster, nur Text-Output
  python run_toni.py --steps 500  # Anzahl Schritte
"""
import argparse
import sys
import numpy as np

# ── Imports ──────────────────────────────────────────────────────────── #
from toni import GridWorld, Toni
from toni.body import NEED_NAMES
from toni.environment import FOOD, WATER, SHELTER, EMPTY


# ── Farben und Konstanten ─────────────────────────────────────────────── #
RESOURCE_COLORS = {
    EMPTY: "#1a1a2e",
    FOOD: "#4caf50",
    WATER: "#2196f3",
    SHELTER: "#9c27b0",
}
NEED_COLORS = ["#ff9800", "#2196f3", "#f44336", "#4caf50"]
OSC_COLORS = ["#e91e63", "#ff5722", "#ffeb3b", "#4caf50", "#00bcd4"]


def make_figure(size: int):
    """Matplotlib-Dashboard aufbauen."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import ListedColormap

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), facecolor="#0d0d1a")
    fig.suptitle("Toni  –  Northoff + Solms  KI-Prototyp",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.05, right=0.97, top=0.93, bottom=0.07)

    # GridWorld (groß, links oben)
    ax_grid = fig.add_subplot(gs[0:2, 0:2])
    ax_grid.set_title("Welt  (A=Toni  F=Nahrung  W=Wasser  S=Schutz)",
                      color="#aaa", fontsize=8)

    # Körperzustandsbalken
    ax_body = fig.add_subplot(gs[0, 2])
    ax_body.set_title("Körperzustand", color="#aaa", fontsize=8)

    # Wohlbefinden & Valenz
    ax_wb = fig.add_subplot(gs[0, 3])
    ax_wb.set_title("Wohlbefinden", color="#aaa", fontsize=8)

    # Northoff-Oszillatoren
    ax_osc = fig.add_subplot(gs[1, 2:4])
    ax_osc.set_title("Northoff – Intrinsische Zeitdynamik (5 Skalen)", color="#aaa", fontsize=8)

    # Valenz-Verlauf
    ax_val = fig.add_subplot(gs[2, 0:2])
    ax_val.set_title("Valenz  (Solms: Δwellbeing)", color="#aaa", fontsize=8)

    # Gedächtniskarte
    ax_mem = fig.add_subplot(gs[2, 2])
    ax_mem.set_title("Gedächtnis-Priors (Orts-Valenz)", color="#aaa", fontsize=8)

    # Präzision & Arousal
    ax_prec = fig.add_subplot(gs[2, 3])
    ax_prec.set_title("Präzision & Arousal", color="#aaa", fontsize=8)

    axes = {
        "grid": ax_grid, "body": ax_body, "wb": ax_wb,
        "osc": ax_osc, "val": ax_val, "mem": ax_mem, "prec": ax_prec,
    }
    return fig, axes


def render_grid(ax, env, agent, size):
    """GridWorld als farbige Heatmap mit Agent-Position."""
    import matplotlib.patches as mpatches

    ax.clear()
    img = np.zeros((size, size, 3))

    color_map = {
        EMPTY:   np.array([0.10, 0.10, 0.18]),
        FOOD:    np.array([0.30, 0.69, 0.31]),
        WATER:   np.array([0.13, 0.59, 0.95]),
        SHELTER: np.array([0.61, 0.15, 0.69]),
    }

    for r in range(size):
        for c in range(size):
            img[r, c] = color_map[env.grid[r, c]]

    ax.imshow(img, origin="upper", aspect="equal")

    # Agent markieren
    ax.scatter(agent.col, agent.row, s=300, c="#ff4444",
               marker="o", zorder=10, edgecolors="white", linewidths=1.5)
    ax.text(agent.col, agent.row, "T", ha="center", va="center",
            color="white", fontsize=7, fontweight="bold", zorder=11)

    # Grid-Linien
    for i in range(size + 1):
        ax.axhline(i - 0.5, color="#ffffff10", linewidth=0.4)
        ax.axvline(i - 0.5, color="#ffffff10", linewidth=0.4)

    # Tag/Nacht Indikator
    light = env.light_level()
    ax.set_facecolor((0.05 * light, 0.05 * light, 0.15 * (1 - light * 0.3)))

    # Legende
    legend_patches = [
        mpatches.Patch(color="#4caf50", label="Nahrung"),
        mpatches.Patch(color="#2196f3", label="Wasser"),
        mpatches.Patch(color="#9c27b0", label="Schutz"),
        mpatches.Patch(color="#ff4444", label="Toni"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=6, framealpha=0.4)

    need = NEED_NAMES[agent.dominant_need]
    ax.set_title(
        f"t={agent.t}  Bedürfnis: {need}  "
        f"Tag {agent.env.time // env.DAY_LENGTH + 1}",
        color="#aaa", fontsize=8
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_body(ax, agent):
    """Körperzustandsbalken."""
    ax.clear()
    ax.set_facecolor("#0d0d1a")

    state = agent.body.state()
    labels = ["Energie", "Hydration", "Temp.", "Integrität"]

    for i, (val, label) in enumerate(zip(state, labels)):
        color = NEED_COLORS[i]
        # Hintergrundbalken
        ax.barh(i, 1.0, color="#ffffff15", height=0.6)
        # Sollwert-Markierung
        from toni.body import SETPOINT
        ax.axvline(SETPOINT[i], ymin=(i) / 4, ymax=(i + 1) / 4,
                   color="#ffffff60", linewidth=1, linestyle="--")
        # Aktueller Wert
        ax.barh(i, val, color=color, alpha=0.85, height=0.6)
        ax.text(val + 0.02, i, f"{val:.2f}",
                va="center", color="white", fontsize=7)
        ax.text(-0.02, i, label,
                va="center", ha="right", color="#aaa", fontsize=7)

    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")

    # Gesamtwohlbefinden
    wb_norm = 1.0 + agent.wellbeing / 4.0  # [0..1] approx
    ax.text(0.5, -0.4, f"Wohlbefinden: {agent.wellbeing:.3f}",
            ha="center", color="#ffeb3b", fontsize=7,
            transform=ax.transData)


def render_wellbeing(ax, history, max_points=200):
    """Wohlbefinden über Zeit."""
    ax.clear()
    ax.set_facecolor("#0d0d1a")

    wb = history["wellbeing"][-max_points:]
    ts = range(len(wb))

    ax.fill_between(ts, wb, alpha=0.3, color="#4caf50")
    ax.plot(ts, wb, color="#4caf50", linewidth=1)
    ax.axhline(0, color="#ffffff30", linewidth=0.5, linestyle="--")

    if wb:
        ax.set_ylim(min(min(wb) * 1.1, -0.5), 0.1)

    ax.set_xlabel("Schritte", color="#aaa", fontsize=6)
    ax.tick_params(colors="#666", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#333")


def render_oscillators(ax, history, max_points=300):
    """Northoff-Oszillatoren: 5 verschachtelte Zeitskalen."""
    ax.clear()
    ax.set_facecolor("#0d0d1a")

    oscs = history["oscillators"][-max_points:]
    if not oscs:
        return

    osc_arr = np.array(oscs)
    freq_labels = ["sehr langsam (0.005 Hz)",
                   "langsam (0.02 Hz)",
                   "mittel (0.1 Hz)",
                   "schnell (0.8 Hz)",
                   "sehr schnell (4.0 Hz)"]
    ts = range(len(osc_arr))

    n = osc_arr.shape[1]
    for i in range(n):
        offset = i * 2.2
        ax.plot(ts, osc_arr[:, i] + offset,
                color=OSC_COLORS[i], linewidth=0.8, alpha=0.9,
                label=freq_labels[i])

    ax.legend(loc="upper left", fontsize=5.5, framealpha=0.3,
              ncol=1, labelcolor="white")
    ax.set_yticks([])
    ax.set_xlabel("Schritte", color="#aaa", fontsize=6)
    ax.tick_params(colors="#666", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#333")


def render_valence(ax, history, max_points=200):
    """Valenz-Verlauf (Solms: Δwellbeing)."""
    ax.clear()
    ax.set_facecolor("#0d0d1a")

    val = history["valence"][-max_points:]
    ts = range(len(val))

    pos = [max(0, v) for v in val]
    neg = [min(0, v) for v in val]

    ax.fill_between(ts, pos, alpha=0.6, color="#4caf50", label="positiv")
    ax.fill_between(ts, neg, alpha=0.6, color="#f44336", label="negativ")
    ax.axhline(0, color="#ffffff40", linewidth=0.5)

    ax.legend(fontsize=6, framealpha=0.3, labelcolor="white")
    ax.set_xlabel("Schritte", color="#aaa", fontsize=6)
    ax.tick_params(colors="#666", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#333")

    # Gleitender Mittelwert
    if len(val) > 20:
        smooth = np.convolve(val, np.ones(20) / 20, mode="valid")
        ax.plot(range(19, 19 + len(smooth)), smooth,
                color="#ffeb3b", linewidth=1.2, label="Trend")


def render_memory_map(ax, agent, size):
    """Gedächtniskarte: Orts-Valenz-Priors."""
    ax.clear()

    mem_map = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            mem_map[r, c] = agent.memory.location_prior((r, c))

    im = ax.imshow(mem_map, origin="upper", aspect="equal",
                   cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax.scatter(agent.col, agent.row, s=100, c="#ff4444",
               marker="o", zorder=10, edgecolors="white")
    ax.set_xticks([])
    ax.set_yticks([])


def render_precision_arousal(ax, history, max_points=200):
    """Präzision & Arousal über Zeit."""
    ax.clear()
    ax.set_facecolor("#0d0d1a")

    prec = history["precision"][-max_points:]
    arous = history["arousal"][-max_points:]
    ts = range(len(prec))

    if prec:
        ax.plot(ts, prec, color="#e91e63", linewidth=1,
                label=f"Präzision ({prec[-1]:.2f})")
    if arous:
        ax.plot(ts, arous, color="#00bcd4", linewidth=1,
                label=f"Arousal ({arous[-1]:.2f})")

    ax.axhline(1.0, color="#ffffff20", linewidth=0.5, linestyle="--")
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=6, framealpha=0.3, labelcolor="white")
    ax.set_xlabel("Schritte", color="#aaa", fontsize=6)
    ax.tick_params(colors="#666", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#333")


# ── Hauptfunktionen ──────────────────────────────────────────────────── #

def run_headless(steps: int = 1000, print_every: int = 50,
                 policy_len: int = 2) -> None:
    """Simulation ohne GUI."""
    env = GridWorld(size=9, seed=42)
    agent = Toni(env, start_pos=(4, 4), policy_len=policy_len)

    print("═" * 60)
    print(" Toni startet  (Northoff + Solms Prototyp)")
    print("═" * 60)
    print(f"  Welt: {env.size}×{env.size}  |  Schritte: {steps}")
    print()

    for _ in range(steps):
        result = agent.step()

        if agent.t % print_every == 0:
            print(agent.status)

        if not result["alive"]:
            print(f"\n[t={agent.t}] Toni ist nicht mehr lebensfähig.")
            break

    print()
    print("═" * 60)
    print(f"  Simulation abgeschlossen nach {agent.t} Schritten.")
    print(f"  Gedächtnis-Episoden: {len(agent.memory)}")
    print(f"  Letztes Wohlbefinden: {agent.wellbeing:.3f}")
    wb_arr = np.array(agent.history["wellbeing"])
    print(f"  Ø Wohlbefinden: {wb_arr.mean():.3f}  (σ={wb_arr.std():.3f})")
    print("═" * 60)


def run_live(steps: int = 2000, update_every: int = 5,
             policy_len: int = 2) -> None:
    """Simulation mit Live-Matplotlib-Dashboard."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    env = GridWorld(size=9, seed=42)
    agent = Toni(env, start_pos=(4, 4), policy_len=policy_len)

    fig, axes = make_figure(env.size)

    def animate(frame):
        for _ in range(update_every):
            result = agent.step()
            if not result["alive"]:
                ani.event_source.stop()
                break

        h = agent.history
        render_grid(axes["grid"], env, agent, env.size)
        render_body(axes["body"], agent)
        render_wellbeing(axes["wb"], h)
        render_oscillators(axes["osc"], h)
        render_valence(axes["val"], h)
        render_memory_map(axes["mem"], agent, env.size)
        render_precision_arousal(axes["prec"], h)

        return list(axes.values())

    ani = animation.FuncAnimation(
        fig, animate,
        frames=steps // update_every,
        interval=80,
        blit=False,
    )

    plt.show()


# ── Entry Point ──────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Toni – KI-Prototyp nach Northoff + Solms"
    )
    parser.add_argument("--headless", action="store_true",
                        help="Ohne Visualisierung (nur Text)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Anzahl Simulationsschritte (default: 2000)")
    parser.add_argument("--update", type=int, default=5,
                        help="Schritte pro Animations-Frame (default: 5)")
    parser.add_argument("--print-every", type=int, default=50,
                        help="Headless: Status alle N Schritte (default: 50)")
    parser.add_argument("--policy-len", type=int, default=2,
                        help="AIF Planungshorizont in Schritten (default: 2, empfohlen: 4-5)")
    args = parser.parse_args()

    if args.headless:
        run_headless(steps=args.steps, print_every=args.print_every,
                     policy_len=args.policy_len)
    else:
        try:
            run_live(steps=args.steps, update_every=args.update,
                     policy_len=args.policy_len)
        except Exception as e:
            print(f"GUI nicht verfügbar ({e}), starte headless...")
            run_headless(steps=args.steps, policy_len=args.policy_len)
