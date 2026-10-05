# Toni — Alle Skripte und Aufrufe

## Umgebungsvariablen (SAP-Proxy)

Die LLM-Skripte brauchen zwei Variablen, die **im Claude-Code-Terminal** bereits gesetzt sind.
Für eigene Terminal-Sitzungen müssen sie manuell gesetzt werden — **nicht** in `~/.zshrc` eintragen,
da sie sich pro Session ändern:

```bash
export ANTHROPIC_AUTH_TOKEN="<token-aus-claude-code-session>"
export ANTHROPIC_BASE_URL="<url-aus-claude-code-session>"
```

Alternativ: normaler Anthropic-API-Key (öffentliche API, kein SAP-Proxy):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Welche Werte?** Den aktuellen Token und die URL kannst du im Claude-Code-Terminal
mit folgendem Befehl auslesen:

```bash
echo "TOKEN: $ANTHROPIC_AUTH_TOKEN"
echo "URL:   $ANTHROPIC_BASE_URL"
```

Dann diese Werte in dein zweites Terminal kopieren.

---

## Übersicht

| Skript                      | Zweck                                           | LLM? |
|-----------------------------|------------------------------------------------|------|
| `run_toni.py`               | Hauptsimulation mit Matplotlib-Visualisierung  | nein |
| `run_toni_llm.py`           | Simulation + LLM-Kortex (Claude)               | ja   |
| `run_comparison.py`         | Reaktiv vs. Antizipatorisch (Northoff-Test)    | nein |
| `run_depression.py`         | Gesund vs. Solms-Depression + LLM-Vergleich    | ja   |
| `run_northoff_comparison.py`| 5-Weg: Gesund / Solms / Northoff / Borderline / Beide | nein |
| `run_toni_social.py`        | Zwei Agenten — soziale Positionierung + LLM    | ja   |

---

## run_toni.py — Hauptsimulation

Zeigt Matplotlib-Dashboard mit Grid, Körperwerten, Oszillatoren, Verlaufsgraphen.

```
--headless          Ohne Fenster, nur Textausgabe im Terminal
--steps N           Simulationsschritte        (default: 2000)
--update N          Frames pro Animationsschritt (default: 5)
--print-every N     Textausgabe alle N Schritte  (default: 50)
--policy-len N      AIF-Planungshorizont         (default: 2, empfohlen: 4–5)
```

**Beispiele:**

```bash
# Grafisches Fenster, 2000 Schritte
python3 run_toni.py

# Nur Terminal, 500 Schritte
python3 run_toni.py --headless --steps 500

# Langsame Animation (1 Schritt pro Frame)
python3 run_toni.py --update 1 --steps 300

# Mit längerem Planungshorizont
python3 run_toni.py --headless --steps 800 --policy-len 4
```

---

## run_toni_llm.py — Simulation + LLM-Kortex

Toni denkt laut nach: Claude artikuliert den inneren Zustand alle N Schritte.

```
--steps N           Simulationsschritte         (default: 100)
--reflect-every N   Kortex-Reflexion alle N Schritte (default: 10)
--model NAME        Claude-Modell               (default: claude-sonnet-4-6)
--dry-run           Kein API-Aufruf, nur Simulation
--seed N            Zufalls-Seed für die Welt   (default: 42)
--policy-len N      AIF-Planungshorizont        (default: 2)
```

**Beispiele:**

```bash
# Standard (braucht ANTHROPIC_AUTH_TOKEN oder ANTHROPIC_API_KEY)
python3 run_toni_llm.py

# Testen ohne API-Aufruf
python3 run_toni_llm.py --dry-run

# 200 Schritte, Reflexion alle 20
python3 run_toni_llm.py --steps 200 --reflect-every 20

# Anderes Modell, anderen Seed
python3 run_toni_llm.py --model claude-opus-5 --seed 99

# Schnell: nur Simulation prüfen
python3 run_toni_llm.py --dry-run --steps 80 --reflect-every 40
```

---

## run_comparison.py — Reaktiv vs. Antizipatorisch

Northoff-Kern-Experiment: gleicher Seed, gleiche Welt — ein Agent nur reaktiv,
einer mit temporalem Selbst (Zukunftsprojektion).

```
--steps N           Simulationsschritte         (default: 200)
--seed N            Zufalls-Seed                (default: 42)
--verbose           Jeden Schritt ausgeben
--policy-len N      AIF-Planungshorizont        (default: 2)
```

**Beispiele:**

```bash
# Standard
python3 run_comparison.py

# Saubererer Vergleich mit mehr Schritten
python3 run_comparison.py --steps 400 --seed 55

# Zeilenweiser Verlauf sichtbar
python3 run_comparison.py --verbose --steps 100

# Mehrere Seeds von Hand testen
for s in 42 55 17 99 123; do
    python3 run_comparison.py --seed $s --steps 300
done
```

---

## run_depression.py — Solms-Depression

Vergleicht gesunden Agenten mit Solms-depressivem Agenten.
Depression = Kollaps des SEEKING-Systems (5 Mechanismen):
Anhedonie, SEEKING-Kollaps, Zeithorizont, Desynchronisation, Rumination.
LLM artikuliert Unterschied alle N Schritte.

```
--steps N           Simulationsschritte         (default: 300)
--reflect-every N   LLM-Reflexion alle N Schritte (default: 60)
--seed N            Zufalls-Seed                (default: 42)
--model NAME        Claude-Modell               (default: claude-sonnet-4-6)
--dry-run           Kein API-Aufruf
--depression F      Depressionsstärke 0.0–1.0   (default: 1.0)
--policy-len N      AIF-Planungshorizont        (default: 2)
```

**Beispiele:**

```bash
# Volles Experiment mit LLM
python3 run_depression.py

# Ohne LLM, nur Simulation
python3 run_depression.py --dry-run

# Leichte Depression (50%)
python3 run_depression.py --dry-run --depression 0.5 --steps 500

# Langer Lauf, seltene Reflexion
python3 run_depression.py --steps 600 --reflect-every 150 --seed 55
```

---

## run_northoff_comparison.py — 5-Weg-Depressions-Vergleich

**Das Northoff-vs-Solms-vs-Borderline-Dissoziations-Experiment.**
Fünf Bedingungen gleichzeitig:

| Bedingung  | Parameter                           | Erwartung              |
|------------|-------------------------------------|------------------------|
| Gesund     | d=0.0, nd=0.0, bl=0.0               | Baseline               |
| Solms      | d=1.0, nd=0.0, bl=0.0  (SEEKING-Kollaps)    | FINDET aber WILL NICHT |
| Northoff   | d=0.0, nd=1.0, bl=0.0  (Rest-Self-Overlap)  | WILL aber FINDET NICHT |
| Borderline | d=0.0, nd=0.0, bl=1.0  (Regulationsstörung) | WEISS NICHT OB ES WILL |
| Beide      | d=1.0, nd=1.0, bl=0.0              | kombiniert             |

```
--seed N            Einzelner Seed              (default: 55)
--seeds N [N ...]   Mehrere Seeds (überschreibt --seed)
--steps N           Simulationsschritte         (default: 300)
```

**Beispiele:**

```bash
# Standard (Seed 55 = sauberste Dissoziation)
python3 run_northoff_comparison.py

# Mehr Schritte für stabilere Ergebnisse
python3 run_northoff_comparison.py --seed 55 --steps 400

# Mehrere Seeds gleichzeitig testen
python3 run_northoff_comparison.py --seeds 42 55 17 99

# Schwacher Seed zum Vergleich
python3 run_northoff_comparison.py --seed 42 --steps 300
```

**Kein LLM erforderlich.** Läuft vollständig lokal.

---

## run_toni_social.py — Zwei Agenten, soziale Positionierung

Northoff: das Selbst positioniert sich temporal, räumlich UND sozial.
Zwei Agenten teilen die gleiche Welt und beobachten sich gegenseitig.
LLM artikuliert die soziale Dimension.

```
--steps N           Simulationsschritte         (default: 150)
--reflect-every N   LLM-Reflexionsintervall     (default: 25)
--seed N            Zufalls-Seed                (default: 42)
--model NAME        Claude-Modell               (default: claude-sonnet-4-6)
--dry-run           Kein API-Aufruf
--policy-len N      AIF-Planungshorizont        (default: 2)
```

**Beispiele:**

```bash
# Mit LLM
python3 run_toni_social.py

# Ohne LLM testen
python3 run_toni_social.py --dry-run

# Längeres Experiment
python3 run_toni_social.py --steps 300 --reflect-every 50 --seed 99
```

---

## Schnell-Referenz: Alle Skripte ohne LLM

Diese laufen ohne Umgebungsvariablen:

```bash
python3 run_toni.py --headless --steps 500
python3 run_comparison.py --steps 300 --seed 55
python3 run_northoff_comparison.py
python3 run_depression.py    --dry-run --steps 400
python3 run_toni_llm.py      --dry-run --steps 100
python3 run_toni_social.py   --dry-run --steps 150
```

## Schnell-Referenz: Modell-IDs (2025/2026)

```
claude-sonnet-4-6      Standard (default in allen Skripten)
claude-opus-5          Stärkstes Modell (langsamer, teurer)
claude-haiku-4-5-20251001  Schnellstes Modell (günstiger)
```
