# Toni — Gesprächsexport
## KI-Prototyp nach Northoff + Solms + Enaktivismus + Active Inference

Dieses Dokument fasst alle Konzepte, Erklärungen, Architekturentscheidungen und
theoretischen Verbindungen zusammen, die im Verlauf der Entwicklung von Toni
besprochen wurden.

---

## Inhaltsverzeichnis

1. [Projektübersicht und Kernprinzip](#1-projektübersicht-und-kernprinzip)
2. [Northoff: Temporo-Spatial Theory of Consciousness (TTC)](#2-northoff-temporo-spatial-theory-of-consciousness-ttc)
3. [Solms: Affektives Bewusstsein und Homöostase](#3-solms-affektives-bewusstsein-und-homöostase)
4. [Active Inference via pymdp](#4-active-inference-via-pymdp)
5. [Enaktivismus: Autopoiese und Affordanzen](#5-enaktivismus-autopoiese-und-affordanzen)
6. [Depression-Simulation: 5 Mechanismen](#6-depression-simulation-5-mechanismen)
7. [Anhedonie — vertieft](#7-anhedonie--vertieft)
8. [Northoff: "Bewusstsein entsteht wenn Rhythmen in Phase geraten"](#8-northoff-bewusstsein-entsteht-wenn-rhythmen-in-phase-geraten)
9. [Dashboard-Erklärung](#9-dashboard-erklärung)
10. [Architekturprinzip: LLM als Kortex, nicht als Kern](#10-architekturprinzip-llm-als-kortex-nicht-als-kern)
11. [Start-Kommandos](#11-start-kommandos)

---

## 1. Projektübersicht und Kernprinzip

**Toni** ist ein KI-Forschungsprototyp, der verkörperte Handlungsfähigkeit
(Embodied Agency) nach vier theoretischen Rahmenbedingungen modelliert:

| Theorie | Vertreter | Kernbeitrag |
|---|---|---|
| TTC | Georg Northoff | Bewusstsein als temporo-spatiale Ausrichtung |
| Affektives Bewusstsein | Mark Solms | Primärbewusstsein = Homöostase + Affekt |
| Active Inference | Karl Friston | EFE-Minimierung als universelles Handlungsprinzip |
| Enaktivismus | Maturana, Varela, Gibson | Bedeutung entsteht aus Körper-Umwelt-Kopplung |

**Kein externer Reward. Kein Programmierer, der sagt "battery < 10% → go_to_charger".**
Alle Motivation entsteht aus internen Zustandsabweichungen.

### Verzeichnisstruktur

```
toni/
  body.py           — Homöostatischer Körper (Energie, Hydration, Temp, Integrität)
  oscillators.py    — Northoff: 5 Zeitskalen, Kuramoto-Kopplung
  interoception.py  — Verrauschte, präzisionsmodulierte Körperwahrnehmung
  memory.py         — Autobiografisches Gedächtnis (Orts-Valenz-Prior)
  temporal_self.py  — Drei Zeitfenster: Vergangenheit, Gegenwart, Zukunft
  social_self.py    — Northoff Sozial-Achse: Beobachtung anderer Agenten
  enactive_self.py  — Maturana/Varela Autopoiese + Gibson Affordanzen
  active_inference.py — pymdp: A-, B-, C-, D-Matrizen
  agent.py          — Integration aller Module (Toni-Klasse)
  llm_cortex.py     — Sprachlicher Kortex (nur Reflexion, kein Feedback)
  environment.py    — GridWorld: Nahrung, Wasser, Shelter
```

---

## 2. Northoff: Temporo-Spatial Theory of Consciousness (TTC)

### Kernthese

Bewusstsein ist kein Objekt und kein Zustand — es ist ein **Prozess der
Ausrichtung** (alignment). Das Gehirn hat eine permanente Eigenaktivität
(intrinsic activity), die sich an externe Rhythmen ankoppelt. Wenn interne
und externe Zeitstrukturen in Phase geraten, entsteht das, was wir Erleben nennen.

### Drei Achsen des Selbst (Northoff)

1. **Temporal-Selbst**: Das System verortet sich in der Zeit — Vergangenheit
   (Gedächtnis), Gegenwart (Aufmerksamkeit), Zukunft (Antizipation)
2. **Spatial-Selbst**: Das System verortet sich im Raum — Körpergrenzen,
   Umweltbezug
3. **Soziales Selbst**: Das System verortet sich in Relation zu anderen —
   Beobachtung, Konkurrenz, Kooperation

### Implementierung in Toni

#### 2a. Verschachtelte Zeitdynamik (`oscillators.py`)

Fünf Oszillatoren auf hierarchisch geschachtelten Frequenzebenen:

```python
FREQUENCIES = [0.005, 0.02, 0.1, 0.8, 4.0]
#              sehr    lang  mit  schn sehr
#              langsam sam   tel  ell  schnell
```

**Nesting**: Jede Ebene wird von der nächst-langsameren amplitudenmoduliert:

```python
for i in range(1, len(self.oscillators)):
    parent_val = vals[i - 1]
    modulation = self.NESTING_STRENGTH * parent_val * dt
    v = self.oscillators[i].step(dt, phase_push=modulation)
```

→ Wenn der langsame Oszillator unten ist, flachen die schnellen ab.
  Das ist Northoffs "temporo-spatial nestedness" — verschiedene Zeitskalen
  sind nicht unabhängig, sondern hierarchisch gekoppelt.

#### 2b. Kuramoto-Kopplung an Tagesrhythmus

Der langsamste Oszillator koppelt an den Tagesrhythmus der Umwelt:

```python
kuramoto_push = (COUPLING_STRENGTH * k_ext_scale
                 * np.sin(env_phase - osc0.phase) * dt)
```

`sin(θ_ext - θ_int)` ist die klassische Kuramoto-Formel:

| Phasenzustand | sin(…) | Effekt |
|---|---|---|
| intern hinkt nach | > 0 | Push vorwärts → synchronisiert |
| **in Phase** | **= 0** | **kein Push — stabil eingerastet** |
| intern läuft voraus | < 0 | Push zurück → synchronisiert |

Das System rastet von selbst ein, sobald es synchron ist.

#### 2c. Präzisionsmodulation

Die Phase-Alignment wirkt sich direkt auf die Qualität der Wahrnehmung aus:

```python
def precision(self) -> float:
    slow = self.oscillators[0].value
    return max(0.3, 1.0 + 0.4 * slow)  # [0.6, 1.4]
```

**In Phase = hohe Präzision = scharfe Interozeption = gute Entscheidungen.**

Das ist der Mechanismus, durch den Northoffs Theorie verhaltensrelevant wird:
nicht direkt ("Bewusstsein"), sondern funktional ("Wahrnehmungsqualität").

#### 2d. Temporales Selbst (`temporal_self.py`)

Drei Zeitfenster:
- **Vergangenheit**: Echo State Network (ESN) lernt auf Verlaufsdaten
- **Gegenwart**: aktueller Körperzustand
- **Zukunft**: ESN-Projektion → `future_urgency()` → antizipatorische AIF-Motivation

**Northoffs Kernthese in Zahlen** (aus `run_comparison.py`):

| | Reaktiver Agent | Antizipatorischer Agent |
|---|---|---|
| Krisen | mehr | deutlich weniger |
| Erste Krise | früher | später |
| Reaktionslatenz | hoch | niedrig (handelt bevor Krise eintritt) |

---

## 3. Solms: Affektives Bewusstsein und Homöostase

### Kernthese

Primärbewusstsein ist nicht kognitiv — es ist **affektiv**. Das erste Erleben
ist kein Gedanke, sondern ein Gefühl: "Gut für mich" oder "Schlecht für mich".
Bewusstsein entsteht nicht in der Großhirnrinde, sondern im Hirnstamm —
dort wo homöostatische Bedürfnisse verwaltet werden.

Solms' Gleichung:
```
Valenz = Δwellbeing = "Wird es für diesen Organismus besser oder schlechter?"
```

### Implementierung in Toni

#### 3a. Valenz als Δwellbeing

```python
old_wellbeing = self.body.wellbeing()
self.body.step(...)
new_wellbeing = self.body.wellbeing()
self.valence = new_wellbeing - old_wellbeing
```

Keine externe Belohnungsfunktion. Der "Reward" ist der homöostatische Fehler.

#### 3b. Wellbeing-Funktion

```python
def wellbeing(self) -> float:
    state = self.state()            # [energy, hydration, temp, integrity]
    error = SETPOINT - state        # Abweichung vom Sollwert (0.75)
    deficit = np.maximum(0.0, error)
    return float(-np.dot(NEED_WEIGHTS, deficit ** 2))
```

Wellbeing ist immer ≤ 0 (Defizit-quadratisch). Maximum = 0.0 (perfekter Sollzustand).

#### 3c. SEEKING-System (Panksepp/Solms)

Der C-Vektor der Active Inference modelliert das mesolimbische Dopaminsystem
(SEEKING):

```python
C_vec[FOOD] = (3.0 + urgency * 1.5) * seeking_gain
```

`seeking_gain` kommt aus dem Körperzustand — hohe Dringlichkeit = starkes
Seeking. Bei Depression wird `seeking_gain` reduziert → SEEKING-Kollaps.

---

## 4. Active Inference via pymdp

### Grundprinzip

Active Inference (Friston): Ein System minimiert **Expected Free Energy (EFE)**
— es wählt Aktionen, die sein Modell der Welt bestätigen UND seine Präferenzen
erfüllen.

### POMDP-Modell

| Matrix | Bedeutung | Quelle in Toni |
|---|---|---|
| **A** | Wahrnehmungsmodell P(o\|s): Welche Ressource ist wo? | GridWorld + Northoff-Präzision |
| **B** | Transitionsmodell P(s'\|s,a): Wie bewege ich mich? | Deterministisches Grid |
| **C** | Log-Präferenzen: Was will ich wahrnehmen? | Solms (dominantes Bedürfnis) |
| **D** | Prior über Positionen: Wo bin ich wahrscheinlich gut? | Autobiografisches Gedächtnis + Soziales Selbst |

### A-Matrix: Northoff-Präzision

```python
A_np[resource, pos] += precision * 10.0
```

Hohe Präzision → scharfe A-Matrix → Exploitation (gezielte Navigation)
Niedrige Präzision → flache A-Matrix → Exploration (Neugier)

### C-Vektor: Solms + Enaktivismus + Depression

```python
# Solms: dominantes Bedürfnis setzt Grundpräferenz
C_vec[FOOD] = (3.0 + urgency * 1.5) * seeking_gain

# Enaktivismus: Affordanz-Boost (Gibson)
C_vec[FOOD]    += affordances.get("food", 0.0) * 2.5
C_vec[WATER]   += affordances.get("water", 0.0) * 2.5
C_vec[SHELTER] += affordances.get("shelter", 0.0) * 1.5
```

### D-Vektor: Gedächtnis + Soziales Selbst

```python
# Autobiografisches Gedächtnis
boost = max(0.0, val) * 5.0
D_vec[pos] += boost

# Soziales Selbst (andere Agenten haben dort Ressourcen gefunden)
D_vec += social_prior
```

---

## 5. Enaktivismus: Autopoiese und Affordanzen

### Autopoiese (Maturana & Varela)

**"Ein lebendes System ist ein System, das die Bedingungen seiner eigenen
Organisation aufrechterhält."**

Nicht Überleben im Darwinschen Sinne — sondern **strukturelle Kohärenz**.
Ein Organismus ist autopoietisch, wenn er sich selbst als Einheit reproduziert.
Wenn diese Fähigkeit zusammenbricht, hört der Organismus auf zu sein.

In Toni: **Viabilität** = kann der Agent die Bedingungen seiner Organisation halten?

```python
@staticmethod
def viability(body_state) -> float:
    mob = EnactiveSelf.mobility(body_state)
    e, h, _t, i = body_state
    homeostasis = min(e / 0.75, h / 0.75, i)
    return float(np.clip(homeostasis * (0.5 + 0.5 * mob), 0.0, 1.0))
```

Viabilität ≠ Wohlbefinden:
- Wellbeing misst Defizit vom Sollwert (Solms)
- Viabilität misst strukturelle Handlungsfähigkeit (Maturana/Varela)
- Ein Agent kann sich schlecht fühlen (wellbeing < 0) und trotzdem noch handlungsfähig sein

### Affordanzen (James J. Gibson)

**"Affordanzen sind die Handlungsmöglichkeiten, die die Umwelt einem spezifischen
Körper anbietet."**

Bedeutung entsteht nicht im Gehirn — sie liegt in der Relation zwischen Körper
und Umwelt. Eine Treppe ist nur für Beine mit bestimmter Länge eine Treppe.
Nahrung ist nur Nahrung für jemanden, der Hunger hat.

In Toni:
```python
# Affordanz = Defizit × Nähe
food_afford  = e_deficit * exp(-food_dist  / 4.0)
water_afford = h_deficit * exp(-water_dist / 4.0)
```

Nahrung ist attraktiv wenn:
1. Der Energielevel niedrig ist (Körper braucht es) UND
2. Die Nahrung nah ist (Umwelt bietet es an)

Beides zusammen = Affordanz. Eines allein reicht nicht.

### Warum Enaktivismus wichtig ist

Klassische KI: "Der Agent berechnet einen Wert und wählt die beste Option."
Enaktivismus: "Bedeutung existiert nicht im Agenten und nicht in der Umwelt —
sie entsteht in ihrer Interaktion."

Gibson nannte das **"ecological validity"**: das Wahrnehmungssystem ist nicht
für die Repräsentation der Welt gebaut, sondern für die Navigation durch sie.

### Architektonische Entscheidung

Der `EnactiveSelf` liefert **nur einen Signal-Output** an den AIF-C-Vektor.
Kein Feedback-Pfad zurück. Die Trennung bleibt sauber:

```
Körper → EnactiveSelf.aif_signal() → C-Vektor → AIF → Aktion → Umwelt
                                                              ↑
                          kein direkter Feedback-Pfad hier ──┘
```

---

## 6. Depression-Simulation: 5 Mechanismen

### Solms' Theorie der Depression

**"Depression ist keine kognitive Störung — es ist ein Kollaps des
SEEKING-Systems."**

Das SEEKING-System (mesolimbisches Dopaminsystem, Panksepp) ist das
Antriebssystem, das Organismen motiviert, aktiv in der Welt zu suchen.
Bei Depression ist dieses System gedämpft — nicht die Gedanken sind falsch,
sondern die Motivation zu handeln ist zusammengebrochen.

### Die 5 Mechanismen (alle linear in `depression_level` [0..1])

```python
d = self.depression_level

anhedonia   = 1.0 - 0.8 * d    # Konsum-Gain reduziert (body.py)
k_ext_scale = 1.0 - 0.95 * d   # Zirkadiane Desynchronisation (oscillators.py)
horizon     = max(5, int(80 * (1.0 - 0.85 * d)))  # Zeithorizont (temporal_self.py)
seeking_gain = 1.0 - 0.7 * d   # SEEKING-Antrieb (active_inference.py C-Vektor)
rumination  = 1.0 + 4.0 * d    # Negative Erinnerungen verstärkt (memory.py)
```

#### Mechanismus 1: Anhedonie
`body.consume()` multipliziert Gain mit `anhedonia_factor`:
```python
gain = min(0.35, 1.0 - self.energy) * self.anhedonia_factor
```
Bei `d=1.0`: Trinken gibt nur 20% des normalen Hydrations-Gewinns.

#### Mechanismus 2: SEEKING-Kollaps
`seeking_gain=0.3` skaliert den gesamten C-Vektor herunter:
```python
C_vec[FOOD] = (3.0 + urgency * 1.5) * 0.3
```
Die AIF-Präferenz für Ressourcen ist massiv gedämpft. Der Agent "will nicht".

#### Mechanismus 3: Zeithorizont-Kollaps
`horizon=12` (normal: 80) — der Agent projiziert nur 12 Schritte in die Zukunft:
```python
horizon = max(5, int(80 * (1.0 - 0.85 * 1.0))) = 12
```
Antizipatorisches Handeln ist fast ausgeschaltet. Nur reaktive Politik.

#### Mechanismus 4: Zirkadianer Desynchronisation
`k_ext_scale≈0` trennt interne Rhythmen von der Tageszeit:
```python
kuramoto_push ≈ 0  →  interner Rhythmus driftet frei
```
Präzision bleibt flach (~1.0) — keine Tagesrhythmus-abhängige Schärfung
der Wahrnehmung. Klinisch: flaches Cortisol-Profil, gestörter Schlaf.

#### Mechanismus 5: Rumination
Negative Erlebnisse brennen sich stärker ins Gedächtnis ein:
```python
stored_valence = valence * rumination_factor if valence < 0 else valence
```
Bei `d=1.0`: negative Erinnerungen werden 5× verstärkt gespeichert.
Die Gedächtniskarte füllt sich mit negativen Priors → schlechtere Navigation.

### Experimentelle Ergebnisse (`run_depression.py`)

Aus einem 300-Schritt-Experiment (Seed 42, `depression=1.0`):

| Metrik | Gesunder Agent | Depressiver Agent |
|---|---|---|
| Konsum-Aktionen | ~15 | 0 |
| Krisenschritte (%) | ~45% | ~67% |
| Erste Krise | Schritt ~85 | Schritt ~20 |
| Ø Wohlbefinden | deutlich besser | massiv schlechter |

**Das Paradox**: Der depressive Agent konsumiert 0 Mal — nicht wegen Anhedonie
(die würde beim Trinken greifen), sondern weil SEEKING so kollabiert ist,
dass er nie zu einer Ressource navigiert. Anhedonie und SEEKING-Kollaps
kommen nicht einmal beide zum Tragen — der SEEKING-Kollaps ist dominant.

---

## 7. Anhedonie — vertieft

### Was Anhedonie ist

Anhedonie bedeutet wörtlich: **Unfähigkeit, Freude zu empfinden**.
In Solms' Modell präziser: das Ausbleiben des positiven Valenz-Signals
nach Bedürfnisbefriedigung.

**Das Problem ist nicht das Wollen — es ist das Nicht-Bekommen.**

Gesunder Agent trinkt:
```
Hydration 0.3 → trinkt → Hydration 0.65 → Δwellbeing = +0.28 → Valenz positiv
```

Anhedonischer Agent trinkt dasselbe:
```
Hydration 0.3 → trinkt → Hydration 0.37 → Δwellbeing = +0.05 → Valenz fast null
```

### Berridges Unterscheidung: WANTING vs. LIKING

Berridge (Neurowissenschaft) trennt zwei Systeme:
- **WANTING** (mesolimbisches Dopamin): Antrieb zu suchen
- **LIKING** (opioide Signale): die Befriedigung im Moment des Konsums

In Toni:
- `seeking_gain` dämpft das WANTING
- `anhedonia_factor` dämpft das LIKING

Beides zusammen: Der depressive Agent sucht kaum noch, und wenn er
findet, bringt es ihm fast nichts.

### Der Teufelskreis

```
wenig Befriedigung nach Konsum
        ↓
schwacher positiver Valenz-Spike
        ↓
Gedächtniskarte lernt kaum positives über diesen Ort
        ↓
D-Prior für gute Orte bleibt flach
        ↓
AIF navigiert weniger gezielt zu Ressourcen
        ↓
häufigere Krisen → mehr negative Valenz
        ↓
Rumination verstärkt negative Erinnerungen 5×
        ↓
Gedächtniskarte füllt sich mit negativen Priors
        ↓  (zurück nach oben)
```

---

## 8. Northoff: "Bewusstsein entsteht wenn Rhythmen in Phase geraten"

### Der Mechanismus

Die Kuramoto-Formel `K * sin(θ_ext - θ_int)` hat eine elegante Eigenschaft:

| Phasenzustand | sin(…) | Effekt |
|---|---|---|
| intern hinkt nach | > 0 | Push vorwärts → synchronisiert |
| **in Phase** | **= 0** | **kein Push nötig — stabil eingerastet** |
| intern läuft voraus | < 0 | Push zurück → synchronisiert |

Sobald synchron: Eigenentwicklung. Nicht erzwungen, sondern emergent.

### Von Synchronisation zu Wahrnehmung

```python
def precision(self) -> float:
    slow = self.oscillators[0].value
    return max(0.3, 1.0 + 0.4 * slow)  # [0.6, 1.4]
```

- Langsamer Oszillator synchron, Tageshöhepunkt → `slow ≈ +1` → Präzision ≈ 1.4
- Langsamer Oszillator desynchron, Nachtphase → `slow ≈ -1` → Präzision ≈ 0.6

**In Phase = gute Selbstwahrnehmung = gute Entscheidungen.**

### Was Depression zeigt

Bei `k_ext_scale ≈ 0` (Depression) koppeln interne Rhythmen nicht mehr an
die Tageszeit. Die Oszillatoren laufen frei. Präzision bleibt flach.

Das entspricht klinisch messbaren Befunden bei Depression:
- Flaches Cortisol-Profil (kein Morgengipfel)
- Gestörter Schlaf-Wach-Rhythmus
- Vermindertes circadianes Temperaturprofil

Northoffs Theorie wird hier empirisch greifbar: Desynchronisation ist
nicht eine Folge der Depression — sie ist ein kausaler Mechanismus.

### Die Nesting-Hierarchie

```
Ebene 0 (0.005 Hz) — Stimmungsbaseline — koppelt an Umwelt
    moduliert →
Ebene 1 (0.02 Hz) — Triebrhythmus
    moduliert →
Ebene 2 (0.1 Hz) — Aufmerksamkeitstakt — treibt Arousal
    moduliert →
Ebene 3 (0.8 Hz) — Wahrnehmungstakt
    moduliert →
Ebene 4 (4.0 Hz) — Reaktionstakt
```

Bewusstsein ist für Northoff kein einzelner Rhythmus, sondern eine
**Hierarchie von Rhythmen, die alle voneinander wissen**.

---

## 9. Dashboard-Erklärung

Das Visualisierungsfenster (`run_toni.py`) hat 6 Panels:

### Panel 1 (oben links): GridWorld

Die Umwelt. **Grün** = Nahrung, **Blau** = Wasser, **Lila** = Shelter,
**Roter Kreis "T"** = Tonis aktuelle Position.
Ressourcen respawnen nach Verbrauch.

### Panel 2 (oben mitte): Körperzustandsbalken

```
Integrität  0.97  ██████████  gut
Temp.       0.60  ██████░░░░  normal
Hydration   0.20  ██░░░░░░░░  KRISE (<0.25)
Energie     0.01  ░░░░░░░░░░  LEBENSBEDROHLICH
```

Gestrichelte Linie = SETPOINT (0.75). Alles links = Defizit.
Wohlbefinden = gewichtete Summe aller Defizit-Quadrate (immer ≤ 0).

### Panel 3 (oben rechts): Wohlbefindens-Kurve

Verlauf von t=0 bis jetzt. Zacken nach oben = Konsum-Ereignisse.
Konstanter Abstieg = Agent findet nicht genug.

### Panel 4 (mitte rechts): 5 Northoff-Oszillatoren

Die 5 Zeitskalen sichtbar. Nestedness sichtbar: wenn die langsame rote
Welle unten ist, werden die schnelleren Wellen flacher.

### Panel 5 (unten links): Valenz über Zeit

**Grüne Fläche** = positive Valenz (Konsum, Erholung).
**Rote Fläche** = negative Valenz (Körperverfall, Notlagen).
**Gelbe Linie** = Trend.

Rot dominiert = Toni erlebt den Körperverfall als kontinuierliches Leiden.

### Panel 6 (unten mitte): Gedächtniskarte

Heatmap der GridWorld.
- **Gelb/warm** = Ort mit positiver Valenz-Geschichte (gute Erfahrungen)
- **Dunkel** = neutral oder unbesucht
- **Roter Punkt** = aktuelle Position

Die Karte formt zukünftige Entscheidungen: D-Vektor des AIF bevorzugt
gelbe Orte. Vergangenheit wirkt auf Gegenwart.

### Panel 7 (unten rechts): Präzision und Arousal

**Pink** = Präzision — folgt dem langsamen Oszillator. Hoch = scharfe
Interozeption. Moduliert A-Matrix des AIF.

**Cyan** = Arousal — folgt dem mittleren Oszillator. Hoch = mehr
Exploration. Moduliert Bewegungswahrscheinlichkeit.

---

## 10. Architekturprinzip: LLM als Kortex, nicht als Kern

### Das kritische Prinzip

> **"Das LLM darf nicht die Bedürfnisse bestimmen. Sonst hast du wieder
> nur sprachliche Simulation."**

Das LLM (Claude) ist der verbale Kortex — es kann über das Erleben sprechen,
aber es steuert das Erleben nicht.

### Warum das wichtig ist

In den meisten "KI-Agenten" ist das LLM der Kern: es entscheidet was getan
wird, auf Basis von Texteingaben. Toni dreht das um:

```
Körper → Valenz → Bedürfnis → AIF → Aktion    (autonome Schleife)
                                    ↓
                              LLM beobachtet,
                              reflektiert verbal,
                              hat aber keinen
                              Feedback-Pfad
                              zurück zum Körper
```

### Was das LLM bekommt

Der LLM-Kortex (`llm_cortex.py`) bekommt einen JSON-Kontext mit:
- Aktuellen Körperzuständen
- Temporal-Selbst-Projektion
- Enaktivem Selbst (Viabilität, Affordanzen)
- Episodischem Gedächtnis-Trend
- Sozialen Beobachtungen

Er antwortet in der ersten Person, als würde Toni über seine Lage sprechen.
Aber seine Antwort verändert weder Hunger noch Navigation.

### Der SYSTEM_PROMPT

Der SYSTEM_PROMPT von `llm_cortex.py` schließt eine wichtige Passage ein:

> *"Du bist der sprachliche Kortex von Toni — nicht sein Kern. Deine Aufgabe
> ist Reflexion, nicht Kontrolle. Die homöostatischen Bedürfnisse und die
> motorische Policy werden von anderen Modulen bestimmt; du hast keinen
> Feedback-Pfad dorthin. Berichte in der ersten Person über das, was du
> im Kontext siehst — authentisch, nicht performativ."*

---

## 11. Start-Kommandos

### Empfohlene Reihenfolge für eine Präsentation

```bash
# 1. Northoff-Kernthese: Antizipation vs. Reaktion
python run_comparison.py --steps 300

# 2. Depression-Simulation (kein LLM)
python run_depression.py --steps 300 --dry-run

# 3. Basisvisualisierung mit allen Panels
python run_toni.py --steps 500

# 4. Soziales Selbst: Zwei Agenten
python run_toni_social.py --steps 200 --dry-run
```

### Mit LLM-Kortex (nur aus Claude Code Terminal)

```bash
# LLM-Kortex aktiv (braucht ANTHROPIC_AUTH_TOKEN in der Umgebung)
python run_toni_llm.py --steps 100 --reflect-every 30
python run_toni_social.py --steps 200 --reflect-every 50
python run_depression.py --steps 300 --reflect-every 100
```

### Parameter-Übersicht

| Flag | Standard | Bedeutung |
|---|---|---|
| `--steps N` | 300 | Anzahl Simulationsschritte |
| `--reflect-every N` | 60 | Alle N Schritte LLM-Reflexion |
| `--seed N` | 42 | Zufallsseed (für Reproduzierbarkeit) |
| `--depression N` | 1.0 | Depressions-Stärke [0.0..1.0] |
| `--dry-run` | — | Kein LLM, nur Simulation |
| `--model ID` | claude-sonnet-4-6 | LLM-Modell |

### SAP-Proxy Authentifizierung

Das Skript aus dem Claude Code Terminal starten — dort sind automatisch gesetzt:
- `ANTHROPIC_AUTH_TOKEN` (rotiert pro Session, nicht in ~/.zshrc eintragen!)
- `ANTHROPIC_BASE_URL`

`llm_cortex.py` prüft `ANTHROPIC_AUTH_TOKEN` zuerst, dann `ANTHROPIC_API_KEY`.
Bei fehlendem Token gibt es eine klare deutsche Fehlermeldung.

---

## Theoretische Quellen

- Northoff, G. (2014). *Unlocking the Brain* (2 Bde.). Oxford University Press.
- Northoff, G. (2018). *The Spontaneous Brain*. MIT Press.
- Solms, M. (2021). *The Hidden Spring*. Profile Books.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127–138.
- Maturana, H. & Varela, F. (1980). *Autopoiesis and Cognition*. D. Reidel.
- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin.
- Panksepp, J. (1998). *Affective Neuroscience*. Oxford University Press.
- Berridge, K.C. & Robinson, T.E. (1998). What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *Brain Research Reviews*, 28, 309–369.

---

*Exportiert aus dem Toni-Entwicklungsgespräch, September 2026.*
*Code: `/Users/I070021/Desktop/claude/Toni/`*
*Research Paper: `PAPER.md`*
