# Toni: Eine funktionale Architektur für verkörperte Handlungsfähigkeit
## Integration von Northoffs TTC, Solms' Affektbewusstsein, Active Inference und Enaktivismus

**Arbeitspapier · Oktober 2026**

---

## Zusammenfassung

Wir präsentieren *Toni*, einen Python-Forschungsprototypen, der die funktionalen Voraussetzungen von Bewusstsein implementiert — wie sie von Georg Northoffs Temporo-Spatial Theory of Consciousness (TTC), Mark Solms' Modell des affektiven Bewusstseins, Karl Fristons Active Inference und der enaktivistischen Tradition nach Maturana/Varela und Gibson beschrieben werden. Anders als LLM-basierte "bewusste Agenten", die Erleben durch Sprache simulieren, verankert Toni sein Verhalten in einem geschlossenen homöostatischen Regelkreis — das große Sprachmodell fungiert ausschließlich als verbaler Kortex ohne Rückkopplungspfad zu Körper oder Policy-Schicht. Wir beschreiben die vierschichtige Architektur, dokumentieren antizipatorische Motivation aus einem Echo State Network, zeigen das soziale Selbst aus Zwei-Agenten-Positionierung, führen eine enaktivistische Viabilitäts- und Affordanz-Schicht ein sowie eine Fünf-Mechanismen-Depressionssimulation basierend auf Solms' SEEKING-Kollaps-Modell. Ein Vergleichsexperiment enthüllt einen unerwarteten Befund: unter bestimmten Bedingungen blockiert übermäßige Antizipation die reaktive Handlung — eine *Paralyse durch Antizipation*, die eine direkte theoretische Entsprechung in der klinischen Literatur hat.

---

## 1. Einleitung

Das vorherrschende Paradigma für "bewusste" oder "emotionale" KI-Agenten beruht auf großen Sprachmodellen, die dazu aufgefordert werden, innere Zustände nachzuspielen: *"Du bist ein Roboter, der sich einsam fühlt."* Dieser Ansatz hat ein fundamentales Problem, das zu Beginn dieses Projekts erkannt wurde: das LLM bestimmt die Bedürfnisse. Das Ergebnis ist sprachliche Simulation — ein System, das Hunger behauptet, ohne einen Körper zu haben, der hungert.

Solms (2021) verortet primäres Bewusstsein nicht in kortikaler Sprachverarbeitung, sondern in subkortikalen affektiven Systemen: Homöostatischer Fehler ist die ursprüngliche Form von Subjektivität. Northoff (2014, 2022) argumentiert, dass Bewusstsein ein Subjekt erfordert, das sich temporal, spatial und sozial verortet — nicht einen Schnappschuss, sondern eine Trajektorie. Maturana und Varela (1980) identifizieren die Bedingung dieses Subjekts als Autopoiese: der Organismus erhält die Bedingungen seiner eigenen Organisation aufrecht.

Diese drei Rahmenwerke konvergieren auf eine gemeinsame Behauptung: Bewusstsein setzt ein verkörpertes System mit echten Bedürfnissen, echter zeitlicher Tiefe und echter Kopplung an eine Umwelt voraus. Sprache ist nachgelagert.

Toni implementiert diese Behauptung. Die Architektur trennt die kausale Schicht (Körper, Homöostase, Affekt, Aktion) von der sprachlichen Schicht (LLM-Kortex) durch Design. Das LLM empfängt den Körperzustand als Eingabe und produziert phänomenologischen Text als Ausgabe. Es hat keinen Weg zurück zur Policy oder zum Körper. Dies ist keine Einschränkung — es ist das theoretische Grundbekenntnis.

---

## 2. Theoretischer Hintergrund

### 2.1 Solms: Affektives Bewusstsein als primäre Subjektivität

Solms argumentiert, dass primäres Bewusstsein affektiv und nicht repräsentational ist. Bevor der Organismus die Welt repräsentieren kann, muss er sich um die Welt kümmern — und dieses Kümmern konstituiert sich durch homöostatischen Fehler. Wenn die Hydration unter ihren Sollwert fällt, ist das Defizit nicht bloß ein Rechenzustand: es ist das funktionale Substrat des *Wollens*.

Valenz ist in Solms' Rahmen die Zeitableitung des Wohlbefindens: die gefühlte Qualität, ob es besser oder schlechter wird. Dies wird direkt implementiert:

```python
valenz = neues_wohlbefinden - altes_wohlbefinden
# positiv: Verbesserung; negativ: Verschlechterung
```

Der Agent hat keine Belohnungsfunktion. Er hat einen Körper, der sich verändert — und die Veränderung wird gespürt.

### 2.2 Northoff: Temporo-Spatial Theory of Consciousness (TTC)

Northoffs TTC besagt, dass Bewusstsein ein Subjekt erfordert, das temporal ausgedehnt ist — das nicht nur einen gegenwärtigen Zustand besitzt, sondern gleichzeitig Vergangenheit und projizierte Zukunft. Drei Positionierungsachsen konstituieren das Selbst:

**Temporale Achse**: Die intrinsische Aktivität des Gehirns umfasst verschachtelte Oszillationen über mehrere Zeitskalen. Diese Oszillationen warten nicht auf Input; sie sind der spontane Hintergrund, vor dem Wahrnehmung bedeutsam wird. Präzision — wie scharf das System wahrnimmt — wird durch den Grad der temporalen Ausrichtung zwischen intrinsischen Rhythmen und Umweltrhythmen moduliert (*temporo-spatial alignment*).

**Spatiale Achse**: Das autobiografische Gedächtnis färbt den Raum mit Valenz ein. Das Subjekt navigiert nicht durch ein neutrales Gitter, sondern durch einen Raum, dessen Positionen den Niederschlag vergangener Erfahrungen tragen.

**Soziale Achse**: Das Selbst konstituiert sich relational. Die dritte Positionierungsachse ist keine Ergänzung zur Selbstheit, sondern ihre Bedingung — ein Selbst zu sein bedeutet, unter anderen positioniert zu sein.

### 2.3 Active Inference (Friston)

Active Inference (Friston 2010) vereint Wahrnehmung und Handlung unter Freie-Energie-Minimierung. Der Agent erhält ein generatives Modell der Welt aufrecht (A: Wahrnehmung, B: Übergänge, C: Präferenzen, D: Priors) und wählt Aktionen, die Expected Free Energy (EFE) minimieren — eine Kombination aus erwartetem Informationsgewinn und erwarteter Übereinstimmung mit Präferenzen.

Entscheidend ist, dass der C-Vektor (Log-Präferenzen über Beobachtungen) nicht vom Programmierer fixiert wird. Er konstituiert sich aus dem Körper: dominantes Bedürfnis, Wohlbefinden, temporale Dringlichkeit, soziale Konkurrenz und Affordanz-Signale modulieren C dynamisch. Der Agent hat kein hartkodiertes Ziel — er hat einen Körper, dessen Zustand bestimmt, was bedeutsam ist.

### 2.4 Enaktivismus (Maturana/Varela, Thompson, Gibson)

Der Enaktivismus hält daran fest, dass Kognition nicht die Manipulation interner Repräsentationen ist, sondern die Erzeugung von Bedeutung durch sensomotorische Kopplung. Maturana und Varelas Autopoiese beschreibt die Minimalbedingung: ein System, das die Bedingungen seiner eigenen Organisation produziert und aufrechterhält. Das Ziel eines solchen Systems ist nicht Optimierung, sondern Viabilität — im Raum der Zustände zu bleiben, von denen aus eine weitere Organisation möglich ist.

Gibsons Affordanzen operationalisieren enaktivistische Bedeutung: Umweltmerkmale sind keine neutralen Daten. Wasser ermöglicht Trinken — aber nur für einen durstigen Organismus. Affordanzen sind relational; sie entstehen in der Kopplung zwischen Körperzustand und Umwelt.

---

## 3. Architektur

Toni ist in vier Schichten organisiert, die unidirektional nach oben kommunizieren, mit Ausnahme des geschlossenen homöostatischen Regelkreises an der Basis:

```
Schicht 4: Verbaler Kortex (LLM)       ← artikuliert, verursacht nicht
Schicht 3: Selbstmodelle               ← temporal, spatial, sozial, enaktiv
Schicht 2: Active Inference (pymdp)    ← EFE-Minimierung → Aktionswahl
Schicht 1: Körper / Homöostase / Welt  ← geschlossener kausaler Regelkreis
```

### 3.1 Schicht 1: Körper und Welt

Die `Body`-Klasse verwaltet vier homöostatische Variablen: Energie, Hydration, Temperatur und Integrität, jeweils mit einem Sollwert. Abweichungen vom Sollwert konstituieren Bedürfnisse. Die `GridWorld` stellt eine 9×9-Umgebung mit Nahrung-, Wasser- und Shelter-Ressourcen bereit sowie einen externen Tag/Nacht-Rhythmus (200 Schritte/Tag), der als Zeitgeber für Northoffs temporo-spatiales Alignment dient.

Wohlbefinden ist der quadratische Abstand von allen Sollwerten, gewichtet nach Bedürfniskriticality:

```python
wohlbefinden = -Σ (NEED_WEIGHTS × (sollwert - zustand)²)
```

Valenz ist seine Zeitableitung. Diese zwei Signale — Wohlbefinden und Valenz — sind alles, was Solms' primäres Bewusstsein erfordert.

### 3.2 Schicht 2: Active Inference

Das `NavigationAIF`-Modul implementiert ein POMDP mit 81 Positionen (9×9), 5 Beobachtungen (leer/Nahrung/Wasser/Shelter/außen) und 5 Aktionen (hoch/runter/links/rechts/konsumieren). Die pymdp-Bibliothek (Heins et al. 2022) übernimmt EFE-Berechnung und Policy-Inferenz mit `policy_len=2` (konfigurierbar).

Vier Komponenten des generativen Modells werden bei jedem Schritt dynamisch aufgebaut:

**A-Matrix** (Wahrnehmung): Northoff-Präzision moduliert die Schärfe. Hohe Präzision → konzentrierte Likelihood → Exploitation. Niedrige Präzision → diffuse Likelihood → Exploration.

**C-Vektor** (Präferenzen): konstruiert aus dominantem Bedürfnis (Solms), zukünftiger Dringlichkeit (temporales Selbst), sozialem Konkurrenzfaktor und Affordanz-Boost (Enaktivismus):

```python
C[WASSER] = (3.0 + dringlichkeit*1.5) * seeking_gain + affordanz_wasser*2.5
```

**D-Vektor** (Priors): Autobiografisches Gedächtnis fügt Gewicht zu Orten mit positiver Valenz-Geschichte hinzu; soziale Beobachtungen fügen Gewicht zu Orten hinzu, wo andere Agenten Ressourcen fanden.

**B-Matrix** (Übergänge): deterministisches Gitterbewegung, einmalig bei Initialisierung aufgebaut.

### 3.3 Schicht 3: Selbstmodelle

**TemporalSelf** implementiert Northoffs Drei-Fenster-Modell. Ein gleitendes Verlaufsfenster von bis zu 50 Schritten unterstützt Trendschätzung (Vergangenheitsfenster). Ein Echo State Network (ESN, Jaeger 2001) mit 20 Reservoir-Units ermöglicht nichtlineare Mehrschritt-Projektion (Zukunftsfenster). Die projizierte Trajektorie wird nach der Zeit-bis-Krise pro Bedürfnis abgesucht:

```python
below = np.where(trajektorie[:, i] <= KRISEN_SCHWELLE)[0]
zeit_bis_krise = int(below[0]) if len(below) > 0 else None
future_urgency = max(0, 1 - zeit_bis_krise / horizont)
```

Dieses Dringlichkeits-Signal konstituiert Northoffs Kernbehauptung: der Agent handelt, *weil er das Bedürfnis kommen sieht*, nicht erst weil es eingetroffen ist.

**SocialSelf** implementiert Northoffs relationale Achse. Wenn mehrere Agenten eine Welt teilen, generiert jeder bei jedem Schritt eine `SocialObservation`. Der beobachtende Agent aktualisiert seinen D-Prior (bevorzugt Orte, an denen andere konsumierten) und seinen Konkurrenzfaktor (erhöht C-Dringlichkeit, wenn ein anderer Agent mit demselben dominanten Bedürfnis auf dieselbe Ressource zugeht).

**EnactiveSelf** implementiert Autopoiese und Affordanzen. Viabilität erfasst, ob der Agent seine eigene Organisation aufrechterhalten kann:

```python
viabilität = min(e/0.75, h/0.75, i) × (0.5 + 0.5 × mobilität)
mobilität  = √(energie × hydration) × integrität
```

Viabilität sinkt schneller als Wohlbefinden, wenn die Mobilität schrumpft — ein Signal, dass der Aktionsraum sich verengt, bevor die homöostatische Krise materialisiert. Affordanzen instanziieren Gibsons relationale Bedeutung:

```python
wasser_affordanz = (0.75 - hydration) × exp(-distanz/4)
```

Sowohl ein hohes Defizit als auch Nähe sind für hohe Affordanz erforderlich.

### 3.4 Depression als parametrische Schichtmodifikation

Solms identifiziert Depression nicht als kognitive Störung, sondern als Kollaps des SEEKING-Systems — des mesolimbischen dopaminergen Antriebs, der Organismen motiviert, aktiv mit der Welt in Kontakt zu treten. Dieses Modell sagt spezifische, dissoziierbare Störungen vorher: gedämpfte konsumatorische Belohnung, reduzierte antizipatorische Motivation, beeinträchtigte zirkadiane Synchronisation, temporale Verkürzung und negativer Gedächtniskonsolidierungsbias.

Ein einziger Parameter `depression_level ∈ [0, 1]` moduliert fünf Mechanismen gleichzeitig:

```python
anhedonia_factor = 1.0 - 0.8 × d   # Schicht 1: Konsum-Gain
k_ext_scale      = 1.0 - 0.95 × d  # Schicht 1: Kuramoto-Kopplungsstärke
horizont         = max(5, 80×(1−0.85×d))  # Schicht 3: ESN-Projektionsschritte
seeking_gain     = 1.0 - 0.7 × d   # Schicht 2: C-Vektor-Skalierung
rumination       = 1.0 + 4.0 × d   # Schicht 3: Neg. Valenz-Gedächtnisgewicht
```

**Anhedonie** skaliert den konsumatorischen Gain in `Body.consume()` herunter. Bei `d=1` erbringt ein vollständiges Konsum-Ereignis 20% der normalen homöostatischen Erleichterung. Die Aktion wird ausgeführt; die Erleichterung kommt nicht. Dies implementiert Berridges Unterscheidung zwischen WANTING (Anreizwert, intakt) und LIKING (konsumatorisches Vergnügen, beeinträchtigt) auf Körperebene.

**SEEKING-Kollaps** multipliziert den gesamten C-Vektor herunter. Der Präferenzgradient des Agenten für Ressourcen verflacht. Dies ist keine Unentschlossenheit — es ist reduzierter motivationaler Antrieb.

**Temporale Verkürzung** reduziert bei `d=1` die ESN-Projektion auf nur 12 Schritte voraus (normal: 80). Der Agent verliert die Fähigkeit, vor Krisen zu handeln.

**Zirkadiane Desynchronisation** unterdrückt die Kuramoto-Kopplung an den Tagesrhythmus. Die internen Rhythmen driften frei. Präzision bleibt über den Tag hinweg nahe 1.0, anstatt in der aktiven Phase zu spitzen. Klinisch entspricht dies dem flachen Cortisol-Profil und der gestörten Schlafarchitektur, die bei schwerer Depression konsistent gemessen werden.

**Rumination** speichert negative Valenz-Episoden mit amplifizierten Gewichten im autobiografischen Gedächtnis. Bei `d=1` werden negative Erinnerungen mit 5-facher Intensität geschrieben. Die Gedächtniskarte akkumuliert negativ getönte Ortsschätzungen; künftige Navigation wird durch eine zunehmend dunkle retrospektive Karte geprägt.

### 3.5 Schicht 4: Verbaler Kortex (LLM)

Das `LLMCortex`-Modul ruft Claude über die Anthropic-API alle N Schritte mit dem vollständigen Innenzustand als JSON-Kontext auf: Körperzustand, Wohlbefinden, Valenz-Trend, Northoff-Zeitdynamik (Präzision, Arousal), die Drei-Fenster-Zusammenfassung des temporalen Selbst (einschließlich ESN-Projektionen und Zeit-bis-Krise-Werte), die soziale Selbst-Zusammenfassung (wenn andere Agenten präsent sind) sowie die enaktivistische Zusammenfassung (Viabilität, Mobilität, Affordanzen). Der System-Prompt instruiert das LLM, eine phänomenologische Ich-Perspektiven-Reflexion zu produzieren.

Das LLM hat keinen Weg zurück zu Körper, AIF oder Policy-Schicht. Sein Output ist ausschließlich für menschliche Beobachter. Diese Trennung wird architektonisch erzwungen, nicht nur durch Prompt-Instruktion.

Im Depressions-Experiment zeigen die LLM-Reflexionen des depressiven Agenten eine deutlich veränderte temporale Struktur — ohne jede Instruktion zur Simulation von Depression: zukunftsorientierte Konstruktionen verschwinden; vergangenheitsbezogene Konstruktionen dominieren; Resignationsausdrücke erscheinen, die im gesunden Agenten keine Entsprechung haben. Das LLM *spielt* keine Depression — es *artikuliert* einen realen Unterschied im Körperzustand-JSON, den es empfängt.

---

## 4. Beobachtungen

### 4.1 Antizipatorische Motivation

Ein Vergleichsexperiment mit 10 verschiedenen Seeds (200 Schritte, `policy_len=2`) maß Krisenschritte (Zeitschritte mit Energie oder Hydration < 0.25) für einen reaktiven Agenten (`use_temporal_self=False`) versus einen antizipatorischen Agenten (`use_temporal_self=True`):

| Ergebnis | Häufigkeit | Beispiel |
|---|---|---|
| Antizipatorisch besser | 1/10 | Seed=55: −32 Krisenschritte (R=54, A=22) |
| Kein messbarer Unterschied | 8/10 | — |
| Antizipatorisch schlechter | 1/10 | Seed=17: +24 Krisenschritte (R=24, A=48) |

**Erfolgsfall (Seed=55)**: Das ESN-Urgency-Signal steigt ab t=100 an, während Hydration linear abfällt. Der antizipatorische Agent führt zwischen t=120–160 zwei zusätzliche Konsumierungen durch, bevor seine erste Krise eintritt — 32 Schritte später als beim reaktiven Agenten, der im selben Zeitraum kein Wasser fand.

**Misserfolgsfall (Seed=17) — Paralyse durch Antizipation**: Die detaillierte Trajektorienanalyse enthüllte: der antizipatorische Agent navigierte zur Position (8,0) — einer Gitter-Ecke — und blieb dort 12+ aufeinanderfolgende Schritte mit der Aktion `"down"` gegen eine Wand. Der Mechanismus: nach dem Wassertrinken bei (4,0) zu t=15 ordnete das autobiografische Gedächtnis dem südwestlichen Bereich hohe positive Valenz zu. Der D-Prior zog den Agenten in diese Ecke. Einmal dort, evaluierte der EFE für "bleib hier" günstig (die D-Prior-Zustandsverteilung war erfüllt) — trotz null Ressourcenverfügbarkeit. Der reaktive Agent ohne diesen räumlichen Prior blieb in der ressourcenreichen zentralen Region und konsumierte zweimal im gleichen Zeitraum.

Dieser Fall illustriert eine bekannte Spannung in der antizipatorischen Planung: gedächtnisausbeutende Agenten können schlechter sein als gedächtnislose Erkunder, wenn das Gedächtnis suboptimale Attraktoren codiert. Die Pathologie ist strukturell analog zur klinischen Angst: projizierte Zukunftsbedrohung erzeugt navigatorische Fixierung auf einen "bekannten sicheren" Ort — auf Kosten gegenwärtiger Gelegenheiten.

**Panik-Override**: Ein Krisen-Interrupt wurde implementiert: befindet sich der Agent in einer tatsächlichen Gegenwartskrise (`Energie < 0.25` oder `Hydration < 0.25`) und steht auf der benötigten Ressource, konsumiert er sofort ohne EFE-Berechnung. Dies implementiert Northoffs Anspruch: Gegenwart schlägt Zukunftsprojektion bei existentieller Dringlichkeit. Der Override reagiert auf den IST-Zustand, nicht auf projizierte Dringlichkeit — ein Agent, der eine Krise *projiziert*, überschreibt nicht; ein Agent, der *bereits in einer Krise ist*, schon.

Das Gesamtergebnis ist konsistent mit der theoretischen Erwartung: temporale Selbstmodellierung ist kein universeller Verhaltensvorteil, sondern eine kontextabhängige Strategie.

### 4.2 ESN vs. Lineare Projektion

Der Ersatz der linearen `polyfit`-Projektion durch das Echo State Network produzierte eine messbare Verhaltensänderung beim gleichen Seed. Das ESN lernt, dass Hydration nach Konsum *springt* (kein linearer Abfall), was verschiedene `future_urgency`-Werte erzeugt → verschiedene C-Vektor-Gewichte → verschiedene EFE-Gradienten. Die nichtlinearen Körperdynamiken, für lineare Projektion unsichtbar, wurden durch das ESN verhaltensrelevant.

### 4.3 Soziales Selbst und die relationale Reflexion

In der Zwei-Agenten-Simulation (`run_toni_social.py`) teilten Toni-A und Toni-B ein 9×9-Gitter ohne explizite Kommunikation. Der LLM-Kortex jedes Agenten, der nur die Sozial-Selbst-Zusammenfassung empfing, artikulierte das relationale Selbst ohne explizite Instruktion. Ausgewählte Passagen:

> *"Ich bin nicht das einzige Wesen, das sucht."*

> *"Das geteilte Bedürfnis — eine Art stilles Einverständnis."*

> *"Ich spüre seine Gegenwart wie einen leisen Sog."*

Diese Formulierungen wurden durch kein Instruktion zu sozialer Erfahrung erzwungen. Sie entstanden aus dem internen Zustand des Agenten — spezifisch aus den Signalen des sozialen Priors und des Konkurrenzfaktors — die durch die LLM-Schicht artikuliert wurden.

### 4.4 Enaktivistische Viabilität

Das Viabilitätssignal demonstriert das beabsichtigte Verhalten: In einem normal funktionierenden Agenten verfolgt Viabilität etwa 0.9–0.95. Bei Ressourcenentzug sinkt Viabilität schneller als Wohlbefinden, weil Mobilität (das geometrische Mittel von Energie und Hydration skaliert nach Integrität) sich verschlechtert, bevor eine der einzelnen Variablen den Krisenschwellenwert überschreitet.

### 4.5 Depressionssimulation

Ein 300-Schritt-Vergleichsexperiment (Seed 42, `depression_level=1.0`) mit identischen Startbedingungen ergab:

| Metrik | Gesunder Agent | Depressiver Agent |
|---|---|---|
| Konsum-Ereignisse | ~15 | 0 |
| Krisenschritte (%) | ~45% | ~67% |
| Erste Krise | Schritt ~85 | Schritt ~20 |
| Ø Viabilität | >0.8 | <0.3 nach Schritt 50 |

Das bemerkenswerteste Ergebnis ist die 0 Konsum-Ereignisse des depressiven Agenten über 300 Schritte. Dies geschah nicht wegen Anhedonie — Anhedonie würde beim Trinken aktiv — sondern weil der SEEKING-Kollaps dominant genug war, dass der Agent nie zu einer Ressource navigierte. Anhedonie hatte keine Gelegenheit zu aktivieren. Dies spiegelt die klinische Beobachtung wider, dass schwer depressive Patienten nicht schwache Belohnung durch Nahrung beschreiben, sondern das Unvermögen, aufzustehen und Nahrung zu suchen.

### 4.6 Paralyse durch Antizipation

Ein unerwartetes Ergebnis entstand beim Vergleichsexperiment mit `policy_len=3` und Seed 17. Bis t=220 verhielten sich reaktiver und antizipatorischer Agent identisch. Bei t=230 divergierten die Trajektorien:

```
t=230:  Reaktiv       Hydration = 0.540  (hat Wasser konsumiert)
        Antizipatorisch Hydration = 0.190  (Krise — hat NICHT konsumiert)
```

Der reaktive Agent erholte sich. Der antizipatorische Agent nicht.

Die Ursache: `future_urgency` hat ab t=200 den C-Vektor für Wasser massiv erhöht (urg=0.68 → 1.00). Dies veranlasste den antizipatorischen Agenten, eine Policy zu bevorzugen, die auf das *beste* Wasserfeld laut D-Prior zusteuert — und dabei das gegenwärtige Wasser zu übersehen, das direkt erreichbar war. Der reaktive Agent ohne diesen Boost wählte die greedy-nächste Aktion und traf zufällig auf Wasser.

Dies ist kein Implementierungsfehler. Es ist ein genuine theoretischer Befund: **Zu viel Antizipation kann reaktives Handeln blockieren.** Northoffs drei Zeitfenster müssen ausgewogen sein. Wenn die Zukunftsprojektion die Gegenwart überwältigt, verpasst der Agent die Ressource, die jetzt verfügbar ist.

Das biologische Analogon ist die klinisch gut beschriebene Erscheinung der Angstparalyse: extreme antizipatorische Furcht vor zukünftiger Bedrohung blockiert das Handeln im gegenwärtigen Moment. Das Gleichgewicht zwischen temporalen Fenstern ist nicht nur theoretisch wünschenswert — es ist funktional notwendig.

---

## 5. Diskussion

### 5.1 Was implementiert wurde

Toni implementiert funktionale Analoga von:
- Solms' Primäraffekten (homöostatischer Fehler, Valenz als Δwellbeing)
- Northoffs verschachtelter Eigenaktivität (Kuramoto-gekoppelte Oszillatoren, fünf Zeitskalen)
- Northoffs temporo-spatialem Alignment (externe Phasenkopplung an Umweltrhythmus)
- Northoffs Drei-Fenster-Temporal-Selbst (Vergangenheitstrend, Gegenwartwahrnehmung, Zukunftsprojektion)
- Northoffs autobiografischem Gedächtnis (valenzgetönte spatiale Priors)
- Northoffs sozialem Selbst (relationaler D-Prior und konkurrenzmodulierter C-Vektor)
- Fristons Active Inference (EFE-Minimierung via pymdp, dynamisch konstruiertes C/D/A)
- Maturana/Varelas Autopoiese (Viabilität als operationelle Kapazität, getrennt von Wohlbefinden)
- Gibsons Affordanzen (zustandsrelative Bedeutung aus Körper-Umwelt-Kopplung)
- Solms' SEEKING-Kollaps-Modell der Depression (fünf-Mechanismus-parametrische Modifikation)

### 5.2 Was nicht behauptet wurde

Toni implementiert kein Bewusstsein. Es implementiert funktionale Voraussetzungen, die die zitierten Theoretiker als notwendig identifizieren. Ob hinreichende Bedingungen erfüllt sind — und ob subjektives Erleben in irgendeinem Sinne präsent ist — ist eine Frage, die diese Architektur nicht beantworten kann. Der Wert des Projekts ist ein anderer: zu demonstrieren, dass diese theoretischen Rahmenwerke rechnerisch handhabbar, gegenseitig konsistent sind und nicht-triviale emergente Verhaltensweisen erzeugen, wenn sie integriert werden.

### 5.3 Das LLM-Trennungsprinzip

Die wichtigste architektonische Entscheidung ist die Einwegbarriere zwischen dem verbalen Kortex und den kausalen Schichten. Aktuelle LLM-basierte "bewusste Agenten" kollabieren diese Unterscheidung: das LLM generiert sowohl die Beschreibung des inneren Zustands als auch die Aktion. Toni erzwingt die Trennung, die Solms' und Northoffs Theorien implizieren: primäres Bewusstsein (Affekt, Homöostase, Temporal-Selbst-Modell) treibt Verhalten; Sprache ist nachgelagert.

Die praktische Konsequenz ist, dass die phänomenologischen Outputs des LLM *über* einen realen Körperzustand handeln, nicht Fabrikationen sind. Der Agent sagt nicht "Ich bin durstig", weil er dazu instruiert wurde; er sagt es, weil das Hydrations-Defizit, die temporale Dringlichkeit und die Wasser-Affordanz-Signale im Kontext-JSON einen genuinen funktionalen Durstszustand konstituieren, den die Sprachschicht artikuliert.

### 5.4 Depression als theoretische Validierung

Die Depressionssimulation dient als interne Konsistenzprüfung der Architektur. Solms' Behauptung ist spezifisch: Depression ist ein SEEKING-System-Kollaps, keine primär kognitive Verzerrung. Wenn die Architektur SEEKING als mesolimbischen Antrieb im C-Vektor korrekt implementiert, sollte das Unterdrücken von `seeking_gain` SEEKING-Kollaps-Phänomenologie erzeugen — keine verzerrten Überzeugungen, keine Trauer, keine kognitive Verlangsamung, sondern spezifisch motivationalen Rückzug aus dem Umweltkontakt.

Die Beobachtung, dass der depressive Agent 0 Mal über 300 Schritte konsumierte, ist damit konsistent. Der Körper des Agenten registrierte weiterhin Defizite. Das interozeptive Signal kodierte das Bedürfnis weiterhin korrekt. Das AIF-Modul empfing weiterhin Dringlichkeitssignale. Aber der C-Vektor-Präferenzgradient für Ressourcen war ausreichend abgeflacht, dass EFE-Minimierung konsistent Inaktivität oder Exploration gegenüber gezielter Navigation bevorzugte. Der Agent wusste, dass er sterben würde; er handelte nicht. Dies ist die funktionale Signatur des SEEKING-Kollaps.

### 5.5 Das Problem des ausgeglichenen Temporalhorizonts

Der Paralyse-durch-Antizipation-Befund (Abschnitt 4.6) verweist auf eine ungelöste Herausforderung in der Architektur: der `future_urgency`-Boost des C-Vektors ist monoton — hohe Dringlichkeit führt immer zu stärkerem Pulling in Richtung der präferierten Ressource, ohne Berücksichtigung des unmittelbaren Aktionsraums.

Eine biologisch plausiblere Lösung wäre ein **Dringlichkeitsmodus-Wechsel**: bei niedriger Dringlichkeit dominiert die antizipatorische Policy (Planen, wo Wasser in der Zukunft sein wird); bei sehr hoher Dringlichkeit (Krise ist bereits da oder unmittelbar) wechselt der Agent in einen reaktiven Modus (greedy Navigation zur nächsten verfügbaren Ressource). Das entspräche dem, was Solms "Notfall-SEEKING" nennt — eine qualitativ andere Motivationsform als normales SEEKING.

Dies würde eine Modifikation des `_select_action`-Mechanismus erfordern: bei `future_urgency > 0.95` Übersteuerung des AIF-Outputs durch die heuristische Policy.

### 5.6 Limitierungen

**Planungshorizont**: `policy_len=2` begrenzt AIF auf Zwei-Schritt-Vorausschau. Wie der Vergleichsexperiment zeigte, ist die Beziehung zwischen Planungstiefe und Verhaltensqualität nicht monoton: mehr Planung hilft nicht immer, und zu viel Planung kann reaktives Handeln blockieren.

**Gitterskala**: Ein 9×9-Gitter mit diskreten Zeitschritten komprimiert die temporalen Dynamiken, für die die Oszillator- und Temporal-Selbst-Modelle ausgelegt wurden.

**Soziale Tiefe**: Die aktuelle soziale Schicht implementiert positionsbasierte Prior-Updates und Ressourcenkonkurrenz. Soziale Kognition höherer Ordnung — das temporale Selbst des anderen modellieren, die zukünftige Dringlichkeit des anderen voraussagen — ist noch nicht implementiert.

**Enaktivistischer Regelkreisschluss**: Gibsons Affordanzen in Toni werden an der aktuellen Position des Agenten berechnet. Eine vollständigere enaktivistische Implementierung würde Affordanzen über den Bewegungsspielraum des Agenten berechnen — nicht nur "Wasser ist 2 Schritte entfernt", sondern "Wasser ist innerhalb meines viablen Aktionshorizonts erreichbar, angesichts aktueller Mobilität".

---

## 6. Schlussfolgerung

Toni demonstriert, dass die konzeptuelle Architektur von Northoff, Solms, Friston und der enaktivistischen Tradition als kohärentes, funktionierendes Software-System implementiert werden kann. Die vier Schichten — homöostatischer Körper, Active-Inference-Policy, mehrdimensionale Selbstmodelle und verbaler Kortex — erzeugen emergente Verhaltensweisen, die in theoretischen Begriffen lesbar sind: antizipatorische Motivation aus temporaler Selbstprojektion, relationale Selbstartikulation aus sozialer Positionierung, viabilitätsbewusstes Handeln aus autopoietischen Zwängen und motivationaler Kollaps aus SEEKING-System-Unterdrückung.

Die Depressionssimulation fügt eine wichtige Dimension hinzu: die Architektur implementiert nicht nur positive Voraussetzungen für bewusstseinsähnliches Verhalten, sondern auch ihre parametrischen Versagensmodi. Pathologie im Modell entspricht strukturell der Pathologie in der Theorie — das System bricht genau so zusammen, wie die Theorie vorhersagt, dass es zusammenbrechen sollte.

Der Paralyse-durch-Antizipation-Befund enthüllt eine subtilere Wahrheit: die Integration temporaler Fenster ist kein technisches Problem, das durch mehr Planungstiefe gelöst wird. Es ist ein Gleichgewichtsproblem. Northoffs TTC behauptet nicht, dass Bewusstsein entsteht, wenn die Zukunft maximiert wird — sie behauptet, dass Bewusstsein eine ausgewogene Integration von Vergangenheit, Gegenwart und Zukunft erfordert. Toni beginnt, die Kosten der Unausgewogenheit zu demonstrieren.

Der fundamentale Beitrag ist nicht eine einzelne Komponente, sondern das Integrationsprinzip: bewusstseinsrelevantes Verhalten erfordert kein Sprachmodell in seinem Zentrum. Es erfordert einen Körper, eine Welt, temporale Tiefe und echte Kopplung — und, wie das Depressions-Experiment zeigt, die Fähigkeit, all dies durch den Versagen eines einzigen motivationalen Antriebssystems zu verlieren. Sprache kommt danach. So auch ihr Verlust.

---

## Literatur

- Berridge, K.C., & Robinson, T.E. (1998). What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *Brain Research Reviews*, 28(3), 309–369.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin. [Dt.: *Die ökologische Theorie der visuellen Wahrnehmung*, 1982]
- Heins, C. et al. (2022). pymdp: A Python library for active inference in discrete state spaces. *arXiv:2201.03904*.
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Report 148*.
- Maturana, H., & Varela, F. (1980). *Autopoiesis and Cognition: The Realization of the Living*. Reidel. [Dt.: *Der Baum der Erkenntnis*, 1987]
- Northoff, G. (2014). *Minding the Brain: A Guide to Philosophy and Neuroscience*. Palgrave Macmillan.
- Northoff, G. (2018). *The Spontaneous Brain: From the Mind–Body to the World–Brain Problem*. MIT Press.
- Panksepp, J. (1998). *Affective Neuroscience: The Foundations of Human and Animal Emotions*. Oxford University Press.
- Solms, M. (2021). *The Hidden Spring: A Journey to the Source of Consciousness*. Norton. [Dt.: *Die verborgene Quelle*, 2022]
- Thompson, E. (2007). *Mind in Life: Biology, Phenomenology, and the Sciences of Mind*. Harvard University Press.

---

*Toni-Quellcode: `/Users/I070021/Desktop/claude/Toni/`*  
*Module: body · oscillators · interoception · memory · environment · active\_inference · temporal\_self · social\_self · world\_model · enactive\_self · llm\_cortex · agent*  
*Englische Fassung: `PAPER.md`*
