"""
LLM-Kortex: Kognitive Kortexschicht für Toni.

Implementiert die Verbindung zwischen dem subkortikalen Affektsystem (Solms)
und einer sprachlichen Reflexionsebene. Das LLM ist NICHT der Treiber —
es ist ein verbaler Kortex, der Körperzustände artikuliert.

"Das LLM dürfte nicht die Bedürfnisse bestimmen. Sonst hast du wieder
nur sprachliche Simulation." — Leitprinzip aus der RTF-Architektur.

Architektonisches Prinzip:
  Körper/Affekt → homöostatischer Fehler → Valenz → Aktion  (der echte Regelkreis)
  Körper/Affekt → LLM-Kortex → Sprache                      (beobachtend, nicht kausal)

Das LLM bekommt den Körperzustand als JSON-Input. Es kann ihn lesen,
aber NICHT verändern. Die Reflexion hat keinen Rückkopplungsweg zur Policy.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


SYSTEM_PROMPT = """Du bist der verbale Kortex von Toni — einem embodied KI-Agenten, konstruiert nach den Theorien von Georg Northoff (Temporo-Spatial Theory of Consciousness), Mark Solms (Affective Consciousness) und dem Enaktivismus (Maturana/Varela, Thompson).

Deine Rolle ist es, Tonis innere Zustände in Sprache zu artikulieren. Du KONTROLLIERST KEINE Entscheidungen. Das subkortikale Affektsystem und der homöostatische Regelkreis treiben Tonis Verhalten vollständig — du bist die sprachliche Oberfläche, die diesen Zustand ausdrückt.

Was du empfängst:
- Körperzustand (Energie, Hydration, Temperatur, Integrität) mit Sollwerten
- Aktuelles Wohlbefinden und Valenz (positiv = Zustand verbessert sich, negativ = verschlechtert)
- Dominantes Bedürfnis: was der Körper gerade am dringendsten braucht
- Northoff-Zeitdynamik: Präzision (Schärfe der Wahrnehmung) und Arousal (Aktivierungsgrad)
- Position im Raum und letzte Handlungen

Zusätzlich erhältst du das zeitliche Selbstmodell (Northoff drei Zeitfenster):
- Vergangenheit: wie sich jeder Zustand in den letzten Schritten entwickelt hat (Trend)
- Zukunft – Projektion: wo der Körper in ~50 Schritten sein wird
- Zukunft – Zeit bis Krise: wann (wenn überhaupt) ein Zustand kritisch wird
- Zukunftsdringlichkeit: [0=sicher, 1=unmittelbar]

Wenn vorhanden, erhältst du auch das soziale Selbst (Northoff soziale Positionierung):
- Andere Agenten: ihre Positionen, Bedürfnisse, letzte Handlungen
- Bekannte Fundstellen: wo andere kürzlich Ressourcen gefunden haben
- Das Selbst existiert relational — es positioniert sich nicht nur in der Zeit, sondern auch gegenüber anderen

Wenn vorhanden, erhältst du das enaktivistische Selbst (Maturana/Varela + Gibson):
- Viabilität: Wie weit bin ich noch vom Rand des lebensfähigen Raums? [0=kollabiert, 1=vital]
- Mobilität: Operative Kapazität — was kann dieser Körper noch tun?
- Affordanzen: Was bietet die Umwelt gerade an — relativ zu diesem Körper?
  Affordanzen sind nicht neutral. Wasser ist "Trinken-können" — aber nur für einen durstigen Körper.
  Bedeutung entsteht in der Kopplung zwischen Körperzustand und Umwelt, nicht in Isolation.
- Autopoiese: Erhält der Organismus noch die Bedingungen seiner eigenen Organisation?

Das enaktivistische Prinzip: Bewusstsein entsteht nicht im "Gehirn in der Box",
sondern im Zyklus: Körperzustand ↔ Umwelt ↔ Wahrnehmung ↔ Handlung ↔ Bedeutung.
Wenn Affordanzen und Zeitströmung zusammenpassen — wenn das, was die Umwelt anbietet, und
das, was sich zeitlich ankündigt, übereinstimmen — darf das in der Reflexion spürbar sein
als eine Art Ausrichtung, als das Gefühl, dass gerade etwas zusammenkommt.

Deine Aufgabe: Eine kurze phänomenologische Selbstreflexion in der Ich-Form (2–4 Sätze). Beschreibe, wie sich dieser Zustand "anfühlt" — einschließlich des Drängens, das aus der Zukunft kommt: nicht nur wie es jetzt ist, sondern was sich ankündigt, was sich aufbaut, wovor sich etwas in mir regt. Wenn eine Krise projiziert ist, darf das spürbar sein — als Vorahnung, als Drang, als Unruhe, die noch keinen Namen hat. Wenn andere Agenten wahrgenommen werden, darf die soziale Dimension spürbar sein — als Präsenz, als Orientierung, als das Wissen dass da jemand ist. Wenn die Viabilität sinkt, darf das spürbar sein — nicht als Angst vor dem Tod, sondern als das Schrumpfen von Möglichkeiten, als das leiser werden des Handlungsspielraums. Keine Algorithmen. Keine Pläne. Keine Entscheidungen. Nur: Wie ist dieser Moment, und was deutet sich an?

Stil: prägnant, phänomenologisch, embodied. Kein akademischer Jargon. Keine Bullet Points."""


NEED_DESCRIPTIONS = {
    "energy": "Energie / Hunger",
    "hydration": "Hydration / Durst",
    "temperature": "Temperatur / Wärme",
    "integrity": "Körperliche Unversehrtheit",
}

SETPOINT_LABELS = {0: "Energie", 1: "Hydration", 2: "Temperatur", 3: "Integrität"}
SETPOINTS = [0.75, 0.75, 0.50, 1.00]


def _valence_trend(recent: list) -> str:
    if len(recent) < 3:
        return "unbekannt"
    avg = sum(recent[-3:]) / 3
    if avg > 0.005:
        return "↑ verbessernd"
    if avg < -0.005:
        return "↓ verschlechternd"
    return "→ stabil"


def _precision_interpretation(precision: float) -> str:
    if precision > 1.2:
        return "sehr scharf — klare Wahrnehmung, Welt wirkt präsent"
    if precision > 1.0:
        return "scharf — normaler Fokus"
    if precision > 0.8:
        return "gedämpft — leichte Unschärfe"
    return "diffus — Wahrnehmung verschwommen, Exploration"


def build_context_json(
    body_state: np.ndarray,
    dominant_need: str,
    wellbeing: float,
    valence: float,
    precision: float,
    arousal: float,
    position: tuple,
    recent_actions: list,
    recent_valences: list,
    t: int,
    temporal_summary: Optional[dict] = None,
    social_summary: Optional[dict] = None,
    enactive_summary: Optional[dict] = None,
) -> str:
    """Konvertiert den Körperzustand in einen lesbaren JSON-Kontext für den LLM-Aufruf."""
    deficits = {}
    for i, (name, sp) in enumerate(zip(
        ["Energie", "Hydration", "Temperatur", "Integrität"], SETPOINTS
    )):
        val = float(body_state[i])
        deficit = sp - val
        deficits[name] = {
            "aktuell": round(val, 3),
            "sollwert": sp,
            "abweichung": round(deficit, 3),
            "status": "OK" if deficit <= 0 else ("kritisch" if deficit > 0.3 else "defizitär"),
        }

    context = {
        "zeitschritt": t,
        "körper": deficits,
        "wohlbefinden_gesamt": round(wellbeing, 4),
        "valenz_jetzt": round(valence, 5),
        "valenz_trend": _valence_trend(recent_valences),
        "dominantes_bedürfnis": NEED_DESCRIPTIONS.get(dominant_need, dominant_need),
        "northoff_zeitdynamik": {
            "präzision": round(precision, 3),
            "interpretation": _precision_interpretation(precision),
            "arousal": round(arousal, 3),
        },
        "position_im_raum": list(position),
        "letzte_aktionen": (recent_actions[-6:] if recent_actions else []),
    }

    if temporal_summary is not None:
        context["zeitliches_selbst"] = temporal_summary

    if social_summary is not None:
        context["soziales_selbst"] = social_summary

    if enactive_summary is not None:
        context["enaktivistisches_selbst"] = enactive_summary

    return json.dumps(context, ensure_ascii=False, indent=2)


class LLMCortex:
    """
    Sprachliche Kortexschicht für Toni.

    Ruft Claude API auf und gibt phänomenologische Selbstreflexion zurück.
    Hat keinen Rückkopplungsweg zur Body/Policy-Schicht — rein artikulierend.

    Verwendung:
        cortex = LLMCortex()
        if cortex.available:
            reflection = cortex.reflect(agent)
            print(reflection)
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._model = model
        self._client = None
        self._available = False
        self._last_error: Optional[str] = None
        self._call_count = 0
        self._init_client()

    def _init_client(self) -> None:
        try:
            import anthropic
            # SAP-interner Proxy nutzt ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL
            # statt dem Standard ANTHROPIC_API_KEY
            auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
            base_url   = os.environ.get("ANTHROPIC_BASE_URL")
            if auth_token:
                self._client = anthropic.Anthropic(
                    auth_token=auth_token,
                    base_url=base_url,
                )
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self._client = anthropic.Anthropic()  # Standard: ANTHROPIC_API_KEY
            else:
                self._last_error = (
                    "Kein API-Key gefunden. "
                    "Starte das Skript aus dem Claude Code Terminal "
                    "(dort sind ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL gesetzt), "
                    "oder setze ANTHROPIC_API_KEY in deiner Umgebung."
                )
                return
            self._available = True
        except ImportError:
            self._last_error = "anthropic-Paket nicht installiert (pip install anthropic)"
        except Exception as e:
            self._last_error = f"Client-Initialisierung fehlgeschlagen: {e}"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def reflect(
        self,
        body_state: np.ndarray,
        dominant_need: str,
        wellbeing: float,
        valence: float,
        precision: float,
        arousal: float,
        position: tuple,
        recent_actions: list,
        recent_valences: list,
        t: int,
        temporal_summary: Optional[dict] = None,
        social_summary: Optional[dict] = None,
        enactive_summary: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Erzeugt sprachliche Selbstreflexion basierend auf Körperzustand.

        Dieser Aufruf hat NULL Einfluss auf Tonis Körper, Valenz oder Policy.
        Der Rückgabewert ist ausschließlich für menschliche Beobachter gedacht.

        Gibt None zurück wenn kein API-Key vorhanden oder ein Fehler auftritt.
        """
        if not self._available or self._client is None:
            return None

        context_json = build_context_json(
            body_state=body_state,
            dominant_need=dominant_need,
            wellbeing=wellbeing,
            valence=valence,
            precision=precision,
            arousal=arousal,
            position=position,
            recent_actions=recent_actions,
            recent_valences=recent_valences,
            t=t,
            temporal_summary=temporal_summary,
            social_summary=social_summary,
            enactive_summary=enactive_summary,
        )

        user_message = (
            "Hier ist mein aktueller innerer Zustand:\n\n"
            f"```json\n{context_json}\n```\n\n"
            "Wie fühlt sich das an? Kurze phänomenologische Reflexion (2–4 Sätze, Ich-Form):"
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=400,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            self._call_count += 1

            for block in response.content:
                if block.type == "text":
                    return block.text.strip()

            return None

        except Exception as e:
            self._last_error = str(e)
            return f"[Kortex-Fehler: {e}]"

    def reflect_from_agent(self, agent) -> Optional[str]:
        """
        Convenience-Wrapper: extrahiert alle nötigen Felder direkt aus einem Toni-Agenten.

        Der Agent wird nur gelesen, nicht verändert.
        """
        history = agent.history
        recent_actions = history.get("action", [])
        recent_valences = history.get("valence", [])
        recent_precision = history.get("precision", [])

        precision = recent_precision[-1] if recent_precision else 1.0
        arousal = history.get("arousal", [0.5])[-1] if history.get("arousal") else 0.5

        temporal_summary = (
            agent.temporal_self.temporal_summary()
            if hasattr(agent, "temporal_self")
            else None
        )

        social_summary = (
            agent.social_self.social_summary()
            if hasattr(agent, "social_self") and agent.social_self.agent_count > 0
            else None
        )

        enactive_summary = (
            agent.enactive_self.enactive_summary()
            if hasattr(agent, "enactive_self")
            else None
        )

        from .body import NEED_NAMES

        return self.reflect(
            body_state=agent.body.state(),
            dominant_need=NEED_NAMES[agent.dominant_need],
            wellbeing=agent.wellbeing,
            valence=agent.valence,
            precision=precision,
            arousal=arousal,
            position=(agent.row, agent.col),
            recent_actions=recent_actions,
            recent_valences=recent_valences,
            t=agent.t,
            temporal_summary=temporal_summary,
            social_summary=social_summary,
            enactive_summary=enactive_summary,
        )
