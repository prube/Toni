"""
Toni - Ein KI-Prototyp nach Northoff (Temporo-Spatial Theory of Consciousness),
Solms (Affective Consciousness / Homöostase) und Enaktivismus (Maturana/Varela).

Vier-Schichten-Architektur:
  Enaktivismus   → Viabilität, Affordanzen, Autopoiese (Maturana/Varela + Gibson)
  Solms          → Körper, homöostatischer Fehler, Valenz (Affective Consciousness)
  Northoff       → Zeitdynamik, räumliche + soziale Positionierung (TTC)
  Active Inf.    → EFE-Minimierung, policy_len=2 (Friston's Free Energy Principle)

Module:
  Body         → interne Zustände (Energie, Hydration, Temperatur, Integrität)
  Valence      → Solms: Wellbeing-Änderung als Valenz (gut/schlecht für dieses System)
  Interoception → verrauschte Wahrnehmung des eigenen Körperzustands
  Oscillators  → Northoff: verschachtelte intrinsische Zeitdynamik (nestedness)
  Alignment    → Northoff: Kopplung an Umweltzeitrhythmen (Kuramoto)
  Memory       → autobiografisches Gedächtnis (vergangene Erfahrungen formen Gegenwart)
  TemporalSelf → Northoff drei Zeitfenster: Vergangenheit/Gegenwart/Zukunft (ESN-Weltmodell)
  SocialSelf   → Northoff soziale Achse: relationale Positionierung
  EnactiveSelf → Maturana/Varela Autopoiese + Gibson Affordanzen
  Environment  → GridWorld mit Ressourcen und Tagesrhythmus
  Agent (Toni) → alles integriert, Policy via vereinfachter Active Inference
"""
from .body import Body
from .oscillators import TemporalDynamics
from .interoception import interoceptive_sample
from .memory import AutobiographicalMemory
from .environment import GridWorld
from .agent import Toni
from .llm_cortex import LLMCortex
from .temporal_self import TemporalSelf
from .social_self import SocialSelf, SocialObservation
from .world_model import WorldModelRNN
from .enactive_self import EnactiveSelf

__all__ = ["Body", "TemporalDynamics", "interoceptive_sample",
           "AutobiographicalMemory", "GridWorld", "Toni",
           "LLMCortex", "TemporalSelf", "SocialSelf", "SocialObservation",
           "WorldModelRNN", "EnactiveSelf"]
