"""Synthesis engines.

Importing this package registers every engine. Registration is deliberately
cheap - no model is touched until an engine is constructed and loaded - so
the registry can be listed on a machine where nothing is installed.
"""

from .asr import ASREngine, Transcript
from .base import (
    Engine,
    EngineCapabilities,
    SynthesisResult,
    TTSEngine,
    VCEngine,
    engine_status,
    get_tts_engine,
    get_vc_engine,
    list_tts_engines,
    list_vc_engines,
    register_tts,
    register_vc,
)

# Side-effect imports: each module registers its engine class.
from .chatterbox_tts import ChatterboxEngine  # noqa: F401  (registers "chatterbox")
from .f5_tts import F5TTSEngine  # noqa: F401  (registers "f5")
from .fish_tts import FishS2Engine  # noqa: F401  (registers "fish")
from .higgs_tts import HiggsTTSEngine  # noqa: F401  (registers "higgs")
from .moss15_tts import Moss15Engine  # noqa: F401  (registers "moss15")
from .moss_tts import MossTTSEngine  # noqa: F401  (registers "moss")
from .omnivoice_ipa import OmniVoiceIPAEngine  # noqa: F401  (registers "omnivoice_ipa")
from .omnivoice_tts import OmniVoiceEngine  # noqa: F401  (registers "omnivoice")
from .pocket_tts import PocketTTSEngine  # noqa: F401  (registers "pocket")
from .vc_seedvc import SeedVCEngine  # noqa: F401  (registers "seedvc")

__all__ = [
    "ASREngine",
    "ChatterboxEngine",
    "Engine",
    "EngineCapabilities",
    "F5TTSEngine",
    "FishS2Engine",
    "HiggsTTSEngine",
    "Moss15Engine",
    "MossTTSEngine",
    "OmniVoiceEngine",
    "OmniVoiceIPAEngine",
    "PocketTTSEngine",
    "SeedVCEngine",
    "SynthesisResult",
    "TTSEngine",
    "Transcript",
    "VCEngine",
    "engine_status",
    "get_tts_engine",
    "get_vc_engine",
    "list_tts_engines",
    "list_vc_engines",
    "register_tts",
    "register_vc",
]
