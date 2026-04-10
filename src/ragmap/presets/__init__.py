from __future__ import annotations

from ragmap.presets.base import Preset
from ragmap.presets.generic import GenericPreset
from ragmap.presets.haystack import HaystackPreset
from ragmap.presets.langchain import LangChainPreset
from ragmap.presets.llamaindex import LlamaIndexPreset

PRESETS: dict[str, type[Preset]] = {
    "generic": GenericPreset,
    "langchain": LangChainPreset,
    "llamaindex": LlamaIndexPreset,
    "haystack": HaystackPreset,
}


def get_preset(name: str, **kwargs) -> Preset:
    cls = PRESETS.get(name)
    if cls is None:
        available = ", ".join(PRESETS)
        raise ValueError(f"Unknown preset: {name!r}. Available: {available}")
    return cls(**kwargs)
