"""Small, immutable request profiles for the two supported local Ollama models."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


DEFAULT_PROFILE_ID = "gemma4-q8"
GEMMA4_DIGEST = "3721c8128dc2e6cabed512039c22f63629c0ad6786b0fc852a9e4da47a6953f3"
GEMMA3_REQUEST_ADAPTER = "gemma3-field-suffix-v1"
GEMMA4_HF_DIGEST = "398824a9fcc9dec0fb931a11500a0193165723235b5033d32ac6c3144e380a4d"
GEMMA3_HF_DIGEST = "29357941a108662ddbf89cb3149593e57feeae56923cf5e529c70e83fa5bb978"


@dataclass(frozen=True)
class OllamaProfile:
    id: str
    label: str
    model: str
    options: Mapping[str, int | float]
    think: bool | None = None
    expected_digest: str | None = None
    request_adapter: str | None = None
    download_size_bytes: int = 0
    fallback_digest: str | None = None

    def __post_init__(self):
        # Freezing the dataclass alone would still permit mutation of a dict.
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def snapshot(self) -> dict:
        result = {"id": self.id, "model": self.model, "options": dict(self.options),
                  "think": self.think, "expected_digest": self.expected_digest}
        if self.request_adapter is not None:
            result["request_adapter"] = self.request_adapter
        if self.fallback_digest is not None:
            result["fallback_digest"] = self.fallback_digest
        return result


_COMMON = {"temperature": 0, "top_p": 1, "top_k": 64, "repeat_penalty": 1,
           "presence_penalty": 0, "seed": 42, "num_ctx": 32768,
           "num_predict": 16384, "draft_num_predict": 0}
_PROFILES = MappingProxyType({
    DEFAULT_PROFILE_ID: OllamaProfile(DEFAULT_PROFILE_ID, "Gemma4 E4B Q8", "gemma4:e4b-it-q8_0",
                                     _COMMON, think=True, expected_digest=GEMMA4_DIGEST,
                                     download_size_bytes=9_100_000_000, fallback_digest=GEMMA4_HF_DIGEST),
    "gemma3-4b": OllamaProfile("gemma3-4b", "Gemma3 4B", "gemma3:4b", _COMMON,
                              request_adapter=GEMMA3_REQUEST_ADAPTER, download_size_bytes=3_500_000_000,
                              fallback_digest=GEMMA3_HF_DIGEST),
})


def get_profile(profile_id: str | None = None) -> OllamaProfile:
    key = DEFAULT_PROFILE_ID if profile_id is None else profile_id
    try:
        return _PROFILES[key]
    except (KeyError, TypeError) as exc:
        raise ValueError("対応するOllamaモデル設定がありません: " + str(key)) from exc


def list_profiles() -> tuple[OllamaProfile, ...]:
    return tuple(_PROFILES.values())


def resolve_profile(profile: str | OllamaProfile | None = None) -> OllamaProfile:
    if isinstance(profile, OllamaProfile):
        return profile
    return get_profile(profile)
