from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

class Rectangle(Strict):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

class ExtractionSettings(Strict):
    item: str = Field(min_length=1, max_length=500)
    region_hint_mm: Rectangle | None = None
    take: Literal["one", "all"] = "one"
    instruction: str = Field(default="", max_length=16000)

    @field_validator("item")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("項目名が空です。")
        return value

class ExecutionSettings(Strict):
    input_folder: str = Field(default="input", min_length=1)
    llm_backend: Literal["gemma", "saluki"] = "gemma"
    ollama_profile: Literal["gemma4-q8", "gemma3-4b"] = "gemma4-q8"
    saluki_thinking: bool = False
    saluki_context_size: int = Field(default=32768, ge=8192, le=131072)
    saluki_max_tokens: int | None = Field(default=None, ge=1, le=32768)
    saluki_image_max_tokens: int | None = Field(default=None, ge=256, le=16384)
    saluki_batch_size: int = Field(default=512, ge=32, le=4096)
    saluki_micro_batch_size: int = Field(default=256, ge=32, le=4096)
    saluki_kv_cache_type: Literal["f16", "q8_0"] = "f16"

    @model_validator(mode="after")
    def compatible_saluki_profile(self):
        if self.saluki_micro_batch_size > self.saluki_batch_size:
            raise ValueError("saluki_micro_batch_sizeはsaluki_batch_size以下です。")
        maximum = self.saluki_max_tokens or (16384 if self.saluki_thinking else 8192)
        if self.llm_backend == "saluki" and maximum > self.saluki_context_size:
            raise ValueError("Salukiの生成上限はコンテキストサイズ以下にしてください。")
        return self
