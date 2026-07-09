from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union, Any
from enum import Enum
from ai_pipeline_toolbox.registry.generated_enums import Provider, Category

class DynamicModel(BaseModel):
    """
    Represents a dynamic model (e.g., LoRA) that is not part of the static registry.
    """
    model_config = {"frozen": True}
    
    url: str = Field(..., description="The direct URL to download the model.")
    provider: Provider = Field(default=Provider.DIRECT_URL, description="The provider (e.g., huggingface, civitai, direct_url).")
    category: Category = Field(default=Category.DYNAMIC, description="The category/subdirectory where the model will be saved. Can be an Enum class.")
    filename: Optional[str] = Field(default=None, description="The preferred filename. If None, it will be extracted from the URL.")

    @field_validator("provider", mode="before")
    @classmethod
    def parse_provider(cls, v):
        if isinstance(v, Enum):
            return v.value
        return v

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, v):
        if isinstance(v, type) and issubclass(v, Enum):
            return v.__name__
        return str(v)
