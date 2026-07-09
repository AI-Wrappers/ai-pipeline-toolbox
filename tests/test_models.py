import pytest
from enum import Enum
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.registry.generated_enums import Provider, Category, Checkpoints

def test_dynamic_model_validators():
    # Test string parsing for provider and category
    model = DynamicModel(url="http://x", provider="huggingface", category="DiT")
    assert model.provider == Provider.HUGGINGFACE
    assert model.category == Category.DIT
    
    # Test Enum parsing for provider and category
    model2 = DynamicModel(url="http://x", provider=Provider.CIVITAI, category=Checkpoints)
    assert model2.provider == Provider.CIVITAI
    assert model2.category == Category.CHECKPOINTS
