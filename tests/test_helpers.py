import pytest
from ai_pipeline_toolbox.core.helpers import resolve_air_urn

def test_resolve_air_urn():
    # Direct URL
    assert resolve_air_urn("https://example.com/model") == "https://example.com/model"
    
    # Civitai AIR with version
    assert resolve_air_urn("urn:air:sd1:lora:civitai:12345@67890") == "https://civitai.com/api/download/models/67890"
    
    # Civitai AIR without version
    assert resolve_air_urn("urn:air:sd1:lora:civitai:12345") == "https://civitai.com/api/download/models/12345"
    
    # Civitai AIR simple
    assert resolve_air_urn("urn:air:civitai:lora:67890") == "https://civitai.com/api/download/models/67890"
    
    # Unknown format fallback
    assert resolve_air_urn("some_weird_string") == "some_weird_string"
