import pytest
from enum import Enum
from unittest.mock import patch, MagicMock
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.core.models import DynamicModel

from ai_pipeline_toolbox.registry.generated_enums import Checkpoints, Vae, TextEncoders

@pytest.fixture
def mock_aria2():
    with patch('ai_pipeline_toolbox.components.model_fetcher.aria2p.API') as mock_api:
        with patch('ai_pipeline_toolbox.components.model_fetcher.aria2p.Client'):
            with patch('ai_pipeline_toolbox.components.model_fetcher.subprocess.Popen'):
                with patch('ai_pipeline_toolbox.components.model_fetcher.socket.socket'):
                    yield mock_api

@pytest.fixture
def fetcher(tmp_path, mock_aria2):
    """Create ModelFetcher with mocked aria2p."""
    return ModelFetcher(cache_dir=str(tmp_path / "models"), hf_token='mock_hf', civitai_token='mock_civitai')

def test_fetcher_enum_models(fetcher, mock_aria2):
    """Test fetching from Enum models with specific providers."""
    mock_api_instance = mock_aria2.return_value
    mock_dl = MagicMock()
    mock_dl.gid = "123"
    mock_dl.is_complete = True
    mock_api_instance.add_uris.return_value = mock_dl
    mock_api_instance.get_downloads.return_value = [mock_dl]
    
    fetcher.fetch([Checkpoints.STABLE_DIFFUSION_V2_1, Vae.FLUX1D, TextEncoders.CLIP_L])
        
    assert mock_api_instance.add_uris.call_count == 3
    
    # Check HF Headers
    calls = mock_api_instance.add_uris.call_args_list
    assert any("Authorization: Bearer mock_hf" in kwargs['options'].get('header', []) for args, kwargs in calls)

def test_fetcher_dynamic_model(fetcher, mock_aria2):
    """Test fetching dynamic models via DynamicModel."""
    mock_api_instance = mock_aria2.return_value
    mock_dl = MagicMock()
    mock_dl.gid = "123"
    mock_dl.is_complete = True
    mock_api_instance.add_uris.return_value = mock_dl
    mock_api_instance.get_downloads.return_value = [mock_dl]
    
    dyn_model = DynamicModel(url="https://civitai.com/api/download/models/12345", provider="civitai", category="Dynamic")
    fetcher.fetch([dyn_model])
        
    mock_api_instance.add_uris.assert_called_once()
    call_args = mock_api_instance.add_uris.call_args[0]
    requested_url = call_args[0][0]
    assert "token=mock_civitai" in requested_url

def test_fetcher_deduplicates_duplicate_models(fetcher, mock_aria2):
    """Test that duplicate models resolving to the same local path are only downloaded once."""
    mock_api_instance = mock_aria2.return_value
    mock_dl = MagicMock()
    mock_dl.gid = "999"
    mock_dl.is_complete = True
    mock_api_instance.add_uris.return_value = mock_dl
    mock_api_instance.get_downloads.return_value = [mock_dl]
    
    # Two unequal DynamicModels (different URLs) that resolve to the same filename and category
    dyn_model1 = DynamicModel(url="https://civitai.com/api/download/models/797871", provider="civitai", category="Dynamic", filename="watercolor.safetensors")
    dyn_model2 = DynamicModel(url="https://civitai.com/api/download/models/797871?another_param=1", provider="civitai", category="Dynamic", filename="watercolor.safetensors")
    
    res = fetcher.fetch([dyn_model1, dyn_model2])
    
    # add_uris should only have been called once because they resolved to the same local_path
    mock_api_instance.add_uris.assert_called_once()
    assert len(res) == 2
    assert res[dyn_model1] == res[dyn_model2]


