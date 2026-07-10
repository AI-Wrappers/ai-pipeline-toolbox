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
    calls = mock_api_instance.add_uris.call_args_list
    assert any("Authorization: Bearer mock_civitai" in kwargs['options'].get('header', []) for args, kwargs in calls)

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


def test_fetcher_403_error_logging(fetcher, mock_aria2):
    """Test that a detailed 403 error message is logged when a 403 response status code is encountered."""
    mock_api_instance = mock_aria2.return_value
    mock_dl = MagicMock()
    mock_dl.gid = "err403"
    mock_dl.has_failed = True
    mock_dl.is_complete = False
    mock_dl.error_code = 22
    mock_dl.error_message = "The response status is not successful. status=403"
    
    mock_file = MagicMock()
    mock_file.uris = [{"uri": "https://httpbin.org/status/403"}]
    mock_dl.files = [mock_file]
    
    mock_api_instance.add_uris.return_value = mock_dl
    mock_api_instance.get_downloads.return_value = [mock_dl]
    
    dyn_model = DynamicModel(url="https://httpbin.org/status/403", provider="direct_url", category="Dynamic", filename="test_403.safetensors")
    
    with patch('ai_pipeline_toolbox.components.model_fetcher.logger') as mock_logger:
        with patch('urllib.request.urlopen') as mock_urlopen:
            with patch('time.sleep') as mock_sleep:
                from urllib.error import HTTPError
                mock_urlopen.side_effect = HTTPError("https://httpbin.org/status/403", 429, "Too Many Requests", {}, None)
                
                with pytest.raises(RuntimeError, match="Failed to download"):
                    fetcher.fetch([dyn_model])
                
                # Check that logger.warning was called for the 403
                warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
                assert any("[HTTP 403 Forbidden]" in warning for warning in warning_calls)
                assert any("HF_TOKEN" in warning for warning in warning_calls)
                assert any("CIVITAI_API_TOKEN" in warning for warning in warning_calls)
                assert any("https://httpbin.org/status/403" in warning for warning in warning_calls)
                
                # Check that it tried to add the URI twice (because it retries 403 once before fallback)
                assert mock_api_instance.add_uris.call_count == 2


@pytest.mark.skipif(
    not __import__("os").environ.get("CIVITAI_API_TOKEN"),
    reason="CIVITAI_API_TOKEN environment variable is not set"
)
def test_real_download_civitai_lora(tmp_path):
    """Real integration test to download a LoRA from Civitai using a URN."""
    import os
    from pathlib import Path
    from ai_pipeline_toolbox.core.helpers import resolve_air_urn
    
    urn = "urn:air:flux1:lora:civitai:1075723@1207606"
    download_url = resolve_air_urn(urn)
    
    civitai_token = os.environ.get("CIVITAI_API_TOKEN")
    fetcher = ModelFetcher(
        cache_dir=str(tmp_path / "models"),
        civitai_token=civitai_token
    )
    
    model = DynamicModel(
        url=download_url,
        provider="civitai",
        category="Lora",
        filename="watercolor_real_test.safetensors"
    )
    
    res = fetcher.fetch([model])
    local_path = Path(res[model])
    
    assert local_path.exists()
    assert local_path.stat().st_size > 0




