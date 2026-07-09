import pytest
from enum import Enum
from unittest.mock import patch, MagicMock
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher

class MockRegistry(Enum):
    HF_MODEL = {'provider': 'huggingface', 'download_url': 'test/repo'}
    CIVITAI_MODEL = {'provider': 'civitai', 'download_url': 'https://civitai.com/api/download/models/12345'}
    URL_MODEL = {'provider': 'direct_url', 'download_url': 'http://example.com/model.bin'}

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
    mock_api_instance.add_uris.return_value = [mock_dl]
    mock_api_instance.get_downloads.return_value = [mock_dl]
    
    fetcher.fetch([MockRegistry.HF_MODEL, MockRegistry.CIVITAI_MODEL, MockRegistry.URL_MODEL])
        
    assert mock_api_instance.add_uris.call_count == 3
    
    # Check HF Headers
    calls = mock_api_instance.add_uris.call_args_list
    assert any("Authorization: Bearer mock_hf" in kwargs['options'].get('header', []) for args, kwargs in calls)
    assert any("Authorization: Bearer mock_civitai" in kwargs['options'].get('header', []) for args, kwargs in calls)

def test_fetcher_dynamic_string(fetcher, mock_aria2):
    """Test fetching dynamic models via string URLs."""
    mock_api_instance = mock_aria2.return_value
    mock_dl = MagicMock()
    mock_dl.gid = "123"
    mock_dl.is_complete = True
    mock_api_instance.add_uris.return_value = [mock_dl]
    mock_api_instance.get_downloads.return_value = [mock_dl]
    
    fetcher.fetch(["https://huggingface.co/username/model"])
        
    mock_api_instance.add_uris.assert_called_once()
    kwargs = mock_api_instance.add_uris.call_args[1]
    assert "Authorization: Bearer mock_hf" in kwargs['options'].get('header', [])
