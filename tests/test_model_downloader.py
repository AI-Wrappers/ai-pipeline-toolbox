import pytest
from enum import Enum
from unittest.mock import patch, mock_open
from ai_pipeline_toolbox.components.model_downloader import ModelDownloader

class MockRegistry(Enum):
    HF_MODEL = {'provider': 'huggingface', 'repo_id': 'test/repo'}
    CIVITAI_MODEL = {'provider': 'civitai', 'model_version_id': '12345'}
    URL_MODEL = {'provider': 'openmodeldb', 'url': 'http://example.com/model.bin'}

@pytest.fixture
def downloader(tmp_path):
    """Створюємо ModelDownloader з тимчасовою кеш-директорією."""
    return ModelDownloader(cache_dir=str(tmp_path / "models"))

@patch("ai_pipeline_toolbox.components.model_downloader.snapshot_download")
def test_downloader_huggingface(mock_snapshot, downloader):
    """Тестуємо логіку виклику завантаження з HuggingFace."""
    mock_snapshot.return_value = "/mock/path/hf"
    
    paths = downloader.download([MockRegistry.HF_MODEL])
    
    assert MockRegistry.HF_MODEL in paths
    assert paths[MockRegistry.HF_MODEL] == "/mock/path/hf"
    mock_snapshot.assert_called_once()

@patch("ai_pipeline_toolbox.components.model_downloader.requests.get")
def test_downloader_url(mock_get, downloader):
    """Тестуємо завантаження моделі за прямим посиланням через requests."""
    mock_response = mock_get.return_value
    mock_response.iter_content.return_value = [b"data"]
    
    # Використовуємо mock_open щоб імітувати збереження файлу
    with patch("builtins.open", mock_open()):
        paths = downloader.download([MockRegistry.URL_MODEL])
        
    assert MockRegistry.URL_MODEL in paths
    assert "model.bin" in paths[MockRegistry.URL_MODEL]
    mock_get.assert_called_once_with('http://example.com/model.bin', stream=True, headers={})
