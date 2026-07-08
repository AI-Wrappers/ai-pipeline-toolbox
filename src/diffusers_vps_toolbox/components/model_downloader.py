import os
import requests
from typing import List, Dict
from enum import Enum
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers_vps_toolbox.core.interfaces import BaseDownloader

class ModelDownloader(BaseDownloader):
    """
    Downloads missing weights and returns local paths.
    Supports Hugging Face, CivitAI, and OpenModelDB schemas.
    """
    def __init__(self, cache_dir: str = "./models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, required_models: List[Enum]) -> Dict[Enum, str]:
        local_paths = {}
        for model_enum in required_models:
            config = model_enum.value
            provider = config.get("provider")
            
            if provider == "huggingface":
                local_paths[model_enum] = self._download_hf(config)
            elif provider in ("civitai", "openmodeldb"):
                local_paths[model_enum] = self._download_url(config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        return local_paths

    def _download_hf(self, config: dict) -> str:
        repo_id = config["repo_id"]
        filename = config.get("filename")
        if filename:
            path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(self.cache_dir))
        else:
            path = snapshot_download(repo_id=repo_id, cache_dir=str(self.cache_dir))
        return path

    def _download_url(self, config: dict) -> str:
        provider = config.get("provider")
        
        # Use CivitAI API for downloads if we have the version ID
        if provider == "civitai" and "model_version_id" in config:
            url = f"https://civitai.com/api/download/models/{config['model_version_id']}"
        else:
            url = config.get("url")
            
        if not url:
            raise ValueError(f"Unable to determine download URL for config: {config}")
            
        filename = config.get("filename")
        if not filename:
            if "model_version_id" in config:
                filename = f"model_{config['model_version_id']}.safetensors"
            else:
                filename = url.split('/')[-1] or "downloaded_model.bin"
                
        local_path = self.cache_dir / filename
        
        if local_path.exists():
            return str(local_path)
            
        headers = {}
        if provider == "civitai":
            civitai_token = os.environ.get("CIVITAI_API_TOKEN")
            if civitai_token:
                headers["Authorization"] = f"Bearer {civitai_token}"
                
        print(f"Downloading from {url} to {local_path}...")
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return str(local_path)
