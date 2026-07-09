import os
import time
import socket
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Union, Optional
from enum import Enum

import aria2p
from ai_pipeline_toolbox.core.interfaces import BaseFetcher
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.registry.generated_enums import Category

logger = logging.getLogger(__name__)

class ModelFetcher(BaseFetcher):
    """
    Fetches missing weights and returns local paths.
    Downloads models using aria2c via aria2p RPC daemon.
    """
    def __init__(self, cache_dir: str = "./models_cache", start_port: int = 6800, hf_token: Optional[str] = None, civitai_token: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.port = start_port
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.civitai_token = civitai_token or os.environ.get("CIVITAI_API_TOKEN")
        self.daemon_process = None
        self.aria2 = None
        
        self._start_daemon_with_fallback()

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def _start_daemon_with_fallback(self):
        max_retries = 20
        for i in range(max_retries):
            current_port = self.port + i
            if self._is_port_in_use(current_port):
                logger.warning(f"Port {current_port} is in use, trying next port...")
                continue
            
            try:
                # Start aria2c daemon
                cmd = [
                    "aria2c", 
                    "--enable-rpc", 
                    f"--rpc-listen-port={current_port}", 
                    "--rpc-listen-all=false", 
                    "--daemon=false", # Keep it as child process to terminate easily
                    f"--dir={self.cache_dir}"
                ]
                self.daemon_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                
                # Wait briefly for daemon to start
                time.sleep(1)
                
                # Initialize aria2p API
                self.aria2 = aria2p.API(
                    aria2p.Client(
                        host="http://localhost",
                        port=current_port,
                        secret=""
                    )
                )
                # Test connection
                self.aria2.get_global_options()
                
                self.port = current_port
                logger.info(f"Successfully started aria2c daemon on port {self.port}")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to aria2c on port {current_port}: {e}")
                if self.daemon_process:
                    self.daemon_process.terminate()
                    self.daemon_process.wait()
                    
        raise RuntimeError("Failed to start aria2c daemon after multiple attempts.")

    def fetch(self, models: List[Union[Enum, DynamicModel]]) -> Dict[Union[Enum, DynamicModel], str]:
        local_paths = {}
        download_tasks = []
        
        for model in models:
            if isinstance(model, Enum):
                config = model.value
                provider = config.get("provider")
                download_url = config.get("download_url")
                
                category_name = type(model).__name__
                ext = Path(download_url).suffix
                filename = f"{model.name}{ext}"
                sub_dir = Category(category_name).value
            elif isinstance(model, DynamicModel):
                provider = model.provider
                download_url = model.url
                filename = model.filename or download_url.split('/')[-1] or "downloaded_model.bin"
                # Autodetect provider from domain if not explicitly set and direct_url is default
                if provider == "direct_url":
                    if "huggingface.co" in download_url:
                        provider = "huggingface"
                    elif "civitai.com" in download_url:
                        provider = "civitai"
                
                # If model.category is Enum class it will be parsed to str, then we validate via Category
                # model.category is a str or Category here because of validation in DynamicModel
                cat_val = model.category.value if isinstance(model.category, Enum) else model.category
                sub_dir = Category(cat_val).value
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
                
            target_dir = self.cache_dir / sub_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            local_path = target_dir / filename
            
            if local_path.exists():
                local_paths[model] = str(local_path)
                continue
                
            headers = []
            if provider == "huggingface":
                if self.hf_token:
                    headers.append(f"Authorization: Bearer {self.hf_token}")
            elif provider == "civitai":
                if self.civitai_token:
                    headers.append(f"Authorization: Bearer {self.civitai_token}")
                    
            options = {
                "dir": str(target_dir),
                "out": filename
            }
            if headers:
                options["header"] = headers
                
            logger.info(f"Adding download for {filename} from {download_url} with provider {provider}")
            try:
                download = self.aria2.add_uris([download_url], options=options)
                download_tasks.append((model, local_path, download))
            except Exception as e:
                logger.error(f"Failed to add download for {download_url}: {e}")
                raise
                
        # Wait for all downloads to finish
        while download_tasks:
            # Refresh all downloads status
            downloads = self.aria2.get_downloads()
            download_dict = {d.gid: d for d in downloads}
            
            pending_tasks = []
            for model, local_path, dl in download_tasks:
                current_dl = download_dict.get(dl.gid)
                if not current_dl:
                    # Depending on aria2 behavior, if it's completed it might still be in the list 
                    # or it might have been purged. Safe assumption: if we can't find it, we need to check the file.
                    if local_path.exists():
                        local_paths[model] = str(local_path)
                    else:
                        raise RuntimeError(f"Download task {dl.gid} disappeared and file not found.")
                    continue
                    
                if current_dl.is_complete:
                    local_paths[model] = str(local_path)
                elif current_dl.has_failed:
                    logger.error(f"Download failed for {local_path}: {current_dl.error_message}")
                    raise RuntimeError(f"Failed to download {local_path}: {current_dl.error_message}")
                else:
                    pending_tasks.append((model, local_path, current_dl))
            
            download_tasks = pending_tasks
            if download_tasks:
                time.sleep(2)
                
        return local_paths

    def __del__(self):
        if getattr(self, "daemon_process", None):
            self.daemon_process.terminate()
            self.daemon_process.wait()
