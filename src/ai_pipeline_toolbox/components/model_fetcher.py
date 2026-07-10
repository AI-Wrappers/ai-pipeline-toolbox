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

    def __init__(
        self,
        cache_dir: str = "./models_cache",
        start_port: int = 6800,
        hf_token: Optional[str] = None,
        civitai_token: Optional[str] = None,
    ):
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
            return s.connect_ex(("localhost", port)) == 0

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
                    "--daemon=false",  # Keep it as child process to terminate easily
                    "--max-concurrent-downloads=1",
                    f"--dir={self.cache_dir}",
                ]
                self.daemon_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

                # Wait briefly for daemon to start
                time.sleep(1)

                # Initialize aria2p API
                self.aria2 = aria2p.API(
                    aria2p.Client(host="http://localhost", port=current_port, secret="")
                )
                # Test connection
                self.aria2.get_global_options()

                self.port = current_port
                logger.info(f"Successfully started aria2c daemon on port {self.port}")
                return
            except Exception as e:
                logger.warning(
                    f"Failed to connect to aria2c on port {current_port}: {e}"
                )
                if self.daemon_process:
                    self.daemon_process.terminate()
                    self.daemon_process.wait()

        raise RuntimeError("Failed to start aria2c daemon after multiple attempts.")

    def fetch(
        self, models: List[Union[Enum, DynamicModel]]
    ) -> Dict[Union[Enum, DynamicModel], str]:
        local_paths = {}
        failed_403_counts = {}

        # Group models by their destination local_path
        # and resolve target directories/filenames first
        models_by_path = {}
        model_details = {}  # local_path -> (provider, download_url, filename, target_dir)

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
                filename = (
                    model.filename
                    or download_url.split("/")[-1]
                    or "downloaded_model.bin"
                )
                # Autodetect provider from domain if not explicitly set and direct_url is default
                if provider == "direct_url":
                    if "huggingface.co" in download_url:
                        provider = "huggingface"
                    elif "civitai.com" in download_url:
                        provider = "civitai"

                # If model.category is Enum class it will be parsed to str, then we validate via Category
                # model.category is a str or Category here because of validation in DynamicModel
                cat_val = (
                    model.category.value
                    if isinstance(model.category, Enum)
                    else model.category
                )
                sub_dir = Category(cat_val).value
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

            target_dir = self.cache_dir / sub_dir
            local_path = target_dir / filename

            models_by_path.setdefault(local_path, []).append(model)
            if local_path not in model_details:
                model_details[local_path] = (
                    provider,
                    download_url,
                    filename,
                    target_dir,
                )

        # Now handle downloads for each unique local_path
        for local_path, models_list in models_by_path.items():
            if local_path.exists():
                for model in models_list:
                    local_paths[model] = str(local_path)
                continue

            provider, download_url, filename, target_dir = model_details[local_path]
            target_dir.mkdir(parents=True, exist_ok=True)

            headers = []
            options = {
                "dir": str(target_dir),
                "out": filename,
                "max-connection-per-server": "1",
                "split": "1",
            }
            if provider == "huggingface":
                if self.hf_token:
                    headers.append(f"Authorization: Bearer {self.hf_token}")
            elif provider == "civitai":
                if self.civitai_token:
                    headers.append(f"Authorization: Bearer {self.civitai_token}")

            if headers:
                options["header"] = headers

            logger.info(
                f"Adding download for {filename} from {download_url} with provider {provider}"
            )
            try:
                download = self.aria2.add_uris([download_url], options=options)
            except Exception as e:
                logger.error(f"Failed to add download for {download_url}: {e}")
                raise

            # Wait for this single download to finish before proceeding to the next
            while True:
                downloads = self.aria2.get_downloads()
                current_dl = next((d for d in downloads if d.gid == download.gid), None)

                if not current_dl:
                    # Depending on aria2 behavior, if it's completed it might still be in the list
                    # or it might have been purged. Safe assumption: if we can't find it, we need to check the file.
                    if local_path.exists():
                        for model in models_list:
                            local_paths[model] = str(local_path)
                    else:
                        raise RuntimeError(
                            f"Download task {download.gid} disappeared and file not found."
                        )
                    break

                if current_dl.is_complete:
                    for model in models_list:
                        local_paths[model] = str(local_path)
                    break
                elif current_dl.has_failed:
                    err_msg = current_dl.error_message or ""
                    # Check for HTTP 403 Forbidden
                    if "403" in err_msg:
                        # Increment 403 count for this download path
                        failed_403_counts[local_path] = failed_403_counts.get(local_path, 0) + 1
                        
                        if failed_403_counts[local_path] < 2:
                            try:
                                current_dl.remove(force=True, files=True)
                            except Exception:
                                pass
                            
                            logger.warning(
                                f"Download for {filename} failed with HTTP 403 Forbidden. "
                                f"Retrying via aria2c (attempt {failed_403_counts[local_path] + 1}/2)..."
                            )
                            try:
                                download = self.aria2.add_uris([download_url], options=options)
                                continue
                            except Exception as e:
                                logger.error(f"Failed to retry download for {download_url}: {e}")
                                raise

                        download_url_info = "unknown URL"
                        if current_dl.files and current_dl.files[0].uris:
                            first_uri = current_dl.files[0].uris[0]
                            if isinstance(first_uri, dict):
                                download_url_info = first_uri.get("uri", "unknown")
                            else:
                                download_url_info = getattr(first_uri, "uri", "unknown")

                        hf_status = "set" if self.hf_token else "not set"
                        civitai_status = "set" if self.civitai_token else "not set"
                        logger.warning(
                            f"\n[HTTP 403 Forbidden] Download failed via aria2c twice for {local_path}!\n"
                            f"Target URL: {download_url_info}\n"
                            f"Possible causes:\n"
                            f"1. Missing or invalid authentication token. Currently, HF_TOKEN is {hf_status} and CIVITAI_API_TOKEN is {civitai_status}.\n"
                            f"2. Cloudflare or host blocked the request (e.g. TLS fingerprinting of aria2c).\n"
                            f"Falling back to native Python download to bypass the block...\n"
                        )
                        try:
                            import urllib.request
                            import urllib.error
                            import shutil
                            
                            # Add a brief cooldown sleep to allow Civitai server connection/limits to clear
                            time.sleep(2)
                            
                            max_retries = 5
                            retry_delay = 2.0
                            for attempt in range(max_retries):
                                try:
                                    fallback_headers = {
                                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                                    }
                                    if provider == "civitai" and self.civitai_token:
                                        fallback_headers["Authorization"] = f"Bearer {self.civitai_token}"
                                    elif provider == "huggingface" and self.hf_token:
                                        fallback_headers["Authorization"] = f"Bearer {self.hf_token}"

                                    req = urllib.request.Request(
                                        download_url, 
                                        headers=fallback_headers
                                    )
                                    # Ensure parent directory exists
                                    local_path.parent.mkdir(parents=True, exist_ok=True)
                                    
                                    # Download to a temporary file first, then rename to prevent partial file corruption
                                    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
                                    with urllib.request.urlopen(req) as response, open(temp_path, 'wb') as out_file:
                                        shutil.copyfileobj(response, out_file)
                                    temp_path.rename(local_path)
                                    
                                    logger.info(f"Successfully downloaded {local_path} using Python fallback!")
                                    for model in models_list:
                                        local_paths[model] = str(local_path)
                                    break
                                except urllib.error.HTTPError as http_err:
                                    if http_err.code == 429 and attempt < max_retries - 1:
                                        logger.warning(
                                            f"HTTP 429 Too Many Requests when downloading via Python fallback. "
                                            f"Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})..."
                                        )
                                        time.sleep(retry_delay)
                                        retry_delay *= 2.0
                                    else:
                                        raise
                            else:
                                raise RuntimeError("Failed to download after max retries due to HTTP 429 rate limiting.")
                            break
                        except Exception as python_err:
                            logger.error(f"Python fallback download failed: {python_err}")
                            raise RuntimeError(
                                f"Failed to download {local_path} via both aria2c (403) and Python fallback: {python_err}"
                            )
                    else:
                        logger.error(
                            f"Download failed for {local_path}: {err_msg}"
                        )
                        raise RuntimeError(
                            f"Failed to download {local_path}: {err_msg}"
                        )

                time.sleep(2)

        return local_paths

    def __del__(self):
        if getattr(self, "daemon_process", None):
            self.daemon_process.terminate()
            self.daemon_process.wait()
