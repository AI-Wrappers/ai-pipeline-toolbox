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
        max_concurrent_downloads: int = 3,
        max_connections_for_provider: Optional[Dict[str, int]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.port = start_port
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.civitai_token = civitai_token or os.environ.get("CIVITAI_API_TOKEN")
        self.max_concurrent_downloads = max_concurrent_downloads

        self.max_connections_for_provider = {
            "huggingface": 8,
            "civitai": 2,
        }
        if max_connections_for_provider:
            self.max_connections_for_provider.update(max_connections_for_provider)

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
                    f"--max-concurrent-downloads={self.max_concurrent_downloads}",
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

    def _get_download_options(
        self,
        provider: str,
        filename: str,
        target_dir: Path,
        allocated_connections: int
    ) -> Dict[str, Union[str, List[str]]]:
        options = {
            "dir": str(target_dir),
            "out": filename,
            "max-connection-per-server": str(allocated_connections),
            "split": str(allocated_connections),
        }
        headers = []
        if provider == "huggingface" and self.hf_token:
            headers.append(f"Authorization: Bearer {self.hf_token}")
        elif provider == "civitai" and self.civitai_token:
            headers.append(f"Authorization: Bearer {self.civitai_token}")

        if headers:
            options["header"] = headers
        return options

    def _python_fallback_download(
        self,
        local_path: Path,
        provider: str,
        download_url: str,
    ):
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
                return
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

    def fetch(
        self, models: List[Union[Enum, DynamicModel]]
    ) -> Dict[Union[Enum, DynamicModel], str]:
        from concurrent.futures import ThreadPoolExecutor

        local_paths = {}

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

        # Initialize the scheduling tasks queue
        tasks = {}
        for local_path, models_list in models_by_path.items():
            if local_path.exists():
                for model in models_list:
                    local_paths[model] = str(local_path)
            else:
                provider, download_url, filename, target_dir = model_details[local_path]
                tasks[local_path] = {
                    "status": "pending",
                    "models": models_list,
                    "provider": provider,
                    "download_url": download_url,
                    "filename": filename,
                    "target_dir": target_dir,
                    "gid": None,
                    "allocated_connections": 0,
                    "403_count": 0,
                }

        # If all requested models are already cached
        if not tasks:
            return local_paths

        # Execute fallback downloads using a thread pool
        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            fallback_futures = {}  # future -> local_path

            while any(task["status"] in ("pending", "downloading", "fallback") for task in tasks.values()):
                downloads = self.aria2.get_downloads()
                newly_started_gids = set()

                # 1. Count currently active tasks (downloading + fallback)
                active_tasks = [t for t in tasks.values() if t["status"] in ("downloading", "fallback")]

                # Calculate sum of active connections per provider
                active_conns = {}
                for t in active_tasks:
                    p = t["provider"]
                    active_conns[p] = active_conns.get(p, 0) + t["allocated_connections"]

                # 2. Try to start pending tasks
                for local_path, task in tasks.items():
                    if task["status"] != "pending":
                        continue

                    # Limit check 1: Max concurrent downloads slot
                    if len(active_tasks) >= self.max_concurrent_downloads:
                        break

                    p = task["provider"]
                    max_p_conn = self.max_connections_for_provider.get(p, 4)
                    current_p_conn = active_conns.get(p, 0)

                    # Limit check 2: Connections available for provider
                    available_conns = max_p_conn - current_p_conn
                    if available_conns >= 1:
                        allocated = min(available_conns, max_p_conn)
                        task["allocated_connections"] = allocated
                        
                        task["target_dir"].mkdir(parents=True, exist_ok=True)
                        options = self._get_download_options(
                            p, task["filename"], task["target_dir"], allocated
                        )

                        logger.info(
                            f"Starting download for {task['filename']} from {task['download_url']} "
                            f"via provider {p} (allocated {allocated} connections)"
                        )
                        try:
                            download = self.aria2.add_uris([task["download_url"]], options=options)
                            task["gid"] = download.gid
                            task["status"] = "downloading"
                            newly_started_gids.add(download.gid)
                            
                            # Update local active state for this loop iteration
                            active_tasks.append(task)
                            active_conns[p] = active_conns.get(p, 0) + allocated
                        except Exception as e:
                            logger.error(f"Failed to add download for {task['download_url']}: {e}")
                            task["status"] = "failed"
                            raise

                # 3. Monitor active aria2 tasks
                for local_path, task in tasks.items():
                    if task["status"] != "downloading":
                        continue

                    if task["gid"] in newly_started_gids:
                        continue

                    current_dl = next((d for d in downloads if d.gid == task["gid"]), None)

                    if not current_dl:
                        # If task is not in active aria2 queue, check if the file was downloaded successfully
                        if local_path.exists():
                            task["status"] = "completed"
                            task["allocated_connections"] = 0
                            for model in task["models"]:
                                local_paths[model] = str(local_path)
                        else:
                            task["status"] = "failed"
                            task["allocated_connections"] = 0
                            raise RuntimeError(
                                f"Download task {task['gid']} disappeared and file not found."
                            )
                        continue

                    if current_dl.is_complete:
                        task["status"] = "completed"
                        task["allocated_connections"] = 0
                        for model in task["models"]:
                            local_paths[model] = str(local_path)
                    elif current_dl.has_failed:
                        err_msg = current_dl.error_message or ""
                        try:
                            current_dl.remove(force=True, files=True)
                        except Exception:
                            pass

                        # Check for HTTP 403 Forbidden for retry
                        if "403" in err_msg and task["403_count"] < 1:
                            task["403_count"] += 1
                            logger.warning(
                                f"Download for {task['filename']} failed with HTTP 403 Forbidden. "
                                f"Retrying via aria2c (attempt {task['403_count'] + 1}/2)..."
                            )
                            try:
                                options = self._get_download_options(
                                    task["provider"], task["filename"], task["target_dir"], task["allocated_connections"]
                                )
                                download = self.aria2.add_uris([task["download_url"]], options=options)
                                task["gid"] = download.gid
                            except Exception as e:
                                logger.error(f"Failed to retry download for {task['download_url']}: {e}")
                                task["status"] = "failed"
                                task["allocated_connections"] = 0
                                raise
                        else:
                            # Fallback to python download
                            task["status"] = "fallback"
                            task["allocated_connections"] = 1

                            hf_status = "set" if self.hf_token else "not set"
                            civitai_status = "set" if self.civitai_token else "not set"
                            logger.warning(
                                f"\n[HTTP 403 Forbidden] Download failed via aria2c twice for {local_path}!\n"
                                f"Target URL: {task['download_url']}\n"
                                f"Possible causes:\n"
                                f"1. Missing or invalid authentication token. Currently, HF_TOKEN is {hf_status} and CIVITAI_API_TOKEN is {civitai_status}.\n"
                                f"2. Cloudflare or host blocked the request (e.g. TLS fingerprinting of aria2c).\n"
                                f"Falling back to native Python download to bypass the block...\n"
                            )
                            future = executor.submit(
                                self._python_fallback_download,
                                local_path,
                                task["provider"],
                                task["download_url"]
                            )
                            fallback_futures[future] = local_path

                # 4. Monitor fallback downloads
                done_futures = [f for f in fallback_futures if f.done()]
                for f in done_futures:
                    l_path = fallback_futures.pop(f)
                    task = tasks[l_path]
                    try:
                        f.result()
                        task["status"] = "completed"
                        task["allocated_connections"] = 0
                        for model in task["models"]:
                            local_paths[model] = str(l_path)
                    except Exception as python_err:
                        logger.error(f"Python fallback download failed for {l_path}: {python_err}")
                        task["status"] = "failed"
                        task["allocated_connections"] = 0
                        raise RuntimeError(
                            f"Failed to download {l_path} via both aria2c and Python fallback: {python_err}"
                        )

                time.sleep(2)

        return local_paths

    def __del__(self):
        if getattr(self, "daemon_process", None):
            self.daemon_process.terminate()
            self.daemon_process.wait()
