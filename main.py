import logging
import os
from typing import List, Dict
from enum import Enum
from diffusers_vps_toolbox.orchestrator.runner import Runner
from diffusers_vps_toolbox.components.workload_processor import PydanticWorkloadProcessor
from diffusers_vps_toolbox.components.state_manager import SQLiteStateManager
from diffusers_vps_toolbox.components.model_downloader import ModelDownloader
from diffusers_vps_toolbox.components.loop_manager import LoopManager
from diffusers_vps_toolbox.components.result_saver import LocalResultSaver
from diffusers_vps_toolbox.pipelines.example_flux_pipeline import ExampleFluxPipeline, FluxWorkload, FluxConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MockDownloader(ModelDownloader):
    def download(self, required_models: List[Enum]) -> Dict[Enum, str]:
        logging.info(f"Mock downloading models: {[m.name for m in required_models]}")
        return {m: f"/mock/path/{m.name}" for m in required_models}

def main():
    # 1. Initialize Components (Dependency Injection)
    workload_processor = PydanticWorkloadProcessor(model_class=FluxWorkload)
    state_manager = SQLiteStateManager(db_path="data/state.db")
    downloader = MockDownloader(cache_dir="data/models_cache")
    loop_manager = LoopManager()
    result_saver = LocalResultSaver(output_dir="data/outputs")
    
    runner = Runner(
        workload_processor=workload_processor,
        state_manager=state_manager,
        downloader=downloader,
        loop_manager=loop_manager,
        result_saver=result_saver
    )
    
    # 2. Define Workload and Config
    raw_workload = [
        {"task_id": "task_001", "prompt": "A futuristic city at sunset, synthwave style", "seed": 42},
        {"task_id": "task_002", "prompt": "A cute cat wearing a spacesuit", "seed": 100},
        {"task_id": "task_003", "prompt": "Cyberpunk portrait of a woman", "seed": 777}
    ]
    
    config = FluxConfig(num_inference_steps=25, guidance_scale=5.0)
    
    # 3. Initialize Pipeline
    pipeline = ExampleFluxPipeline()
    
    # 4. Run!
    print("\n=== Starting First Run ===")
    runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)
    
    print("\n=== Starting Second Run (Should skip completed tasks) ===")
    runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)

if __name__ == "__main__":
    main()
