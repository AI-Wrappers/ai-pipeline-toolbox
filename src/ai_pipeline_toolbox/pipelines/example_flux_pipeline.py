import time
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.registry.generated_enums import Checkpoints, Vae

logger = logging.getLogger(__name__)

class FluxConfig(BaseModel):
    num_inference_steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)

class FluxWorkload(BaseModel):
    task_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = -1

class ExampleFluxPipeline(BaseGenerationPipeline[FluxConfig, FluxWorkload, str]):
    """
    A dummy implementation of a generative pipeline.
    """
    required_models = [Checkpoints.FLUX_DEV, Vae.AE]
        
    def setup(self, models_paths: Dict[Enum, str]) -> None:
        logger.info(f"Setting up Flux pipeline with models: {models_paths}")
        time.sleep(1) # Fake loading time
        logger.info("Pipeline setup complete.")
        
    def __call__(self, config: FluxConfig, workload: FluxWorkload) -> str:
        logger.info(f"Generating image for prompt: '{workload.prompt}' (Steps: {config.num_inference_steps})")
        time.sleep(2) # Fake generation time
        return f"Fake image result for prompt: '{workload.prompt}' with seed {workload.seed}"
