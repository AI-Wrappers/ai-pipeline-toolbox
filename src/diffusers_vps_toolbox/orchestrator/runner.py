import logging
from typing import Any, Dict, Optional
from diffusers_vps_toolbox.core.interfaces import (
    BaseLoopManager, BaseStateManager, BaseDownloader, 
    BaseResultSaver, BaseWorkloadProcessor
)
from diffusers_vps_toolbox.core.pipeline import BaseGenerationPipeline

logger = logging.getLogger(__name__)

class Runner:
    """
    The orchestrator that manages the execution flow of the pipeline.
    """
    def __init__(
        self,
        workload_processor: BaseWorkloadProcessor,
        state_manager: BaseStateManager,
        downloader: BaseDownloader,
        loop_manager: BaseLoopManager,
        result_saver: BaseResultSaver,
    ):
        self.workload_processor = workload_processor
        self.state_manager = state_manager
        self.downloader = downloader
        self.loop_manager = loop_manager
        self.result_saver = result_saver

    def run(
        self,
        pipeline: BaseGenerationPipeline,
        raw_workload: Any,
        config: Any
    ):
        """
        Executes the main pipeline flow.
        """
        logger.info("Starting Runner flow.")
        
        # 2. Pass Workload to WorkloadProcessor
        validated_tasks = list(self.workload_processor.process(raw_workload))
        
        # 3. Query StateManager to filter out already completed tasks
        pending_tasks = []
        for task in validated_tasks:
            task_id = getattr(task, 'task_id', None)
            if not task_id:
                raise ValueError("Tasks must have a 'task_id' field for StateManager tracking.")
                
            if not self.state_manager.is_completed(task_id):
                pending_tasks.append(task)
            else:
                logger.info(f"Skipping completed task: {task_id}")

        if not pending_tasks:
            logger.info("No pending tasks to execute.")
            return

        # 4. Call ModelDownloader to ensure dependencies are met
        models_paths = self.downloader.download(pipeline.required_models)
        
        # 5. Instantiates the Pipeline with downloaded model paths
        pipeline.setup(models_paths)
        
        # 6. Hand the remaining tasks to LoopManager
        for task in self.loop_manager.iterate(pending_tasks):
            task_id = getattr(task, 'task_id')
            try:
                logger.info(f"Executing pipeline for task: {task_id}")
                # 7. Execute the Pipeline
                result = pipeline(config=config, workload=task)
                
                # 8. Pass results to ResultSaver
                # Handle standard dump methods for pydantic models
                config_dict = config.model_dump() if hasattr(config, "model_dump") else (config.dict() if hasattr(config, "dict") else str(config))
                metadata = {"task_id": task_id, "config": config_dict}
                
                saved_path = self.result_saver.save(result, metadata)
                logger.info(f"Result saved to {saved_path}")
                
                # 8. Update StateManager on success
                self.state_manager.mark_completed(task_id)
            except Exception as e:
                logger.error(f"Error executing task {task_id}: {e}", exc_info=True)
                # Update StateManager on failure
                self.state_manager.mark_failed(task_id, e)
