from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, List, Union
from enum import Enum

ConfigType = TypeVar('ConfigType')
WorkloadType = TypeVar('WorkloadType')
ReturnType = TypeVar('ReturnType')

class BaseGenerationPipeline(ABC, Generic[ConfigType, WorkloadType, ReturnType]):
    """
    Base contract for all generative AI pipelines.
    """
    
    # Populated from registry/generated_enums.py by subclasses
    required_models: List[Enum] = []
    
    @abstractmethod
    def setup(self, models_paths: Dict[Union[Enum, str], str]) -> None:
        """
        Initializes models into memory using the provided local paths.
        
        Args:
            models_paths: Mapping of Enum model identifiers to local directory paths.
        """
        pass
        
    def get_dynamic_models(self, workload: WorkloadType) -> List[str]:
        """
        Extracts dynamically required models (e.g., LoRAs) from a workload.
        
        Args:
            workload: A single validated task/workload item.
            
        Returns:
            A list of string URLs pointing to dynamic models.
        """
        return []
        
    @abstractmethod
    def __call__(self, config: ConfigType, workload: WorkloadType) -> ReturnType:
        """
        Executes the generation based on the config and workload.
        
        Args:
            config: Validated configuration (e.g., Pydantic model).
            workload: A single validated task/workload item.
            
        Returns:
            The generated result.
        """
        pass
