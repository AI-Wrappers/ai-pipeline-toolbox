from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Union, Generic, TypeVar
from enum import Enum

ReturnType = TypeVar('ReturnType')

class BaseLoopManager(ABC):
    """Implements queue logic and iterates over the workload."""
    
    @abstractmethod
    def iterate(self, workload: Iterable[Any]) -> Iterable[Any]:
        """Iterates over the workload agnostic of its internal structure."""
        pass

class BaseStateManager(ABC):
    """Persists execution state and tracks tasks."""
    
    @abstractmethod
    def is_completed(self, task_id: str) -> bool:
        """Checks if a task has already been completed."""
        pass

    @abstractmethod
    def mark_completed(self, task_id: str) -> None:
        """Marks a task as successfully completed."""
        pass

    @abstractmethod
    def mark_failed(self, task_id: str, error: Exception) -> None:
        """Marks a task as failed and records the error."""
        pass

class BaseFetcher(ABC):
    """Fetches missing weights and returns local paths."""
    
    @abstractmethod
    def fetch(self, models: List[Union[Enum, str]]) -> Dict[Union[Enum, str], str]:
        """Maps Enums or URL strings to local paths, downloads weights, returns local paths."""
        pass

class BaseResultSaver(ABC, Generic[ReturnType]):
    """Dynamically constructs output directory trees and saves results."""
    
    @abstractmethod
    def save(self, result: ReturnType, metadata: Dict[str, Any]) -> str:
        """Saves the result based on metadata and returns the path."""
        pass

class BaseWorkloadProcessor(ABC):
    """Validates and standardizes inputs into Pydantic models."""
    
    @abstractmethod
    def process(self, raw_workload: Any) -> Iterable[Any]:
        """Parses complex raw workloads into typed objects."""
        pass
