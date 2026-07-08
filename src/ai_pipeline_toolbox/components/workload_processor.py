import json
from typing import Any, Iterable, Type
from pydantic import BaseModel
from ai_pipeline_toolbox.core.interfaces import BaseWorkloadProcessor

class PydanticWorkloadProcessor(BaseWorkloadProcessor):
    """
    Validates and standardizes JSON inputs into Pydantic models.
    """
    def __init__(self, model_class: Type[BaseModel]):
        """
        Args:
            model_class: The Pydantic model to validate against.
        """
        self.model_class = model_class

    def process(self, raw_workload: Any) -> Iterable[BaseModel]:
        """
        Parses raw workload (JSON string, list of dicts, or dict) into Pydantic objects.
        """
        if isinstance(raw_workload, str):
            try:
                raw_workload = json.loads(raw_workload)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")

        if isinstance(raw_workload, dict):
            raw_workload = [raw_workload]
            
        if not isinstance(raw_workload, list):
            raise ValueError("Workload must be a list of tasks or a single task dictionary.")

        validated_tasks = []
        for task in raw_workload:
            validated_task = self.model_class.model_validate(task)
            validated_tasks.append(validated_task)
            
        return validated_tasks
