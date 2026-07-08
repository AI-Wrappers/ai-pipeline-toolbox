import os
import json
from pathlib import Path
from typing import Any, Dict
from diffusers_vps_toolbox.core.interfaces import BaseResultSaver

class LocalResultSaver(BaseResultSaver):
    """
    Saves pipeline results locally.
    """
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: Any, metadata: Dict[str, Any]) -> str:
        """
        Saves a fake result (string representation) to a file.
        """
        task_id = metadata.get("task_id", "unknown_task")
        output_path = self.output_dir / f"{task_id}.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(result))
            f.write(f"\n\nMetadata: {json.dumps(metadata)}")
            
        return str(output_path)
