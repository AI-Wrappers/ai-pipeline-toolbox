from typing import Any, Iterable
from diffusers_vps_toolbox.core.interfaces import BaseLoopManager

class LoopManager(BaseLoopManager):
    """
    Implements a simple FIFO queue for iterating over a workload.
    """
    
    def iterate(self, workload: Iterable[Any]) -> Iterable[Any]:
        """
        Iterates over the workload in a FIFO manner.
        """
        for item in workload:
            yield item
