import pytest
from diffusers_vps_toolbox.components.loop_manager import LoopManager

def test_loop_manager_fifo():
    """
    Тестуємо, що LoopManager повертає елементи у порядку FIFO 
    (перший прийшов - перший вийшов).
    """
    manager = LoopManager()
    workload = [1, 2, 3]
    result = list(manager.iterate(workload))
    
    assert result == [1, 2, 3]
