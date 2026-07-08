import pytest
from diffusers_vps_toolbox.components.state_manager import SQLiteStateManager

@pytest.fixture
def state_manager(tmp_path):
    """Створюємо тимчасову БД для ізольованого тестування."""
    db_path = tmp_path / "test_state.db"
    return SQLiteStateManager(db_path=str(db_path))

def test_state_manager_mark_completed(state_manager):
    """Тестуємо, що задача коректно відмічається як виконана."""
    task_id = "task_1"
    assert not state_manager.is_completed(task_id)
    
    state_manager.mark_completed(task_id)
    assert state_manager.is_completed(task_id)

def test_state_manager_mark_failed(state_manager):
    """Тестуємо, що статус помилки зберігається і задача не вважається виконаною."""
    task_id = "task_error"
    error = ValueError("Something went wrong")
    
    state_manager.mark_failed(task_id, error)
    
    # Задача з помилкою не має вважатись 'completed'
    assert not state_manager.is_completed(task_id)
