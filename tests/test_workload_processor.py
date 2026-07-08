import pytest
from pydantic import BaseModel, ValidationError
from ai_pipeline_toolbox.components.workload_processor import PydanticWorkloadProcessor

class DummyTask(BaseModel):
    task_id: str
    prompt: str

@pytest.fixture
def processor():
    """Створюємо процесор для нашої тестової моделі DummyTask."""
    return PydanticWorkloadProcessor(model_class=DummyTask)

def test_workload_processor_valid_list(processor):
    """Тестуємо обробку валідного списку словників."""
    raw = [{"task_id": "1", "prompt": "test 1"}, {"task_id": "2", "prompt": "test 2"}]
    tasks = list(processor.process(raw))
    
    assert len(tasks) == 2
    assert tasks[0].task_id == "1"
    assert isinstance(tasks[0], DummyTask)

def test_workload_processor_valid_json_string(processor):
    """Тестуємо парсинг валідного JSON рядка."""
    raw = '{"task_id": "3", "prompt": "test 3"}'
    tasks = list(processor.process(raw))
    
    assert len(tasks) == 1
    assert tasks[0].prompt == "test 3"

def test_workload_processor_invalid_data(processor):
    """Тестуємо, що невалідні дані викликають помилку валідації Pydantic."""
    raw = [{"task_id": "1"}] # відсутнє обов'язкове поле 'prompt'
    with pytest.raises(ValidationError):
        list(processor.process(raw))
