import pytest
from unittest.mock import MagicMock
from ai_pipeline_toolbox.orchestrator.runner import Runner

def test_runner_flow():
    """
    Тестуємо, що Runner правильно викликає всі компоненти:
    процесор -> перевірка статусу -> завантаження -> налаштування пайплайна -> виконання.
    """
    processor = MagicMock()
    
    # Створюємо 2 фейкові задачі
    mock_task1 = MagicMock()
    mock_task1.task_id = "1"
    mock_task2 = MagicMock()
    mock_task2.task_id = "2"
    processor.process.return_value = [mock_task1, mock_task2]
    
    state_manager = MagicMock()
    # Імітуємо, що Задача 1 вже виконана, а задача 2 ще ні
    state_manager.is_completed.side_effect = lambda t_id: t_id == "1"
    
    loop_manager = MagicMock()
    # LoopManager просто повертає задачі, які йому передали
    loop_manager.iterate.side_effect = lambda tasks: iter(tasks)
    
    fetcher = MagicMock()
    fetcher.fetch.return_value = {"model": "/path"}
    
    result_saver = MagicMock()
    result_saver.save.return_value = "/saved/path"
    
    pipeline = MagicMock()
    pipeline.required_models = ["model"]
    pipeline.get_dynamic_models.return_value = []
    pipeline.return_value = "Result"
    
    runner = Runner(
        workload_processor=processor,
        state_manager=state_manager,
        fetcher=fetcher,
        loop_manager=loop_manager,
        result_saver=result_saver
    )
    
    # Створюємо заглушку для config
    config_mock = MagicMock()
    
    runner.run(pipeline=pipeline, raw_workload="dummy", config=config_mock)
    
    # Перевіряємо, що пайплайн був викликаний ТІЛЬКИ для другої задачі (перша пропущена)
    pipeline.assert_called_once_with(config=config_mock, workload=mock_task2)
    
    # Перевіряємо, що після виконання статус задачі 2 змінився на 'completed'
    state_manager.mark_completed.assert_called_once_with("2")
    
    # Перевіряємо, що результат був збережений
    result_saver.save.assert_called_once()
