# Diffusers VPS Toolbox

**Diffusers VPS Toolbox** — це гнучкий, слабо зв'язаний Python-фреймворк (обгортка) для зручного запуску кастомних генеративних пайплайнів (на базі Diffusers, Transformers тощо). 

Головна мета проєкту — **відділити логіку самої генерації** від інфраструктурних задач: завантаження моделей, оркестрації, управління чергами, збереження станів та вводу-виводу результатів. Це дозволяє легко створювати CLI, Jupyter або GUI інтерфейси без дублювання базового коду.

## 🌟 Основні можливості

- **Абстрактні інтерфейси (ABCs)**: Пайплайни та інфраструктурні компоненти ізольовані один від одного через Dependency Injection.
- **Динамічний реєстр моделей**: Файл `models.yml` слугує джерелом правди. З нього автоматично генеруються Python Enums для безпечного та типізованого використання моделей у коді.
- **Model Downloader**: Автоматично перевіряє та завантажує потрібні пайплайну моделі з HuggingFace, CivitAI (через API) або за прямими URL перед початком генерації.
- **Сувора типізація (Pydantic)**: Всі вхідні дані (JSON workload) та параметри конфігурації проходять сувору валідацію через Pydantic.
- **State Manager (SQLite)**: Зберігає статус виконання завдань (`pending`, `completed`, `failed`). У разі падіння процесу, при наступному запуску виконані задачі будуть автоматично пропущені.
- **Loop Manager**: Обробляє чергу навантаження (працює за принципом FIFO).
- **Result Saver**: Зберігає результати роботи пайплайну у потрібну структуру каталогів на основі метаданих.

## 📂 Структура проєкту

Фреймворк використовує `src-layout` та повністю готовий для встановлення як Python пакет:

```text
/
├── src/diffusers_vps_toolbox/
│   ├── core/                   # Базові інтерфейси (ABC) та контракт BaseGenerationPipeline
│   ├── components/             # Реалізація LoopManager, StateManager, Downloader, WorkloadProcessor
│   ├── orchestrator/           # Клас Runner (головний оркестратор потоку)
│   ├── pipelines/              # Ваші кастомні пайплайни генерації (бізнес-логіка)
│   └── registry/               # YAML реєстр моделей та скрипт кодогенерації Enums
├── data/                       # Локальна службова папка для БД та завантажених ваг (ігнорується git)
├── tests/                      # Модульні тести (pytest)
└── main.py                     # Демонстраційний скрипт інтеграції E2E
```

## 🚀 Встановлення

Для роботи потрібен **Python 3.13+**. Оскільки це незалежна бібліотека, ви можете легко додати її до свого існуючого проєкту за допомогою `uv` або `pip`, використовуючи посилання на Git-репозиторій:

```bash
# Якщо використовуєте uv:
uv add git+https://github.com/khar-ma2/diffusers-vps-toolbox.git@main

# Якщо використовуєте pip:
pip install git+https://github.com/khar-ma2/diffusers-vps-toolbox.git@main
```

*(Якщо ж ви хочете локально розробляти саму бібліотеку, ви можете клонувати репозиторій та виконати `uv sync` або `pip install -e .`)*

## 🛠️ Як додати нову модель?

1. Створіть власний файл `models.yml` у своєму проєкті.
2. Додайте вашу модель у відповідну категорію. Наразі підтримуються такі провайдери як:
  - HuggingFace
  - CivitAI
  - Direct URL
   
   ```yaml
   Checkpoints:
     my_new_model:
       provider: huggingface
       repo_id: username/model-id
   ```
3. Запустіть скрипт кодогенерації за допомогою вбудованої CLI утиліти:
   ```bash
   toolbox-gen-enums --input models.yml --output my_enums.py
   ```
4. У вашому пайплайні тепер можна імпортувати згенерований `my_enums.py` і використовувати `Checkpoints.MY_NEW_MODEL`.

## 📖 Приклад використання

Скрипт `main.py` в корені проєкту наочно показує, як ініціалізувати всі компоненти, передати їх в `Runner` та запустити генерацію.

```python
from diffusers_vps_toolbox.orchestrator.runner import Runner
from diffusers_vps_toolbox.components.workload_processor import PydanticWorkloadProcessor
from diffusers_vps_toolbox.components.state_manager import SQLiteStateManager
from diffusers_vps_toolbox.components.model_downloader import ModelDownloader
from diffusers_vps_toolbox.components.loop_manager import LoopManager
from diffusers_vps_toolbox.components.result_saver import LocalResultSaver
from diffusers_vps_toolbox.pipelines.example_flux_pipeline import ExampleFluxPipeline, FluxWorkload

def main():
    # 1. Ініціалізуємо інфраструктурні компоненти
    runner = Runner(
        workload_processor=PydanticWorkloadProcessor(FluxWorkload),
        state_manager=SQLiteStateManager("data/state.db"),
        downloader=ModelDownloader("data/models_cache"),
        loop_manager=LoopManager(),
        result_saver=LocalResultSaver("data/outputs")
    )
    
    # 2. Виконуємо пайплайн для масиву задач (JSON-like)
    runner.run(pipeline=ExampleFluxPipeline(), raw_workload=[...], config=...)
```

## 🧪 Запуск тестів

Код покритий модульними тестами, які можна запустити за допомогою `pytest`:

```bash
# Якщо використовуєте uv
uv run pytest
```
