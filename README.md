# AI Pipeline Toolbox

**AI Pipeline Toolbox** — це гнучкий, слабо зв'язаний Python-фреймворк (обгортка) для зручного запуску кастомних генеративних пайплайнів (на базі Diffusers, Transformers тощо). 

Головна мета проєкту — **відділити логіку самої генерації** від інфраструктурних задач: завантаження моделей, оркестрації, управління чергами, збереження станів та вводу-виводу результатів. Це дозволяє легко створювати CLI, Jupyter або GUI інтерфейси без дублювання базового коду.

## 🌟 Основні можливості

- **Абстрактні інтерфейси (ABCs)**: Пайплайни та інфраструктурні компоненти ізольовані один від одного через Dependency Injection.
- **Динамічний реєстр моделей**: Файл `models.yml` слугує джерелом правди. З нього автоматично генеруються Python Enums для безпечного та типізованого використання моделей у коді.
- **Model Fetcher**: Автоматично перевіряє та завантажує моделі через фоновий демон `aria2c` (за допомогою `aria2p`). Підтримує завантаження як статичних моделей з реєстру, так і динамічних (наприклад, LoRA) з робочого навантаження, автоматично додаючи потрібні токени авторизації для провайдерів `huggingface` та `civitai`.
- **Сувора типізація (Pydantic)**: Всі вхідні дані (JSON workload) та параметри конфігурації проходять сувору валідацію через Pydantic.
- **State Manager (SQLite)**: Зберігає статус виконання завдань (`pending`, `completed`, `failed`). У разі падіння процесу, при наступному запуску виконані задачі будуть автоматично пропущені.
- **Loop Manager**: Обробляє чергу навантаження (працює за принципом FIFO).
- **Result Saver**: Зберігає результати роботи пайплайну у потрібну структуру каталогів на основі метаданих.

## 📂 Структура проєкту

Фреймворк використовує `src-layout` та повністю готовий для встановлення як Python пакет:

```text
/
├── src/ai_pipeline_toolbox/
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
uv add git+https://github.com/khar-ma2/ai-pipeline-toolbox.git@main

# Якщо використовуєте pip:
pip install git+https://github.com/khar-ma2/ai-pipeline-toolbox.git@main
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
       download_url: https://huggingface.co/username/model
   ```
3. Запустіть скрипт кодогенерації за допомогою вбудованої CLI утиліти:
   ```bash
   toolbox-gen-enums --input models.yml --output my_enums.py
   ```
4. У вашому пайплайні тепер можна імпортувати згенерований `my_enums.py` і використовувати `Checkpoints.MY_NEW_MODEL`.

### Динамічні моделі (LoRA, тощо)
Якщо вам потрібно завантажити модель, якої немає в `models.yml` (наприклад, користувач передав лінк на кастомну LoRA), використовуйте клас `DynamicModel`. Він суворо типізований через Pydantic. 
Також фреймворк має вбудований хелпер `resolve_air_urn`, що дозволяє автоматично конвертувати AIR/URN формати (наприклад `urn:air:civitai:lora:12345@67890`) у прямі посилання на завантаження.

```python
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.core.helpers import resolve_air_urn
from my_enums import Provider, Category, DiT

# Створюємо типізований запит на завантаження
lora_model = DynamicModel(
    url=resolve_air_urn("urn:air:civitai:lora:12345@67890"),
    provider=Provider.CIVITAI,
    category=Category.DIT, # або просто клас DiT
    filename="my_custom_lora.safetensors"
)
```
ModelFetcher автоматично валідуватиме передані категорії, уникаючи створення неправильних директорій.

## 📖 Як писати кастомні пайплайни

Щоб створити власний пайплайн (наприклад, для Stable Diffusion або аудіо генерації), ви повинні наслідувати `BaseGenerationPipeline` і вказати типи конфігурації, завдання та результату за допомогою дженериків:

```python
from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from pydantic import BaseModel
from PIL import Image

class MyConfig(BaseModel):
    steps: int = 20

class MyWorkload(BaseModel):
    prompt: str

# Вказуємо, що пайплайн приймає MyConfig, MyWorkload і повертає Image.Image
class MyImagePipeline(BaseGenerationPipeline[MyConfig, MyWorkload, Image.Image]):
    required_models = [Checkpoints.MY_MODEL]
    
    def setup(self, models_paths):
        # Ініціалізуємо ваги моделей
        self.pipe = SomePipeline.from_pretrained(models_paths[Checkpoints.MY_MODEL])
        
    def __call__(self, config: MyConfig, workload: MyWorkload) -> Image.Image:
        # Pydantic вже провалідував config і workload
        image = self.pipe(workload.prompt, num_inference_steps=config.steps).images[0]
        return image # Повертаємо об'єкт картинки
```

## 💾 Як писати Result Savers

Фреймворк використовує Dependency Injection та дженерики для суворого контролю форматів збереження. Щоб ваш пайплайн зберігав результати правильно, створіть або використайте існуючий `ResultSaver`, який приймає той самий тип даних, що повертає пайплайн.

Приклад створення сейвера для тексту:

```python
from ai_pipeline_toolbox.core.interfaces import BaseResultSaver
from typing import Dict, Any

class TextResultSaver(BaseResultSaver[str]): # Працює тільки зі str
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def save(self, result: str, metadata: Dict[str, Any]) -> str:
        # Логіка збереження текстового файлу
        ...
```

## 🚀 Як використовувати фреймворк (Запуск задач)

Скрипт `main.py` в корені проєкту наочно показує, як ініціалізувати всі компоненти, об'єднати їх в `Runner` та запустити виконання.

```python
from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.workload_processor import PydanticWorkloadProcessor
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager
from ai_pipeline_toolbox.components.result_saver import TextResultSaver
from ai_pipeline_toolbox.pipelines.example_flux_pipeline import ExampleFluxPipeline, FluxWorkload

def main():
    # 1. Ініціалізуємо інфраструктурні компоненти
    runner = Runner(
        workload_processor=PydanticWorkloadProcessor(FluxWorkload),
        state_manager=SQLiteStateManager("data/state.db"),
        fetcher=ModelFetcher("data/models_cache"),
        loop_manager=LoopManager(),
        result_saver=TextResultSaver("data/outputs") # Тип сейвера має збігатися з ReturnType пайплайну
    )
    
    # 2. Виконуємо пайплайн для масиву задач
    runner.run(
        pipeline=ExampleFluxPipeline(), 
        raw_workload=[{"task_id": "1", "prompt": "cat"}], 
        config=...
    )
```

## 🧪 Запуск тестів

Код покритий модульними тестами, які можна запустити за допомогою `pytest`:

```bash
# Якщо використовуєте uv
uv run pytest
```
