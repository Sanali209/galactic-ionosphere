# Tech Stack & Dependencies

## Core Technologies

-   **Language**: Python 3.10+
-   **GUI Framework**: PySide6 (Qt for Python) + PySide6-QtAds (Advanced Docking System)
-   **Async Runtime**: `asyncio` + `qasync` (Qt Event Loop integration)
-   **Database**:
    -   **MongoDB** (via `motor`): Primary metadata storage.
    -   **FAISS** (via `numpy`): Vector similarity search.

## AI & Processing

-   **PyTorch**: Deep learning backend.
-   **CLIP**: Image embeddings (OpenAI).
-   **BLIP**: Image captioning (Salesforce).
-   **MTCNN / YOLO**: Face and object detection.
-   **Pillow / OpenCV**: Image processing.

## Key Libraries

| Library | Version | Purpose |
| :--- | :--- | :--- |
| `PySide6` | Latest | Main UI framework. |
| `motor` | Latest | Async MongoDB driver. |
| `pydantic` | V2 | Data validation and settings management. |
| `loguru` | Latest | Structured, thread-safe logging. |
| `facenet-pytorch`| >=2.5.2 | Face detection models. |
| `huggingface_hub` | >=0.16.0 | Model downloading. |
| `pandas` | >=2.0.0 | Data analysis (used in some extractors). |

## Development Tools

-   **Beanie** (Internal): The project uses a custom ODM layer similar to Beanie but lightweight (`src.core.database.orm`).
-   **pytest**: Testing framework.
