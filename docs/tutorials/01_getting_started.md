# Tutorial: Getting Started

This guide covers how to set up the development environment and run UExplorer.

## Prerequisites

-   Python 3.10+
-   MongoDB 6.0+
-   Visual Studio Code (Recommended)

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YourUser/USCore.git
    cd USCore
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Database**
    -   Ensure MongoDB is running on `localhost:27017`.
    -   Or copy `config.example.json` to `config.json` and edit database settings.

## Running UExplorer

```bash
python samples/uexplorer/main.py
```

## Running Tests

```bash
pytest tests/
```
