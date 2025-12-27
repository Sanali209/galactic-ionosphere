# USCore Documentation

Welcome to the USCore Project Documentation. This hub contains detailed information about the project's architecture, core components, and modules.

## 📚 Contents

- **[Architecture](architecture.md)**: High-level overview of the system, directory structure, and core design patterns.
- **[Foundation](foundation.md)**: Deep dive into the core framework, including the Service Locator, Event System, and Configuration.
- **[Modules](modules.md)**: Documentation for specific subsystems like UCoreFS (Filesystem DB) and NodeGraph.
- **[Tutorials](tutorials.md)**: Step-by-step guides for common development tasks.

## Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the UExplorer Sample**:
    ```bash
    python samples/uexplorer/main.py
    ```

3.  **Run Tests**:
    ```bash
    pytest tests/
    ```

## Project Overview

USCore is a professional PySide6 desktop application framework designed for building complex, modular tools. It emphasizes:
- **Dependency Injection**: Via a robust Service Locator.
- **Event-Driven Architecture**: Supporting both synchronous observers and asynchronous event buses.
- **Visual Programming**: Integrated Node Graph engine.
- **AI Integration**: Built-in support for vector embeddings and semantic search (UCoreFS).
