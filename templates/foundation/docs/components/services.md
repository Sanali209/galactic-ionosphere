# Core Services

## Asset Manager (`AssetManager`)

Handles ingestion and management of digital assets.

```python
assets = sl.get_system(AssetManager)
await assets.ingest("C:/Images/photo.jpg")
```

## Journal Service (`JournalService`)

Provides structured logging to the database.

```python
journal = sl.get_system(JournalService)
await journal.log("INFO", "User", "User logged in")
```

## Command Bus (`CommandBus`)

Decouples UI inputs from business logic actions. See [Command Pattern](../architecture/commands.md) for details.
