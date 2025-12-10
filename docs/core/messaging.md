# Messaging Protocol (`src.core.messaging`)

A standardized way for "Workers" (background tasks) and "Core" components to communicate status updates to the UI or Logs.

## `SystemMessage`

A DTO (Data Transfer Object) containing:
-   `level`: `INFO`, `WARNING`, `ERROR`, `SUCCESS`.
-   `topic`: `SYSTEM`, `AI`, `DB`, etc.
-   `body`: Human-readable text.
-   `payload`: Optional raw data or exceptions.

## `MessageBuilder`

Helper to create valid messages easily.

```python
# via ServiceLocator
msg = sl.msg_builder.info("AI_ENGINE", "Processing started")
sl.broadcast(msg)
```

## Receiving Messages

The UI (or other consumers) connects to the global bus:

```python
sl.bus.connect(self.on_message)
```
