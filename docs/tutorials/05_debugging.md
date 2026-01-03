# Tutorial: Debugging

## Logs

USCore uses `loguru` for logging. Logs are printed to console and saved to `logs/app.log`.

-   **Trace**: Detailed flow information.
-   **Debug**: Component state changes.
-   **Info**: Major lifecycle events.
-   **Error**: Exceptions and failures.

## Common Issues

### "No running event loop"
-   **Cause**: Calling async code from a synchronous Qt slot without `asyncio.create_task` or `qasync.asyncSlot`.
-   **Fix**: Decorate your slot with `@asyncSlot()`.

```python
from qasync import asyncSlot

@asyncSlot()
async def on_button_click(self):
    await some_async_function()
```

### "Service not registered"
-   **Cause**: Requesting a service via `locator.get_system()` before it has been registered.
-   **Fix**: Ensure `ApplicationBuilder.add_system()` is called or check dependency order.

### UI Freezes
-   **Cause**: Running heavy sync code on the main thread.
-   **Fix**: Offload to `ThreadPoolExecutor` or `TaskSystem`.

```python
# BAD
result = heavy_computation()

# GOOD
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, heavy_computation)
```
