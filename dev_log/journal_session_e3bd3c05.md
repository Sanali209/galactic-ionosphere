# Session Journal: Process-Based Heavy Task Handlers

**Session ID**: e3bd3c05-9d33-48bb-baf8-8c2ec9a5b29d  
**Date**: 2026-01-17  
**Status**: Planning Complete

## Objective

Enable dedicated LLM worker process for CPU-heavy AI inference, with persistent model loading.

## Research Findings

### Existing Infrastructure (DISCOVERED)

| Component | Location | Status |
|-----------|----------|--------|
| `LLMWorkerService` | `src/core/llm/worker_service.py` | **DISABLED** (line 69) |
| `AITaskQueue` | `src/core/llm/task_queue.py` | **DISABLED** (thread-based) |
| `LLMJobRequest/Result` | `src/core/llm/models.py` | ✓ Ready |
| Bundle Registration | `src/ucorefs/bundle.py:49` | ✓ In place |

### Why Currently Disabled

From `worker_service.py` line 64-70:
```python
# DISABLED: Multiprocessing workers cause UI blocking on Windows.
# The worker process model loading blocks the queue operations.
# TODO: Investigate async-safe inter-process communication
```

### Root Cause Analysis

1. **Windows multiprocessing**: Uses `spawn` by default, but code doesn't specify context
2. **Queue.get() blocking**: Main thread blocks waiting for worker ready
3. **Model loading time**: 5-10 seconds per model blocks initialization

### Solution

1. Explicit `spawn` context for Windows compatibility
2. Pre-load models BEFORE signaling ready
3. Use `__READY__` signal pattern to indicate worker initialization complete
4. Single worker mode (user requirement: resource constraints)

## Design Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | Enable existing `LLMWorkerService` | Infrastructure already built, just disabled |
| D2 | Single worker, models pre-loaded | User requirement: 1 worker, no load/unload |
| D3 | `spawn` context + ready signal | Fix Windows blocking issue |
| D4 | Config flag `llm_workers.enabled` | Safe rollout, easy fallback |

## Next Steps

1. User review implementation plan
2. Add `enabled` flag to config
3. Fix `worker_service.py` blocking issue
4. Test with single image processing
5. Verify UI responsiveness
