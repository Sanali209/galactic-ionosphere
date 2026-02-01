[DESIGN_DOC]
Context:
- Problem: Disjointed selection systems (SelectionManager vs Image Viewer) and lack of unified data context synchronization for detections and ROI.
- Constraints: Avoid creating redundant selection managers. Leverage existing MVVM (BindableProperty). Maintain performance with large collection sets.

Architecture:
- Components:
  - `ContextSyncManager`: Central bus for property synchronization.
  - `BindableProperty`: Extended with `sync_channel` metadata.
  - `BindableList/Dict`: Reactive wrappers for Python collections.
  - `PropertiesViewModel` & `ImageViewerViewModel`: Shared state via sync channels.
- Data flow:
  - VM Property Change -> BindableProperty.__set__ -> ContextSyncManager.publish -> Other VMs linked to channel.
  - Mutation (List) -> collectionChanged Signal -> ContextSyncManager broadcast -> Subscriber update.

Key Decisions:
- [D1] Global Reactive Property Sync – Use a central registry of properties tagged with channels to propagate state changes without explicit wiring.
- [D2] Reactive Collection Wrappers – Wrap lists/dicts in proxy objects that emit signals to support synchronization of complex data structures.
- [D3] Diagnostics Interface – Implement `ContextMonitorPanel` to expose internal state of the sync manager for real-time debugging.
- [D4] Debounced Updates – Introduce ViewModel-level debouncing for performance optimization when handling high-frequency selection changes.

Interfaces:
- `ContextSyncManager.publish(channel, value, source_vm)`
- `BindableProperty(sync_channel=..., mapper=...)`
- `PropertiesViewModel.loading_requested`: Signal for debounced loading.

Assumptions & TODOs:
- Completed: Phase 1-6 of Reactive Sync refactor.
  - [High] Implement `sync_channel` in `bindable.py`.
  - [High] Implement `ContextSyncManager` service.
  - [Med] Implement `BindableList` and `BindableDict`.
[/DESIGN_DOC]
