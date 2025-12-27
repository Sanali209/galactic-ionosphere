# Image Rating Application Consolidation Plan

## **Objective:**
Solidify the application following Single Source of Truth (SSOT) principle through systematic consolidation, factory patterns, and architectural improvements.

## **Current Issues:**

### **1. Data Management Duplication**
- 3 separate data access patterns: `data_manager` module, direct `all_annotations`, scattered `manual_voted_list`
- No centralized data validation or business rules
- Cache management duplicated across 4+ modules

### **2. UI Logic Scattered**
- Duplicate dialog classes in different files
- Helper functions spread across `ui_helpers.py`, `rating_helpers.py`, inline methods
- Configuration split between `constants.py`, `model_config.py`

### **3. Business Logic Fragmentation**
- Rating calculations appear in 6+ locations
- Error handling patterns not standardized
- Validation logic duplicated

## **Consolidation Strategy:**

---

## **Phase 1: Core Services Consolidation** 🔴 *START HERE*

### **A. Service Container & Factory Pattern**
**Goal:** Single access point for all services following Factory pattern

**Implementation:**
```python
class ServiceContainer:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._services = {}
        self._register_services()

    def get_service(self, service_type):
        return self._services.get(service_type)

    def _register_services(self):
        self._services[DataService] = DataService()
        self._services[CacheService] = CacheService()
        self._services[ConfigurationService] = ConfigurationService()
```

### **B. Repository Pattern Implementation**
**Goal:** Consolidate all data access through single repository interface

**Files to Create:**
- `services/data_service.py` - Unified data operations
- `repository/annotation_repository.py` - Single data access layer

### **C. Configuration Consolidation**
**Goal:** Single source for all application configuration

**Implementation:**
```python
class ApplicationConfig:
    def __init__(self):
        self._load_from_constants_py()
        self._load_from_model_config_py()
        self._merge_and_validate()
```

---

## **Phase 2: Business Logic Consolidation**

### **A. Rating Service Unification**
- Consolidate rating calculations from:
  - `rating_helpers.py`
  - Inline calculations in `image_rating_gui_v4.py`
  - `rating_operations.py` methods

### **B. Validation Service**
- Single validation entry point for:
  - Rating ranges
  - Data integrity
  - UI state validation

### **C. Cache Management Unification**
- Merge cache operations from:
  - `data_manager.py` cache methods
  - Direct cache calls in multiple files
  - Rating-specific cache logic

---

## **Phase 3: UI Architecture Redesign**

### **A. Component Factory Pattern**
```python
class DialogFactory:
    @staticmethod
    def create_cached_relations_editor(record, parent=None):
        return CachedRelationsEditor(record, parent)

    @staticmethod
    def create_confirmation_dialog(message, parent=None):
        return QMessageBox.question(parent, "Confirm", message, ...)
```

### **B. View Model Pattern Implementation**
```python
class RatingListViewModel:
    def __init__(self, data_service):
        self._data_service = data_service
        self.records = ObservableList()

    def load_records(self, anchors_only=False):
        # Single source of truth for data loading
        pass

    def update_rating(self, record_id, direction):
        # Standardized rating modification
        pass
```

### **C. Error Handling Standardization**
- Implement consistent error display
- Centralized error logging
- User-friendly error messages

---

## **Phase 4: Communication Infrastructure**

### **A. Mediator Pattern for Components**
```python
class ApplicationMediator:
    def __init__(self):
        self._components = {}

    def register(self, name, component):
        self._components[name] = component

    def notify(self, from_component, event, data):
        # Handle cross-component communication
        pass
```

### **B. Command Pattern for Actions**
```python
class CommandBus:
    def execute(self, command: Command):
        # Handle user actions uniformly
        pass

class RateImagesCommand(Command):
    def execute(self):
        # Consolidate rating logic
        pass
```

---

## **Phase 5: Dependency Injection Setup**

### **A. Service Locator Pattern**
- Replace direct imports with service resolution
- Enable testability through mock services
- Simplify component initialization

---

## **Implementation Roadmap:**

### **Step 1-5 (Days 1-2): Core Services** 🏆 *COMPLETED*
- [x] Create ServiceContainer and service discovery
- [x] Implement ApplicationConfig consolidation
- [x] Build AnnotationRepository interface
- [x] Create DataService abstraction layer
- [x] Update imports to use services

**✅ Completed Files:**
- `services/service_container.py` - Singleton service registry with lazy loading
- `services/configuration_service.py` - Unified config consolidates constants.py + model_config.py
- `services/data_service.py` - Repository pattern for all data operations
- `services/cache_service.py` - Single cache management interface
- `services/rating_service.py` - Unified TrueSkill and rating calculations
- `services/validation_service.py` - Centralized validation system

### **Step 6-8 (Days 3-4): Business Logic**
- [x] Started migrating rating_operations.py to use services
- [x] Consolidated rating calculations to RatingService
- [x] Implemented service-based normalization
- [x] Updated method calls to delegate to services
- [ ] Complete rating_operations.py migration
- [ ] Consolidate cache operations
- [ ] Update business rule callers

### **Step 9-11 (Days 5-6): UI Architecture**
- [ ] Build component factory
- [ ] Create view model layer
- [ ] Standardize error handling
- [ ] Update UI components

### **Step 12-15 (Days 7-8): Communication Layer**
- [ ] Implement mediator pattern
- [ ] Add command pattern
- [ ] Create notification system
- [ ] Update component communication

---

## **Success Metrics:**
- ✅ **Code Duplication Reduction:** 50% less duplicate code
- ✅ **Import Reduction:** 60% fewer scattered imports
- ✅ **Single Source Files:** <10 key service files
- ✅ **Testability:** Each service independently testable
- ✅ **Maintainability:** Clear change boundaries

---

## **Risk Mitigation:**
- **Gradual Migration:** Each phase maintains compatibility
- **Tests First:** Add tests before consolidation
- **Rollback Plans:** Git branches for each major change
- **Feature Flags:** Disable new patterns if issues arise

---

## **File Structure After Consolidation:**
```
services/
├── __init__.py
├── service_container.py
├── data_service.py
├── cache_service.py
└── configuration_service.py

repository/
├── __init__.py
├── annotation_repository.py
└── interfaces.py

ui/
├── factories/
│   ├── dialog_factory.py
│   └── component_factory.py
├── viewmodels/
│   ├── rating_list_viewmodel.py
│   ├── comparison_viewmodel.py
│   └── base_viewmodel.py
└── mediators/
    └── application_mediator.py
```

**Status:** Ready for implementation starting with Phase 1 services consolidation.
