# Command Pattern

We use the Command Pattern to decouple the User Interface from business logic. This ensures that the UI does not know *how* an action is performed, only *what* action to request.

## Interaction Diagram (UML)

*(See [command_flow.puml](command_flow.puml))*

![Command Flow](command_flow.puml)

## Components

### 1. ICommand
A simple data class implementing `ICommand` (marker interface).

```python
@dataclass
class CreateUserCommand(ICommand):
    username: str
    role: str
```

### 2. ICommandHandler
Logic that executes the command.

```python
class CreateUserHandler(ICommandHandler[CreateUserCommand]):
    async def handle(self, command: CreateUserCommand):
        # Business logic here
        print(f"Creating user {command.username}")
```

### 3. CommandBus
The registry and dispatcher.

```python
# Registration (usually in main.py)
bus = sl.get_system(CommandBus)
bus.register(CreateUserCommand, CreateUserHandler())

# Dispatching (from UI)
await bus.dispatch(CreateUserCommand("alice", "admin"))
```

## Benefits
- **Testability**: Handlers can be tested in isolation.
- **Reusability**: Commands can be triggered from UI, CLI, or API.
- **History**: Commands can be logged or undone (if undo logic is added).
