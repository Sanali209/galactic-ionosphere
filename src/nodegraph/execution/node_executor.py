# -*- coding: utf-8 -*-
"""
Node Executors - Handlers for executing specific node types.

Each node type has an executor that knows how to run it,
compute outputs, and handle flow control.
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from ..core.base_node import BaseNode
    from .executor import ExecutionContext, GraphExecutor


class BaseNodeExecutor(ABC):
    """
    Base class for node executors.
    
    Each node type can have a specialized executor that
    handles its specific execution logic.
    """
    
    @abstractmethod
    async def execute(
        self, 
        node: 'BaseNode', 
        context: 'ExecutionContext',
        executor: 'GraphExecutor'
    ):
        """
        Execute the node.
        
        Args:
            node: Node to execute
            context: Current execution context
            executor: Parent graph executor
        """
        pass


# Registry of executors
_executors: Dict[str, BaseNodeExecutor] = {}


def register_executor(node_type: str, executor: BaseNodeExecutor):
    """Register an executor for a node type."""
    _executors[node_type] = executor


def get_executor(node_type: str) -> Optional[BaseNodeExecutor]:
    """Get executor for a node type."""
    return _executors.get(node_type)


# =============================================================================
# Event Node Executors
# =============================================================================

class StartExecutor(BaseNodeExecutor):
    """Executor for Start node - just passes through."""
    
    async def execute(self, node, context, executor):
        context.log(node, "Execution started")


class UpdateExecutor(BaseNodeExecutor):
    """Executor for Update/Tick node."""
    
    async def execute(self, node, context, executor):
        # Would be called each frame in a real engine
        node.set_output("delta_time", 0.016)  # ~60fps
        context.log(node, "Tick")


# =============================================================================
# Flow Control Executors
# =============================================================================

class IfExecutor(BaseNodeExecutor):
    """Executor for If node - single path conditional."""
    
    async def execute(self, node, context, executor):
        condition = executor.evaluate_input(node, "condition")
        
        if condition:
            context.log(node, f"Condition is True, executing 'then'")
            await executor.execute_output_pin(node, "then")


class BranchExecutor(BaseNodeExecutor):
    """Executor for Branch node - if/else."""
    
    async def execute(self, node, context, executor):
        condition = executor.evaluate_input(node, "condition")
        
        if condition:
            context.log(node, "Taking TRUE branch")
            await executor.execute_output_pin(node, "true")
        else:
            context.log(node, "Taking FALSE branch")
            await executor.execute_output_pin(node, "false")


class SequenceExecutor(BaseNodeExecutor):
    """Executor for Sequence node - execute outputs in order."""
    
    async def execute(self, node, context, executor):
        # Execute each output in order
        for i in range(4):  # Sequence has 4 outputs
            pin_name = f"then_{i}"
            if node.get_output_pin(pin_name):
                await executor.execute_output_pin(node, pin_name)


class ForLoopExecutor(BaseNodeExecutor):
    """Executor for ForLoop node."""
    
    async def execute(self, node, context, executor):
        first = executor.evaluate_input(node, "first_index") or 0
        last = executor.evaluate_input(node, "last_index") or 0
        
        context.log(node, f"Looping from {first} to {last}")
        
        for i in range(int(first), int(last) + 1):
            # Set index output
            node.set_output("index", i)
            
            # Execute loop body
            await executor.execute_output_pin(node, "loop_body")
            
            # Check for break
            if context._loop_state.get(f"{node.node_id}_break"):
                context._loop_state[f"{node.node_id}_break"] = False
                break
        
        # Execute completed
        await executor.execute_output_pin(node, "completed")


class ForEachLoopExecutor(BaseNodeExecutor):
    """Executor for ForEachLoop node."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        
        context.log(node, f"Iterating over {len(array)} items")
        
        for i, element in enumerate(array):
            # Set outputs
            node.set_output("element", element)
            node.set_output("index", i)
            
            # Execute loop body
            await executor.execute_output_pin(node, "loop_body")
            
            # Check for break
            if context._loop_state.get(f"{node.node_id}_break"):
                context._loop_state[f"{node.node_id}_break"] = False
                break
        
        # Execute completed
        await executor.execute_output_pin(node, "completed")


class WhileLoopExecutor(BaseNodeExecutor):
    """Executor for WhileLoop node."""
    
    async def execute(self, node, context, executor):
        iteration = 0
        max_iterations = 10000  # Safety limit
        
        while iteration < max_iterations:
            condition = executor.evaluate_input(node, "condition")
            
            if not condition:
                break
            
            await executor.execute_output_pin(node, "loop_body")
            iteration += 1
            
            # Check for break
            if context._loop_state.get(f"{node.node_id}_break"):
                context._loop_state[f"{node.node_id}_break"] = False
                break
        
        context.log(node, f"Loop completed after {iteration} iterations")
        await executor.execute_output_pin(node, "completed")


class DoOnceExecutor(BaseNodeExecutor):
    """Executor for DoOnce node."""
    
    async def execute(self, node, context, executor):
        state_key = node.node_id
        
        if not context._do_once_state.get(state_key, False):
            context._do_once_state[state_key] = True
            context.log(node, "Executing (first time)")
            await executor.execute_output_pin(node, "completed")
        else:
            context.log(node, "Already executed, skipping")


class GateExecutor(BaseNodeExecutor):
    """Executor for Gate node."""
    
    async def execute(self, node, context, executor):
        state_key = node.node_id
        start_closed = executor.evaluate_input(node, "start_closed")
        
        # Initialize state
        if state_key not in context._gate_state:
            context._gate_state[state_key] = not start_closed
        
        # This is simplified - would need separate exec handling for open/close/toggle
        if context._gate_state.get(state_key, True):
            await executor.execute_output_pin(node, "exit")


class FlipFlopExecutor(BaseNodeExecutor):
    """Executor for FlipFlop node."""
    
    async def execute(self, node, context, executor):
        state_key = node.node_id
        
        # Toggle state
        is_a = not context._flipflop_state.get(state_key, False)
        context._flipflop_state[state_key] = is_a
        
        node.set_output("is_A", is_a)
        
        if is_a:
            context.log(node, "Flip - executing A")
            await executor.execute_output_pin(node, "A")
        else:
            context.log(node, "Flop - executing B")
            await executor.execute_output_pin(node, "B")


class BreakLoopExecutor(BaseNodeExecutor):
    """Executor for BreakLoop node."""
    
    async def execute(self, node, context, executor):
        # Find parent loop and set break flag
        # This is simplified - would need to track loop stack
        context.log(node, "Break requested")
        # Set global break flag (simplified)
        for key in list(context._loop_state.keys()):
            if "_break" not in key:
                context._loop_state[f"{key}_break"] = True


class DelayExecutor(BaseNodeExecutor):
    """Executor for Delay node."""
    
    async def execute(self, node, context, executor):
        duration = executor.evaluate_input(node, "duration") or 1.0
        context.log(node, f"Waiting {duration}s")
        await asyncio.sleep(float(duration))
        await executor.execute_output_pin(node, "completed")


# =============================================================================
# Variable Executors
# =============================================================================

class GetVariableExecutor(BaseNodeExecutor):
    """Executor for GetVariable node."""
    
    async def execute(self, node, context, executor):
        var_name = node.variable_name
        value = context.get_variable(var_name)
        node.set_output("value", value)
        context.log(node, f"Get '{var_name}' = {value}")


class SetVariableExecutor(BaseNodeExecutor):
    """Executor for SetVariable node."""
    
    async def execute(self, node, context, executor):
        var_name = node.variable_name
        value = executor.evaluate_input(node, "value")
        
        context.set_variable(var_name, value)
        node.set_output("new_value", value)
        
        context.log(node, f"Set '{var_name}' = {value}")
        
        # Notify callbacks
        for cb in executor._on_variable_changed:
            cb(var_name, value)
        
        await executor.execute_output_pin(node, "exec_out")


class IncrementVariableExecutor(BaseNodeExecutor):
    """Executor for IncrementVariable node."""
    
    async def execute(self, node, context, executor):
        var_name = node.variable_name
        amount = executor.evaluate_input(node, "amount") or 1
        
        current = context.get_variable(var_name) or 0
        new_value = current + amount
        
        context.set_variable(var_name, new_value)
        node.set_output("new_value", new_value)
        
        context.log(node, f"'{var_name}' = {current} + {amount} = {new_value}")
        
        await executor.execute_output_pin(node, "exec_out")


# =============================================================================
# Utility Executors
# =============================================================================

class PrintExecutor(BaseNodeExecutor):
    """Executor for Print node."""
    
    async def execute(self, node, context, executor):
        message = executor.evaluate_input(node, "message") or ""
        context.log(node, f"PRINT: {message}")
        print(f"[NodeGraph] {message}")
        await executor.execute_output_pin(node, "exec_out")


class PrintVariableExecutor(BaseNodeExecutor):
    """Executor for PrintVariable node."""
    
    async def execute(self, node, context, executor):
        name = executor.evaluate_input(node, "name") or "var"
        value = executor.evaluate_input(node, "value")
        context.log(node, f"PRINT: {name} = {value}")
        print(f"[NodeGraph] {name} = {value}")
        await executor.execute_output_pin(node, "exec_out")


class MakeArrayExecutor(BaseNodeExecutor):
    """Executor for MakeArray node."""
    
    async def execute(self, node, context, executor):
        items = []
        for i in range(4):
            value = executor.evaluate_input(node, f"item_{i}")
            if value is not None:
                items.append(value)
        node.set_output("array", items)


class ToStringExecutor(BaseNodeExecutor):
    """Executor for ToString node."""
    
    async def execute(self, node, context, executor):
        value = executor.evaluate_input(node, "value")
        node.set_output("string", str(value) if value is not None else "")


class ToIntegerExecutor(BaseNodeExecutor):
    """Executor for ToInteger node."""
    
    async def execute(self, node, context, executor):
        value = executor.evaluate_input(node, "value")
        try:
            node.set_output("integer", int(value) if value is not None else 0)
        except (ValueError, TypeError):
            node.set_output("integer", 0)


class ToFloatExecutor(BaseNodeExecutor):
    """Executor for ToFloat node."""
    
    async def execute(self, node, context, executor):
        value = executor.evaluate_input(node, "value")
        try:
            node.set_output("float", float(value) if value is not None else 0.0)
        except (ValueError, TypeError):
            node.set_output("float", 0.0)


class ToBooleanExecutor(BaseNodeExecutor):
    """Executor for ToBoolean node."""
    
    async def execute(self, node, context, executor):
        value = executor.evaluate_input(node, "value")
        node.set_output("boolean", bool(value))


class IsValidExecutor(BaseNodeExecutor):
    """Executor for IsValid node."""
    
    async def execute(self, node, context, executor):
        value = executor.evaluate_input(node, "value")
        node.set_output("is_valid", value is not None)


# =============================================================================
# Register All Executors
# =============================================================================

def _register_all():
    """Register all built-in executors."""
    # Events
    register_executor("Start", StartExecutor())
    register_executor("Update", UpdateExecutor())
    
    # Flow Control
    register_executor("If", IfExecutor())
    register_executor("Branch", BranchExecutor())
    register_executor("Sequence", SequenceExecutor())
    register_executor("ForLoop", ForLoopExecutor())
    register_executor("ForEachLoop", ForEachLoopExecutor())
    register_executor("WhileLoop", WhileLoopExecutor())
    register_executor("DoOnce", DoOnceExecutor())
    register_executor("Gate", GateExecutor())
    register_executor("FlipFlop", FlipFlopExecutor())
    register_executor("BreakLoop", BreakLoopExecutor())
    register_executor("Delay", DelayExecutor())
    
    # Variables
    register_executor("GetVariable", GetVariableExecutor())
    register_executor("SetVariable", SetVariableExecutor())
    register_executor("IncrementVariable", IncrementVariableExecutor())
    
    # Utilities
    register_executor("Print", PrintExecutor())
    register_executor("PrintVariable", PrintVariableExecutor())
    register_executor("MakeArray", MakeArrayExecutor())
    register_executor("ToString", ToStringExecutor())
    register_executor("ToInteger", ToIntegerExecutor())
    register_executor("ToFloat", ToFloatExecutor())
    register_executor("ToBoolean", ToBooleanExecutor())
    register_executor("IsValid", IsValidExecutor())
    
    # Phase 5 Executors
    from .file_executors import register_file_executors
    from .string_executors import register_string_executors
    from .array_executors import register_array_executors
    from .image_executors import register_image_executors
    from .matplotlib_executors import register_matplotlib_executors
    
    register_file_executors()
    register_string_executors()
    register_array_executors()
    register_image_executors()
    register_matplotlib_executors()


# Auto-register on import
_register_all()
