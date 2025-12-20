# -*- coding: utf-8 -*-
"""
Array Node Executors - Execution logic for array operations.
"""
import fnmatch
from .node_executor import BaseNodeExecutor, register_executor


class ArrayJoinExecutor(BaseNodeExecutor):
    """Join array into string."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        sep = executor.evaluate_input(node, "separator") or ","
        
        result = sep.join(str(item) for item in array)
        node.set_output("result", result)


class ArrayLengthExecutor(BaseNodeExecutor):
    """Get array length."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        node.set_output("length", len(array))


class ArrayGetExecutor(BaseNodeExecutor):
    """Get element at index."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        index = executor.evaluate_input(node, "index") or 0
        
        try:
            element = array[int(index)]
            node.set_output("element", element)
            node.set_output("valid", True)
        except (IndexError, TypeError):
            node.set_output("element", None)
            node.set_output("valid", False)


class ArraySetExecutor(BaseNodeExecutor):
    """Set element at index."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        index = executor.evaluate_input(node, "index") or 0
        value = executor.evaluate_input(node, "value")
        
        result = list(array)
        try:
            result[int(index)] = value
        except IndexError:
            pass
        
        node.set_output("result", result)


class ArrayAppendExecutor(BaseNodeExecutor):
    """Append element to array."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        element = executor.evaluate_input(node, "element")
        
        result = list(array)
        result.append(element)
        node.set_output("result", result)


class ArrayMergeExecutor(BaseNodeExecutor):
    """Merge two arrays."""
    
    async def execute(self, node, context, executor):
        array1 = executor.evaluate_input(node, "array1") or []
        array2 = executor.evaluate_input(node, "array2") or []
        
        result = list(array1) + list(array2)
        node.set_output("result", result)


class ArrayFilterExecutor(BaseNodeExecutor):
    """Filter array with wildcard pattern."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        pattern = executor.evaluate_input(node, "pattern") or "*"
        
        # Support multiple patterns separated by ;
        patterns = [p.strip() for p in pattern.split(";")]
        
        matched = []
        unmatched = []
        
        for item in array:
            item_str = str(item)
            if any(fnmatch.fnmatch(item_str, p) for p in patterns):
                matched.append(item)
            else:
                unmatched.append(item)
        
        node.set_output("matched", matched)
        node.set_output("unmatched", unmatched)
        node.set_output("count", len(matched))


class ArrayReverseExecutor(BaseNodeExecutor):
    """Reverse array."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        node.set_output("result", list(reversed(array)))


class ArraySortExecutor(BaseNodeExecutor):
    """Sort array."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        descending = executor.evaluate_input(node, "descending") or False
        
        try:
            result = sorted(array, reverse=descending)
        except TypeError:
            result = sorted(array, key=str, reverse=descending)
        
        node.set_output("result", result)


class ArrayUniqueExecutor(BaseNodeExecutor):
    """Remove duplicates."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        
        seen = set()
        result = []
        for item in array:
            try:
                key = item
                if isinstance(item, list):
                    key = tuple(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            except TypeError:
                result.append(item)
        
        node.set_output("result", result)


class ArraySliceExecutor(BaseNodeExecutor):
    """Slice array."""
    
    async def execute(self, node, context, executor):
        array = executor.evaluate_input(node, "array") or []
        start = executor.evaluate_input(node, "start") or 0
        end = executor.evaluate_input(node, "end")
        if end == -1 or end is None:
            end = None
        
        result = list(array)[int(start):end if end is None else int(end)]
        node.set_output("result", result)


def register_array_executors():
    """Register all array executors."""
    register_executor("ArrayJoin", ArrayJoinExecutor())
    register_executor("ArrayLength", ArrayLengthExecutor())
    register_executor("ArrayGet", ArrayGetExecutor())
    register_executor("ArraySet", ArraySetExecutor())
    register_executor("ArrayAppend", ArrayAppendExecutor())
    register_executor("ArrayMerge", ArrayMergeExecutor())
    register_executor("ArrayFilter", ArrayFilterExecutor())
    register_executor("ArrayReverse", ArrayReverseExecutor())
    register_executor("ArraySort", ArraySortExecutor())
    register_executor("ArrayUnique", ArrayUniqueExecutor())
    register_executor("ArraySlice", ArraySliceExecutor())
