# -*- coding: utf-8 -*-
"""
String Node Executors - Execution logic for string operations.
"""
from .node_executor import BaseNodeExecutor, register_executor


class StringConcatExecutor(BaseNodeExecutor):
    """Concatenate strings."""
    
    async def execute(self, node, context, executor):
        a = executor.evaluate_input(node, "a") or ""
        b = executor.evaluate_input(node, "b") or ""
        sep = executor.evaluate_input(node, "separator") or ""
        
        result = f"{a}{sep}{b}"
        node.set_output("result", result)


class StringSplitExecutor(BaseNodeExecutor):
    """Split string."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        delimiter = executor.evaluate_input(node, "delimiter") or ","
        
        parts = text.split(delimiter) if text else []
        node.set_output("parts", parts)
        node.set_output("count", len(parts))


class StringReplaceExecutor(BaseNodeExecutor):
    """Replace in string."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        search = executor.evaluate_input(node, "search") or ""
        replace = executor.evaluate_input(node, "replace") or ""
        
        result = text.replace(search, replace) if search else text
        node.set_output("result", result)


class StringFormatExecutor(BaseNodeExecutor):
    """Format string with placeholders."""
    
    async def execute(self, node, context, executor):
        template = executor.evaluate_input(node, "template") or ""
        args = []
        for i in range(4):
            arg = executor.evaluate_input(node, f"arg{i}")
            if arg is not None:
                args.append(arg)
        
        try:
            result = template.format(*args)
        except (IndexError, KeyError):
            result = template
        
        node.set_output("result", result)


class StringLengthExecutor(BaseNodeExecutor):
    """Get string length."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        node.set_output("length", len(text))


class StringContainsExecutor(BaseNodeExecutor):
    """Check if string contains substring."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        search = executor.evaluate_input(node, "search") or ""
        case_sensitive = executor.evaluate_input(node, "case_sensitive")
        if case_sensitive is None:
            case_sensitive = True
        
        if case_sensitive:
            result = search in text
        else:
            result = search.lower() in text.lower()
        
        node.set_output("contains", result)


class StringStartsWithExecutor(BaseNodeExecutor):
    """Check if string starts with prefix."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        prefix = executor.evaluate_input(node, "prefix") or ""
        
        node.set_output("result", text.startswith(prefix))


class StringEndsWithExecutor(BaseNodeExecutor):
    """Check if string ends with suffix."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        suffix = executor.evaluate_input(node, "suffix") or ""
        
        node.set_output("result", text.endswith(suffix))


class StringTrimExecutor(BaseNodeExecutor):
    """Trim whitespace."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        node.set_output("result", text.strip())


class StringUpperExecutor(BaseNodeExecutor):
    """Convert to uppercase."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        node.set_output("result", text.upper())


class StringLowerExecutor(BaseNodeExecutor):
    """Convert to lowercase."""
    
    async def execute(self, node, context, executor):
        text = executor.evaluate_input(node, "input") or ""
        node.set_output("result", text.lower())


def register_string_executors():
    """Register all string executors."""
    register_executor("StringConcat", StringConcatExecutor())
    register_executor("StringSplit", StringSplitExecutor())
    register_executor("StringReplace", StringReplaceExecutor())
    register_executor("StringFormat", StringFormatExecutor())
    register_executor("StringLength", StringLengthExecutor())
    register_executor("StringContains", StringContainsExecutor())
    register_executor("StringStartsWith", StringStartsWithExecutor())
    register_executor("StringEndsWith", StringEndsWithExecutor())
    register_executor("StringTrim", StringTrimExecutor())
    register_executor("StringUpper", StringUpperExecutor())
    register_executor("StringLower", StringLowerExecutor())
