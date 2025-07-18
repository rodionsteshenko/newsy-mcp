"""
Tools Module for Function Schema Generation and Tool Management

Features:
1. Tool Class Management
   - Tool dataclass for function metadata and schema storage
   - Serialization support via to_dict() and from_dict()
   - Schema validation and generation
   - Debug logging support

2. Docstring Parsing
   - Google-style docstring parsing with griffe
   - Function signature validation
   - Parameter extraction and validation
   - Error handling for mismatched signatures

3. Schema Generation
   - OpenAI function schema generation
   - Parameter type and description extraction
   - Required parameter tracking
   - Schema validation

4. Type Conversion
   - Support for basic types (int, float, bool, str)
   - List handling with JSON parsing
   - Fallback to comma-separated values
   - Detailed error reporting

5. Tool Execution
   - Async and sync function support
   - Type conversion of arguments
   - Error handling and reporting
   - Debug logging of execution

6. Tool Generation
   - Combined schema and function wrapping
   - Optional function override support
   - Debug logging of tool creation
"""

import inspect
import json
from griffe import (
    Docstring,
    DocstringSectionText,
    DocstringSectionParameters,
)
from typing import get_origin, get_args
from dataclasses import dataclass
from utils import dprint


@dataclass
class Tool:
    name: str
    func: callable
    schema: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tool":
        return cls(data["name"], None, data["schema"])


def parse_docstring(func) -> dict:
    """
    Parses the docstring of the given function to extract description and parameters.

    Args:
        func (callable): The function to parse.

    Returns:
        dict: Parsed docstring details.
    """
    function_name = func.__name__
    function_args = list(inspect.signature(func).parameters.keys())

    text = str(func.__doc__)
    docstring = Docstring(text)
    parsed = docstring.parse("google")

    docstring_description = None
    docstring_args = []

    for section in parsed:
        if isinstance(section, DocstringSectionText):
            docstring_description = section.value
        elif isinstance(section, DocstringSectionParameters):
            docstring_args = [arg.as_dict() for arg in section.value]

    docstring_args_names = [d["name"] for d in docstring_args]
    if docstring_args_names != function_args:
        print(f"Error: Docstring args must match function args for {function_name}")
        raise AssertionError("Docstring args must match function args")

    dprint(f"Successfully parsed docstring for {function_name}")
    return {
        "function_name": function_name,
        "function_args": function_args,
        "docstring_description": docstring_description,
        "docstring_args": docstring_args,
    }


def generate_tool_schema(func) -> dict:
    """
    Generates the tool schema for a given function using its signature and docstring.

    Args:
        func (callable): The function for which to generate the schema.

    Returns:
        dict: The tool schema as a dictionary.
    """
    docstring_details = parse_docstring(func)
    dprint(f"Generating schema for {func.__name__}")

    parameters = {}
    for param in docstring_details["docstring_args"]:
        parameters[param["name"]] = {
            "type": param.get("type", "string"),
            "description": param.get("description", ""),
        }

    tool_schema = {
        "type": "function",
        "function": {
            "name": docstring_details["function_name"],
            "description": docstring_details["docstring_description"],
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": docstring_details["function_args"],
            },
        },
    }

    return tool_schema


def convert_type(value, target_type):
    """
    Converts the given value to the specified target type.

    Args:
        value (str): The value to convert.
        target_type (type): The type to which the value should be converted.

    Returns:
        The value converted to the target type.
    """
    # Handle case where no type annotation was provided
    if target_type == inspect._empty:
        return value  # Return value as-is when no type hint is specified

    origin = get_origin(target_type)
    args = get_args(target_type)

    try:
        if target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is bool:
            return value.lower() in ("true", "1")
        elif target_type is str:
            return value
        elif origin is list and args[0] is str:
            if isinstance(value, list):
                return value
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value.split(",")  # Fallback to comma-separated values
        else:
            print(f"Error: Unsupported type conversion - {target_type}")
            raise ValueError(f"Unsupported target type: {target_type}")
    except Exception as e:
        print(f"Error converting value '{value}' to type {target_type}: {e}")
        raise


async def call_tool_by_func(func, tool_args):
    """
    Calls a tool by func with the provided arguments.

    Args:
        func (Callable): The function to call.
        tool_args (dict): The arguments to pass to the tool.

    Returns:
        The result of the tool call.

    Raises:
        ValueError: If argument conversion fails
        RuntimeError: If tool execution fails
        Any other exceptions that the tool might raise
    """
    # Add check for None function
    if func is None:
        dprint("Error: Received None instead of a callable function", error=True)
        raise ValueError("Tool function is None")

    try:
        sig = inspect.signature(func)
        try:
            typed_args = {
                k: convert_type(v, sig.parameters[k].annotation)
                for k, v in tool_args.items()
            }
        except Exception as e:
            dprint(f"Error converting arguments: {e}", error=True)
            raise ValueError(f"Failed to convert arguments: {e}")

        dprint(f"Calling {func.__name__} with args: {typed_args}")
        if inspect.iscoroutinefunction(func):
            result = await func(**typed_args)
        else:
            result = func(**typed_args)

        return result
    except Exception as e:
        if func is not None:
            dprint(f"Error executing tool {func.__name__}: {e}", error=True)
        else:
            dprint(f"Error executing tool: {e}", error=True)
        raise RuntimeError(f"Tool execution failed: {str(e)}")


def gen_tool(func: callable, real_func: call_tool_by_func = None) -> Tool:
    tool = Tool(func.__name__, func, generate_tool_schema(func))
    if real_func:
        tool.func = real_func
    dprint(f"Generated tool for {func.__name__}")
    return tool
