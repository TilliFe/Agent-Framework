import inspect
from typing import Optional, Any, Union
from .runnable import Runnable, CallbackManager


class Tool(Runnable):
    """Converts Python functions into tool schemas that can be used by the framework."""

    def __init__(self, function):
        self.executable = function
        self.tool_json = self.create_tool(function)

    def create_tool(self, func):
        """Converts a Python function into a JSON schema describing its interface."""

        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)

        schema = {
            "name": func.__name__,
            "description": docstring if docstring else "",
            "arguments": {"type": "object", "properties": {}, "required": []},
        }

        for param_name, param in signature.parameters.items():
            param_type = param.annotation

            if param_type is inspect.Parameter.empty:
                schema["arguments"]["properties"][param_name] = {
                    "description": param_name,
                    "type": "string",
                }
            elif isinstance(param_type, type):
                schema["arguments"]["properties"][param_name] = {
                    "description": param_name,
                    "type": self.python_type_to_json_type(param_type),
                }
            elif getattr(param_type, "__origin__", None) is Union:
                union_types = getattr(param_type, "__args__", [])
                schema["arguments"]["properties"][param_name] = {
                    "description": param_name,
                    "anyOf": [
                        {"type": self.python_type_to_json_type(t)}
                        for t in union_types
                        if t is not type(None)
                    ],
                }
                if type(None) in union_types:
                    schema["arguments"]["properties"][param_name]["nullable"] = True
            elif getattr(param_type, "__origin__", None) is list:
                item_type = getattr(param_type, "__args__", [inspect.Parameter.empty])[
                    0
                ]
                schema["arguments"]["properties"][param_name] = {
                    "description": param_name,
                    "type": "array",
                    "items": (
                        {"type": self.python_type_to_json_type(item_type)}
                        if item_type is not inspect.Parameter.empty
                        else {"type": "string"}
                    ),
                }

            elif getattr(param_type, "__origin__", None) is dict:
                schema["arguments"]["properties"][param_name] = {
                    "description": param_name,
                    "type": "object",
                    "additionalProperties": True,
                }
            else:
                schema["arguments"]["properties"][param_name] = {
                    "description": param_name,
                    "type": "string",
                }

            if param.default is inspect.Parameter.empty:
                schema["arguments"]["required"].append(param_name)

        return schema

    def python_type_to_json_type(self, python_type):
        """Maps Python types to their corresponding JSON Schema type names."""
        if python_type == int:
            return "integer"
        elif python_type == float:
            return "number"
        elif python_type == str:
            return "string"
        elif python_type == bool:
            return "boolean"
        elif python_type == list:
            return "array"
        elif python_type == dict:
            return "object"
        elif python_type == type(None):
            return "null"
        else:
            return "string"

    def invoke(
        self, arguments, callbacks: Optional[CallbackManager] = None, **kwargs: Any
    ):
        """Executes the wrapped function with the given arguments and manages callbacks."""
        if callbacks:
            callbacks.on_invoke_start(self.tool_json["name"], arguments, **kwargs)
        result = self.executable(**arguments)
        if callbacks:
            callbacks.on_invoke_end(self.tool_json["name"], result, **kwargs)
        return result


def tool(func):
    """Decorator that wraps a Python function into a Tool instance."""
    return Tool(func)
