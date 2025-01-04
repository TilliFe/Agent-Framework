from pydantic import ValidationError
from typing import Dict, Any
import json


def is_valid_json(json_data, model_class):
    """Ensure that the JSON data is valid according to the specified Pydantic model."""
    try:
        _ = model_class.model_validate(json_data)
        return True, None
    except ValidationError as e:
        return False, e
    except Exception as e:
        return False, e


def extract_json(message):
    """Extract JSON data from a message."""
    try:
        if message.startswith("```json\n"):
            message = message[7:]
        elif message.startswith("```json"):
            message = message[6:]
        elif message.startswith("```"):
            message = message[3:]
        elif message.startswith("\n```json"):
            message = message[:-7]
        if message.endswith("```\n"):
            message = message[:-4]
        elif message.endswith("\n```"):
            message = message[:-4]
        elif message.endswith("```"):
            message = message[:-3]
        parsed_message = json.loads(message)
        return parsed_message
    except:
        print("extract_json: Error parsing JSON")
        return message


def append_to_state(state: Dict, key: Any, value: Any) -> Dict:
    """
    Append a value to a list in the state.
    """
    state[key].extend(value)
    return state


def set_state_value(state: Dict, key: Any, value: Any) -> Dict:
    """
    Set a value in the state.
    """
    state[key] = value
    return state


class HumanMessage:
    """Represents a message from a human user in the conversation."""

    def __init__(self, content="Hi, How can you help me?"):
        self.role = "human"
        self.content = content

    def __call__(self):
        return {"role": self.role, "content": self.content}


class ToolResponse:
    """Represents a response from a tool execution."""

    def __init__(self, name=None, content=None):
        self.name = name
        self.content = content

    def __call__(self):
        return {"name": self.name, "content": self.content}


class ToolMessage:
    """Represents a message containing tool execution results."""

    def __init__(self, content=None):
        self.role = "tool_response"
        self.content = content

    def __call__(self):
        return {"role": self.role, "content": self.content}


class AIMessage:
    """Represents a message from the AI assistant, including potential tool calls."""

    def __init__(self, content="Hi, how can I assist you today?", tool_calls=None):
        self.role = "AI"
        self.content = content
        self.tool_calls = tool_calls

    def __call__(self):
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
        }
