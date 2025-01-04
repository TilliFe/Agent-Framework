from typing import Any, List, Optional
from abc import ABC, abstractmethod
from typing import List, Optional
from framework import *


class CallbackHandler(ABC):
    """Base class for handling callbacks during runnable execution."""

    @abstractmethod
    def on_invoke_start(
        self, runnable_name: str, input_data: Any, **kwargs: Any
    ) -> None:
        """Called when a runnable starts execution."""
        pass

    @abstractmethod
    def on_invoke_end(
        self, runnable_name: str, output_data: Any, **kwargs: Any
    ) -> None:
        """Called when a runnable completes execution."""
        pass


class TraceLog:
    """Maintains a list of trace messages for logging purposes."""

    def __init__(self):
        self.logs = []

    def add_log(self, message: str):
        """Adds a new message to the trace log."""
        self.logs.append(message)

    def get_logs(self) -> List[str]:
        """Returns all stored log messages."""
        return self.logs

    def clear_logs(self):
        """Clears all stored log messages."""
        self.logs = []


class TraceCallbackHandler(CallbackHandler):
    """Handles callbacks by storing them in a TraceLog with proper indentation."""

    def __init__(
        self, trace_log: TraceLog, indent_level: int = 0, indent_step: int = 2
    ):
        self.trace_log = trace_log
        self.indent_level = indent_level
        self.indent_step = indent_step

    def on_invoke_start(
        self, runnable_name: str, input_data: Any, **kwargs: Any
    ) -> None:
        """Records the start of a runnable execution in the trace log."""
        message = f"{' ' * self.indent_level}Starting: {runnable_name} with input: {input_data}"
        self.trace_log.add_log(message)
        self.indent_level += self.indent_step

    def on_invoke_end(
        self, runnable_name: str, output_data: Any, **kwargs: Any
    ) -> None:
        """Records the completion of a runnable execution in the trace log."""
        self.indent_level -= self.indent_step
        message = f"{' ' * self.indent_level}Finished: {runnable_name} with output: {output_data}"
        self.trace_log.add_log(message)


class ConsoleCallbackHandler(CallbackHandler):
    """Handles callbacks by printing them to the console."""

    def on_invoke_start(
        self, runnable_name: str, input_data: Any, **kwargs: Any
    ) -> None:
        """Prints the start of a runnable execution to the console."""
        print(f"Starting: {runnable_name} with input: {input_data}")

    def on_invoke_end(
        self, runnable_name: str, output_data: Any, **kwargs: Any
    ) -> None:
        """Prints the completion of a runnable execution to the console."""
        print(f"Finished: {runnable_name} with output: {output_data}")


class CallbackManager:
    """Manages multiple callback handlers and distributes callbacks to them."""

    def __init__(self, handlers: Optional[List[CallbackHandler]] = None) -> None:
        self.handlers = handlers or []

    def on_invoke_start(
        self, runnable_name: str, input_data: Any, **kwargs: Any
    ) -> None:
        """Notifies all handlers about the start of a runnable execution."""
        for handler in self.handlers:
            handler.on_invoke_start(runnable_name, input_data, **kwargs)

    def on_invoke_end(
        self, runnable_name: str, output_data: Any, **kwargs: Any
    ) -> None:
        """Notifies all handlers about the completion of a runnable execution."""
        for handler in self.handlers:
            handler.on_invoke_end(runnable_name, output_data, **kwargs)
