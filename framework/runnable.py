from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from framework import *
from .tracer import CallbackManager


class Runnable(ABC):
    """Base class for all runnable components that can be chained together."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def __or__(self, other: "Runnable") -> "Chain":
        return Chain(self, other)

    @abstractmethod
    def invoke(
        self, input_data: Any, callbacks: Optional[CallbackManager] = None, **kwargs
    ) -> Any:
        """Execute the runnable component with the given input."""
        pass


class RunnableLambda(Runnable):
    """Wraps a function to make it a runnable component."""

    def __init__(self, func: Callable, name: Optional[str] = None):
        super().__init__(name=name or func.__name__)
        self.func = func

    def invoke(
        self,
        input_data: Any,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped function with the given input."""
        if callbacks:
            callbacks.on_invoke_start(self.name, input_data, **kwargs)

        output = self.func(input_data)

        if callbacks:
            callbacks.on_invoke_end(self.name, output, **kwargs)
        return output


class RunnableParallel(Runnable):
    """Executes multiple runnables in parallel with the same input."""

    def __init__(self, runnables: Dict[str, Runnable], name: Optional[str] = None):
        super().__init__(name=name or "Parallel")
        self.runnables = runnables

    def invoke(
        self,
        input_data: Any,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute all runnables in parallel and return their outputs as a dictionary."""
        if callbacks:
            callbacks.on_invoke_start(self.name, input_data, **kwargs)

        output = {
            key: runnable.invoke(input_data, callbacks, **kwargs)
            for key, runnable in self.runnables.items()
        }

        if callbacks:
            callbacks.on_invoke_end(self.name, output, **kwargs)
        return output


class RunnablePassthrough(Runnable):
    """Passes through input data either completely or partially based on specified keys."""

    def __init__(
        self, passthrough_keys: Optional[List[str]] = None, name: Optional[str] = None
    ):
        super().__init__(name=name or "Passthrough")
        self.passthrough_keys = passthrough_keys

    def invoke(
        self,
        input_data: Any,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Any:
        """Pass through the input data based on the configured passthrough keys."""
        if callbacks:
            callbacks.on_invoke_start(self.name, input_data, **kwargs)

        if self.passthrough_keys is None:
            output = input_data
        else:
            if len(self.passthrough_keys) == 1:
                key = self.passthrough_keys[0]
                output = input_data.get(key)
            else:
                output = {
                    key: input_data[key]
                    for key in self.passthrough_keys
                    if key in input_data
                }

        if callbacks:
            callbacks.on_invoke_end(self.name, output, **kwargs)
        return output


class RunnableBranch(Runnable):
    """Executes different runnables based on conditional logic."""

    def __init__(
        self, *branches: Tuple[Callable, Callable], name: Optional[str] = None
    ):
        super().__init__(name=name or "Branch")
        self.branches = branches

    def invoke(
        self,
        input_data: Any,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the first runnable whose condition evaluates to True."""
        if callbacks:
            callbacks.on_invoke_start(self.name, input_data, **kwargs)

        for condition, func in self.branches:
            if condition(input_data):
                output = func(input_data)
                if callbacks:
                    callbacks.on_invoke_end(self.name, output, **kwargs)
                return output
        return None


class Chain(Runnable):
    """Chains two runnables together, passing output of first as input to second."""

    def __init__(self, runnable1: Runnable, runnable2: Runnable):
        super().__init__(name=f"{runnable1.name} | {runnable2.name}")
        self.runnable1 = runnable1
        self.runnable2 = runnable2

    def invoke(
        self,
        input_data: Any,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute both runnables in sequence."""
        if callbacks:
            callbacks.on_invoke_start(self.name, input_data, **kwargs)

        output1 = self.runnable1.invoke(input_data, callbacks=callbacks, **kwargs)
        output2 = self.runnable2.invoke(output1, callbacks=callbacks, **kwargs)

        if callbacks:
            callbacks.on_invoke_end(self.name, output2, **kwargs)
        return output2
