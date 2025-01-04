from typing import Dict, Optional, Any
from .runnable import Runnable, CallbackManager


class LLM(Runnable):
    """Represents a Language Learning Model (LLM) node that can generate content."""

    def __init__(self, model, generate_content_method, name="LLM"):
        self.name = name
        self.model = model
        self.generate_content_method = generate_content_method

    def invoke(
        self, query, callbacks: Optional[CallbackManager] = None, **kwargs: Any
    ) -> Dict:
        """Execute the LLM to generate content based on the input query."""
        if callbacks:
            callbacks.on_invoke_start(self.name, query, **kwargs)
        result = self.generate_content_method(self.model, query)
        if callbacks:
            callbacks.on_invoke_end(self.name, result, **kwargs)
        return result
