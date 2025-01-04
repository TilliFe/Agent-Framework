from typing import Callable, Dict, Optional, Any
from .runnable import Runnable, CallbackManager


class Node(Runnable):
    def __init__(self, name: str, action: Callable[[Dict], Dict]):
        """Initialize a Node with a name and an action."""
        self.name = name
        self.action = action
        self.next: Optional[Node] = None
        self.transitions: Dict[str, Node] = {}

    def __eq__(self, other: object) -> bool:
        """Compare two nodes for equality based on their names."""
        if isinstance(other, Node):
            return self.name == other.name
        return False

    def invoke(
        self, state: Dict, callbacks: Optional[CallbackManager] = None, **kwargs: Any
    ) -> Dict:
        """Execute the node's action and return the updated state."""
        if callbacks:
            callbacks.on_invoke_start(self.name, state, **kwargs)
        result = self.action(state)
        if callbacks:
            callbacks.on_invoke_end(self.name, result, **kwargs)
        return result

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"{self.name}"
