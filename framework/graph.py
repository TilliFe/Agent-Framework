from matplotlib import pyplot as plt
from framework import *
from typing import Callable, Dict, List, Optional, Any, Union
from .node import Node
from .runnable import Runnable, CallbackManager


class Graph(Runnable):
    def __init__(
        self,
        state_schema: Dict,
        update_mechanisms: Optional[Dict[str, Callable[[Dict, Any, Any], Dict]]] = None,
        name="Graph",
    ):
        """Initialize a new Graph with state schema and optional update mechanisms."""
        self.name = name
        self.nodes: List[Node] = []
        self.edges: List[tuple[str, str]] = []
        self.state: Dict = state_schema
        self.update_mechanisms: Dict[str, Callable[[Dict, Any, Any], Dict]] = (
            update_mechanisms or {}
        )
        self.start_node: Optional[Node] = None
        self.end_nodes: Optional[List[Node]] = None

    def add_node(self, node: Node) -> None:
        """Add a new node to the graph."""
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, edge: tuple[Node, Node]) -> None:
        """Add a directed edge between two nodes in the graph."""
        if edge[0] not in self.nodes or edge[1] not in self.nodes:
            raise ValueError("Node not in graph." + edge[0].name + edge[1].name)
        if edge[0].next is None:
            edge[0].next = edge[1]
            self.edges.append((edge[0].name, edge[1].name))
        else:
            raise ValueError("Node already has a next node.")

    def add_conditional_edges(
        self,
        source_node: Node,
        transition_resolver: Callable[[Dict], str],
        target_nodes: Dict[str, Node],
    ) -> None:
        """Add conditional edges from a source node to multiple target nodes."""
        if source_node not in self.nodes:
            raise ValueError("Source node not in graph." + source_node.name)

        conditional_node = Node(f"{source_node.name}_conditional", transition_resolver)
        self.add_node(conditional_node)
        self.add_edge((source_node, conditional_node))
        conditional_node.transitions = target_nodes

        self.edges.append((source_node.name, f"{source_node.name}_conditional"))
        for key, value in target_nodes.items():
            self.edges.append((f"{source_node.name}_conditional", value.name))

    def update_state(self, key: Any, value: Any) -> None:
        """Update the graph's state with a new key-value pair."""
        if key in self.update_mechanisms:
            self.update_mechanisms[key](self.state, key, value)
        else:
            self.state[key] = value

    def set_update_mechanism(
        self, key: Any, mechanism: Callable[[Dict, Any, Any], Dict]
    ) -> None:
        """Set a custom update mechanism for a specific state key."""
        self.update_mechanisms[key] = mechanism

    def set_start_node(self, node: Node) -> None:
        """Define the starting node of the graph."""
        if node not in self.nodes:
            raise ValueError("Node not in graph: " + node.name)
        self.start_node = node

    def set_end_nodes(self, nodes: Union[Node, List[Node]]) -> None:
        """Define the terminal nodes of the graph."""
        if isinstance(nodes, Node):
            nodes = [nodes]

        for node in nodes:
            if node not in self.nodes:
                raise ValueError("Node not in graph: " + node.name)
        self.end_nodes = nodes

    def invoke_next_node(
        self,
        current_node: Node,
        depth: int = 0,
        max_depth: Optional[int] = None,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Execute the next node in the graph's execution sequence."""
        if current_node in self.end_nodes or (
            max_depth is not None and depth >= max_depth
        ):
            return

        if current_node.name.endswith("_conditional"):
            next_node_name = current_node.invoke(
                self.state, callbacks=callbacks, **kwargs
            )
            if next_node_name in current_node.transitions:
                current_node = current_node.transitions[next_node_name]
            else:
                raise ValueError(f"Invalid transition: {next_node_name}")

        state_updates = current_node.invoke(
            state=self.state, callbacks=callbacks, **kwargs
        )

        if state_updates is not None:
            for key, value in state_updates.items():
                self.update_state(key, value)

        if current_node.next is not None:
            self.invoke_next_node(
                current_node.next,
                depth=depth + 1,
                max_depth=max_depth,
                callbacks=callbacks,
                **kwargs,
            )

    def invoke(
        self,
        state: Dict,
        max_depth: Optional[int] = None,
        callbacks: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Execute the entire graph starting from the start node."""
        if callbacks:
            callbacks.on_invoke_start(self.name, state, **kwargs)
        self.state = state
        if self.start_node is None:
            raise ValueError("Start node not set.")
        if self.end_nodes is None:
            raise ValueError("End node not set.")

        self.invoke_next_node(
            self.start_node, depth=0, max_depth=max_depth, callbacks=callbacks, **kwargs
        )

        if callbacks:
            callbacks.on_invoke_end(self.name, self.state, **kwargs)

        return self.state

    def __repr__(self) -> str:
        """Return a string representation of the Graph."""
        return f"Graph(nodes={self.nodes}, edges={self.edges}, state={self.state})"

    def compile(self):
        """Return a compiled version of the Graph as a Runnable."""
        if not self.start_node:
            raise ValueError("Start node not set.")
        if not self.end_nodes:
            raise ValueError("End node not set.")
        if not self.nodes:
            raise ValueError("No nodes in the graph.")

        return self

    def plot(self) -> None:
        """Plot the graph using networkx with a spring layout."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Please install networkx to use this method.")

        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        pos = nx.spring_layout(G, k=1, iterations=50)

        plt.figure(figsize=(12, 8))

        rect_width = 0.8
        rect_height = 0.2

        for node, (x, y) in pos.items():
            color = "lightblue" if "_conditional" in node else "lightgreen"
            plt.gca().add_patch(
                plt.Rectangle(
                    (x - rect_width / 2, y - rect_height / 2),
                    rect_width,
                    rect_height,
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.7,
                )
            )
            plt.text(x, y, node, fontsize=8, ha="center", va="center")

        # Draw edges with arrows
        for edge in G.edges():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]

            # Calculate direction vector
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = (dx**2 + dy**2) ** 0.5

            # Normalize direction vector
            dx, dy = dx / length, dy / length

            # Adjust start and end points to rectangle borders
            start_x = start_pos[0] + dx * rect_width / 2
            start_y = start_pos[1] + dy * rect_height / 2
            end_x = end_pos[0] - dx * rect_width / 2
            end_y = end_pos[1] - dy * rect_height / 2

            if (edge[1], edge[0]) in G.edges():  # bidirectional edge
                plt.gca().annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0.2", color="gray"
                    ),
                )
            else:  # one-way edge
                plt.gca().annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0.1", color="gray"
                    ),
                )

        plt.margins(0.3)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
