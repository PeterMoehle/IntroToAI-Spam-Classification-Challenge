import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .abc_node import Node
from .digraph_generator import DigraphGenerator
from .nodes import FunctionNode
from .nodes.value_node import ValueNode


class BaseGraph(ABC):
    """
    Abstract base class for all computational graphs.

    Provides the fundamental structure for building computational graphs
    with forward and backward pass capabilities.
    """

    def __init__(self, output_node: Node, loss_node: Node):
        """Initialize the computational graph."""
        self.output_node = output_node
        self.loss_node = loss_node
        self.all_nodes = self._collect_nodes(self.loss_node)
        self.forward_pass_order = self.all_nodes
        self.backward_pass_order = list(reversed(self.forward_pass_order))
        self.predict_pass_order = self._collect_nodes(self.output_node)
        self.learnable_params = [n for n in self.all_nodes if isinstance(n, ValueNode) and n.is_learnable]
        self.digraph_generator = DigraphGenerator()

    def reset_gradients(self):
        """Reset gradients for all nodes in the graph."""
        for node in self.all_nodes:
            node.reset_gradient()

    def solve(self, print_details: bool = False) -> Dict[str, Any]:
        """Execute forward and backward pass."""
        # Forward pass
        for node in self.forward_pass_order:
            node.forward()

        # Backward pass
        self.loss_node.backward(np.ones_like(self.loss_node.value))

        return {"loss": self.loss_node.value}

    def _collect_nodes(self, node: Node, visited=None) -> List[Node]:
        """Collect all nodes in topological order."""
        if visited is None:
            visited = set()
        if node in visited:
            return []
        visited.add(node)

        nodes = []
        # Import here to avoid circular imports
        from .nodes.function_node import FunctionNode
        if isinstance(node, FunctionNode):
            for input_node in node.input_nodes:
                nodes.extend(self._collect_nodes(input_node, visited))
        nodes.append(node)
        return nodes

    def solve_details(self, print_details: bool = False) -> Dict[str, Any]:
        """Execute the solve process and return results."""
        if print_details: print("--- Forward Pass ---")
        for node in self.forward_pass_order:
            node.forward()
            # Use digraph generator for formatting
            value_summary = self.digraph_generator._format_value_summary(node)
            if print_details: print(f"Computed {node.name}: {value_summary}")

        if print_details: print("\n--- Backward Pass ---")
        self.loss_node.backward(np.ones_like(self.loss_node.value))

        for node in self.backward_pass_order:
            if isinstance(node, FunctionNode):  # and node.gradient is not None:
                # Use digraph generator for formatting
                grad_summary = self.digraph_generator._format_gradient_summary(node)
                if print_details: print(f"Gradient at {node.name}: {grad_summary}")

        return {
            "loss": self.loss_node.value
        }

    def generate_digraph(self, title: Optional[str] = None) -> str:
        """Generate DOT format digraph using the digraph generator."""
        return self.digraph_generator.generate_digraph(self.all_nodes, title)

    def save_digraph(self, filename: str, title: Optional[str] = None) -> None:
        """Save the digraph to a file using the digraph generator."""
        self.digraph_generator.save_digraph(self.all_nodes, filename, title)

    def print_digraph(self, title: Optional[str] = None) -> None:
        """Print the digraph to console using the digraph generator."""
        self.digraph_generator.print_digraph(self.all_nodes, title)


class GraphModel(BaseGraph):
    """
    Main model class for training neural networks using computational graphs.

    Extends BaseGraph with training capabilities including parameter updates
    and training history tracking.
    """

    def __init__(self, input_nodes: List[ValueNode], target_node: ValueNode, output_node: Node, loss_node: Node):
        """Initialize the graph model."""
        super().__init__(output_node, loss_node)
        self.training_history = {"loss": [], "epoch": []}
        self.input_nodes = input_nodes
        self.target_node = target_node

    def _update_parameters(self, lr: float):
        """Update model parameters using gradients with clipping."""
        for param in self.learnable_params:
            if param.gradient is not None:
                # Gradient clipping for numerical stability
                # Otherwise they could eventually explode
                clipped_grad = np.clip(param.gradient, -1.0, 1.0)
                param.value -= lr * clipped_grad

    def _train_step(self, lr: float) -> float:
        """Perform one training step."""
        self.reset_gradients()
        result = self.solve()
        self._update_parameters(lr)
        return result["loss"].item()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model using the provided data."""
        self.input_nodes[0].value = X_train
        self.target_node.value = y_train

        for epoch in range(epochs):
            loss = self._train_step(lr)
            avg_loss = loss / X_train.shape[0]
            self.training_history["loss"].append(avg_loss)
            self.training_history["epoch"].append(epoch)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return self.training_history

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        self.input_nodes[0].value = input_data
        for node in self.predict_pass_order:
            node.forward()
        return self.output_node.value