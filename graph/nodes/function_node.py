import numpy as np

from typing import List, Union, Tuple

from ..abc_node import Node
from ..functions.arithmetic import Function


class FunctionNode(Node):
    """
    Node that applies a mathematical function to input nodes.

    FunctionNodes represent mathematical operations in the computational graph
    and handle both forward computation and backward gradient propagation.
    """

    def __init__(self, idx: int, function: Function, input_nodes: List[Node]):
        """Initialize a function node."""
        super().__init__(idx)
        self.function = function
        self.input_nodes = input_nodes
        self.name = f"{function.name}_f{self.id}"
        self.node_type = "FunctionNode"

    def forward(self) -> np.ndarray:
        """Compute the function output and store it."""
        self.value = self.function.compute(self.input_nodes)
        return self.value

    def backward(self, upstream_gradient: np.ndarray) -> None:
        """Propagate gradients to input nodes using the chain rule."""
        super().backward(upstream_gradient)

        for node in self.input_nodes:
            local_gradient_info = self.function.derivate(node, self.input_nodes)
            grad_to_propagate = self._compute_chain_rule_gradient(upstream_gradient, local_gradient_info)
            node.backward(grad_to_propagate)

    def _compute_chain_rule_gradient(self, upstream_gradient: np.ndarray,local_gradient_info: Union[np.ndarray, Tuple[str, np.ndarray]]) -> np.ndarray:
        """Compute the gradient using the chain rule."""
        if isinstance(local_gradient_info, tuple):
            op_type, local_gradient = local_gradient_info
            if op_type == "pre-multiply":
                return local_gradient @ upstream_gradient
        else:
            local_gradient = local_gradient_info

        if upstream_gradient.size == 1 or local_gradient.size == 1:
            return upstream_gradient * local_gradient

        if len(upstream_gradient.shape) >= 2 and len(local_gradient.shape) >= 2:
            try:
                return upstream_gradient @ local_gradient
            except ValueError:
                pass

        # Handle broadcasting for biases
        if upstream_gradient.shape != local_gradient.shape:
            if np.prod(upstream_gradient.shape) == np.prod(local_gradient.shape):
                return upstream_gradient.reshape(local_gradient.shape) * local_gradient
            if len(local_gradient.shape) > 1 and local_gradient.shape[0] == 1:
                return np.sum(upstream_gradient, axis=0, keepdims=True) * local_gradient

        return upstream_gradient * local_gradient