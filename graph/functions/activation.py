import numpy as np

from typing import List

from .arithmetic import Function
from ..abc_node import Node


class ReLU(Function):
    """Rectified Linear Unit activation function."""

    def __init__(self):
        super().__init__("ReLU")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        return np.maximum(0, nodes[0].value)

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        return (nodes[0].value > 0).astype(float)


class Sigmoid(Function):
    """Sigmoid activation function."""

    def __init__(self):
        super().__init__("Sigmoid")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        clipped_input = np.clip(nodes[0].value, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped_input))

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        s = self.compute(nodes)
        return s * (1 - s)


class Tanh(Function):
    """Hyperbolic tangent activation function."""

    def __init__(self):
        super().__init__("Tanh")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        return np.tanh(nodes[0].value)

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        tanh_val = np.tanh(nodes[0].value)
        return 1 - tanh_val ** 2