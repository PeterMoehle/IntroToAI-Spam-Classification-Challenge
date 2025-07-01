import numpy as np

from typing import List

from .arithmetic import Function
from ..abc_node import Node


class Exp(Function):
    """Exponential function."""

    def __init__(self):
        super().__init__("Exp")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        return np.exp(np.clip(nodes[0].value, -500, 500))

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        return np.exp(np.clip(nodes[0].value, -500, 500))


class Log(Function):
    """Natural logarithm function."""

    def __init__(self):
        super().__init__("Log")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        return np.log(np.maximum(nodes[0].value, 1e-9))

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        return 1.0 / np.maximum(nodes[0].value, 1e-9)