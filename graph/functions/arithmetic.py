import numpy as np

from abc import ABC, abstractmethod
from typing import List, Union, Tuple

from ..abc_node import Node


class Function(ABC):
    """
    Abstract base class for all mathematical functions.

    Functions represent mathematical operations that can be applied to nodes
    in the computational graph framework.
    """

    def __init__(self, name: str):
        """Initialize a function with a name."""
        self.name = name

    @abstractmethod
    def compute(self, nodes: List[Node]) -> np.ndarray:
        """Compute the function output given input nodes."""
        raise NotImplementedError("Subclasses must implement compute method")

    @abstractmethod
    def derivate(self, node: Node, nodes: List[Node]) -> Union[np.ndarray, Tuple[str, np.ndarray]]:
        """Compute the derivative with respect to the given node."""
        raise NotImplementedError("Subclasses must implement derivate method")


class Addition(Function):
    """Element-wise addition function."""

    def __init__(self):
        super().__init__("Addition")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        result = nodes[0].value.copy()
        for node in nodes[1:]:
            result = result + node.value
        return result

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        return np.ones_like(node.value)


class Subtraction(Function):
    """Element-wise subtraction function."""

    def __init__(self):
        super().__init__("Subtraction")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        result = nodes[0].value.copy()
        for node in nodes[1:]:
            result = result - node.value
        return result

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        if node is nodes[0]:
            return np.ones_like(node.value)
        else:
            return -np.ones_like(node.value)


class ElementwiseMultiplication(Function):
    """Element-wise multiplication function."""

    def __init__(self):
        super().__init__("ElementwiseMultiplication")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        if not nodes:
            return np.array(1.0)
        result = nodes[0].value.copy()
        for node in nodes[1:]:
            result = result * node.value
        return result

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        product = np.ones_like(node.value)
        for n in nodes:
            if n != node:
                product = product * n.value
        return product


class DotProduct(Function):
    """Matrix multiplication function."""

    def __init__(self):
        super().__init__("DotProduct")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        assert len(nodes) == 2, "DotProduct expects exactly two input nodes"
        return nodes[0].value @ nodes[1].value

    def derivate(self, node: Node, nodes: List[Node]) -> Union[np.ndarray, Tuple[str, np.ndarray]]:
        A, B = nodes
        if node == A:
            return B.value.T
        elif node == B:
            return ("pre-multiply", A.value.T)
        else:
            raise ValueError("Node not found in input list.")


class Sum(Function):
    """Sum reduction function."""

    def __init__(self):
        super().__init__("Sum")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        return np.array(np.sum(nodes[0].value))

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        return np.ones_like(nodes[0].value)


class Max(Function):
    """Element-wise maximum function."""

    def __init__(self):
        super().__init__("Max")

    def compute(self, nodes: List[Node]) -> np.ndarray:
        assert len(nodes) == 2, "Max expects exactly two input nodes"
        return np.maximum(nodes[0].value, nodes[1].value)

    def derivate(self, node: Node, nodes: List[Node]) -> np.ndarray:
        A, B = nodes
        if node == A:
            return (nodes[0].value >= nodes[1].value).astype(float)
        elif node == B:
            return (nodes[1].value > nodes[0].value).astype(float)
        else:
            raise ValueError("Node not found in input list.")