import numpy as np

from graph import BaseGraph
from graph.functions import DotProduct, ElementwiseMultiplication, Exp, Addition, Log, Sum
from graph.functions.arithmetic import Subtraction, Abs
from graph.nodes import ValueNode, FunctionNode



class BackpropExercise(BaseGraph):
    """A graph for demonstrating computation with automatic differentiation."""

    def __init__(self):
        w_val = np.array([[2.0]])
        x_val = np.array([[1.0]])
        y_val = np.array([[2.0]])

        self.w = ValueNode(1, w_val, name_prefix="w", is_learnable=True)
        self.x = ValueNode(2, x_val, name_prefix="x", is_learnable=False)
        self.y = ValueNode(3, y_val, name_prefix="y", is_learnable=False)

        # Compute w^T x (dot product)
        self.dot = FunctionNode(4, DotProduct(), [self.w, self.x])

        # Compute y * (w^T x)
        self.y_dot = FunctionNode(5, ElementwiseMultiplication(), [self.y, self.dot])

        # Compute exp(y * w^T x)
        self.exp_term = FunctionNode(6, Exp(), [self.y_dot])

        # Compute 1 + exp(y * w^T x)
        one = ValueNode(7, np.array([[1.0]]), name_prefix="one", is_learnable=False)
        self.sum = FunctionNode(8, Addition(), [one, self.exp_term])

        # Compute log(1 + exp(y * w^T x))
        self.log_result = FunctionNode(9, Log(), [self.sum])

        # This is the loss node
        loss_node = FunctionNode(10, Sum(), [self.log_result])

        super().__init__(self.dot, loss_node)



if __name__ == '__main__':
    graph = BackpropExercise()
    graph.solve_details(True)
    graph.print_digraph()
    graph.save_digraph("BackpropExercise", "BackpropExercise")