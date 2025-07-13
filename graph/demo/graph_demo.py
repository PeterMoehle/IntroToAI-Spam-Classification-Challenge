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


class L1LossModel(BaseGraph):
    """Model using L1 norm (absolute error) loss."""

    def __init__(self):
        w_val = np.array([[2.0]])
        x_val = np.array([[1.0]])
        y_val = np.array([[2.0]])

        self.w = ValueNode(1, w_val, name_prefix="w", is_learnable=True)
        self.x = ValueNode(2, x_val, name_prefix="x", is_learnable=False)
        self.y = ValueNode(3, y_val, name_prefix="y", is_learnable=False)

        # Prediction: w^T x
        self.pred = FunctionNode(4, DotProduct(), [self.w, self.x])

        # Error: prediction - y
        self.error = FunctionNode(5, Subtraction(), [self.pred, self.y])

        # Absolute error
        self.abs_error = FunctionNode(6, Abs(), [self.error])

        # Sum of absolute errors (scalar loss)
        self.loss = FunctionNode(7, Sum(), [self.abs_error])

        super().__init__(self.pred, self.loss)

import numpy as np

from graph import BaseGraph
from graph.nodes import ValueNode, FunctionNode

class L2LossModel(BaseGraph):
    """Model using L2 norm (mean squared error) loss."""

    def __init__(self):
        w_val = np.array([[3.0]])
        x_val = np.array([[4.0]])
        y_val = np.array([[20.0]])

        self.w = ValueNode(1, w_val, name_prefix="w", is_learnable=True)
        self.x = ValueNode(2, x_val, name_prefix="x", is_learnable=False)
        self.y = ValueNode(3, y_val, name_prefix="y", is_learnable=False)

        # Prediction: w^T x
        self.pred = FunctionNode(4, DotProduct(), [self.w, self.x])

        # Error: prediction - y
        self.error = FunctionNode(5, Subtraction(), [self.pred, self.y])

        # Squared error = error * error
        self.sq_error = FunctionNode(6, ElementwiseMultiplication(), [self.error, self.error])

        # Sum of squared errors (scalar loss)
        self.loss = FunctionNode(7, Sum(), [self.sq_error])

        super().__init__(self.pred, self.loss)



if __name__ == '__main__':
    graph = L1LossModel()
    graph.solve_details(True)
    graph.print_digraph()
    graph.save_digraph("BackpropExercise", "BackpropExercise")