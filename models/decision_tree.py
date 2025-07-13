import numpy as np

from typing import Dict, Any, List
from scipy.stats import entropy, mode

from models import Model

def H(x):
    _, counts = np.unique(x, return_counts=True)
    return entropy(counts)


class Node:
    max_depth = 0

    def __init__(self, current_depth: int, X: np.ndarray, y: np.ndarray):
        self.current_depth = current_depth
        self.X = X
        self.y = y

    @classmethod
    def set_max_depth(cls, max_depth: int):
        cls.max_depth = max(1, max_depth)


class Score:
    def __init__(self, value: float, threshold: float, feature_idx: int):
        self.value = value
        self.threshold = threshold
        self.feature_idx = feature_idx


class LeafNode(Node):
    def __init__(self, current_depth: int, X: np.ndarray, y: np.ndarray, label: int):
        super().__init__(current_depth, X, y)
        self.label = label
        self.scores: List[Score] = []

        if current_depth < Node.max_depth:
            self._add_decision()

    def _define_scores(self):
        n_features = self.X.shape[1]
        for j in range(n_features):
            thresholds = np.unique(self.X[:, j])
            for threshold in thresholds:
                left_idx = np.where(self.X[:, j] < threshold)[0]
                right_idx = np.where(self.X[:, j] >= threshold)[0]

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                score_val = H(self.y) - len(left_idx) / len(self.y) * H(self.y[left_idx]) - len(right_idx) / len(self.y) * H(self.y[right_idx])

                if score_val > 0:
                    self.scores.append(Score(score_val, threshold, j))

    def _get_best_score(self) -> Score:
        return max(self.scores, key=lambda s: s.value)

    def _add_decision(self):
        self._define_scores()
        if not self.scores:
            return

        best_score = self._get_best_score()
        self.__class__ = DecisionNode
        self.__init__(self.current_depth, self.X, self.y, best_score.threshold, best_score.feature_idx)


class DecisionNode(Node):
    def __init__(self, current_depth: int, X: np.ndarray, y: np.ndarray, threshold: float, feature_idx: int):
        super().__init__(current_depth, X, y)
        self.threshold = threshold
        self.feature_idx = feature_idx

        left_idx = np.where(X[:, feature_idx] < threshold)[0]
        right_idx = np.where(X[:, feature_idx] >= threshold)[0]

        left_label = int(mode(y[left_idx], keepdims=True).mode[0]) if len(left_idx) > 0 else 0
        right_label = int(mode(y[right_idx], keepdims=True).mode[0]) if len(right_idx) > 0 else 0

        self.left_child = LeafNode(current_depth + 1, X[left_idx], y[left_idx], left_label)
        self.right_child = LeafNode(current_depth + 1, X[right_idx], y[right_idx], right_label)


class DecisionTreeClassifier(Model):
    def __init__(self, max_depth: int = 2):
        super().__init__(name="DecisionTreeClassifier")
        self.max_depth = max_depth
        self.root: Node = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        Node.set_max_depth(self.max_depth)
        y_train = y_train.ravel().astype(int)
        most_frequent_label = np.bincount(y_train).argmax()
        self.root = LeafNode(1, X_train, y_train, int(most_frequent_label))
        return {"status": "trained", "depth": self.max_depth}

    def predict(self, X: np.ndarray) -> np.ndarray:
        def traverse_tree(node: Node, x: np.ndarray) -> int:
            if isinstance(node, LeafNode):
                return node.label
            elif isinstance(node, DecisionNode):
                if x[node.feature_idx] < node.threshold:
                    return traverse_tree(node.left_child, x)
                else:
                    return traverse_tree(node.right_child, x)
            else:
                raise TypeError("Unknown node type")

        return np.array([traverse_tree(self.root, x) for x in X])

    def get_binary_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict(X)



