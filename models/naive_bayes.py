import numpy as np

from typing import Dict, Any

from .abc_model import Model


class NaiveBayes(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.pNoSpam = None
        self.pSpam = None
        self.pFeatureGivenNoSpam = None
        self.pFeatureGivenSpam = None
        self.noSpam = None
        self.spam = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        XTrain = X_train.copy()
        yTrain = y_train.copy()

        # Compute marginal and conditional probabilities -- case noSpam
        self.noSpam = XTrain[yTrain[:, 0] == 0]  # noSpam emails
        self.pNoSpam = len(self.noSpam) / len(XTrain)  # marginal probability noSpam
        # Count how many non-spam mails contain each feature
        count0 = np.count_nonzero(self.noSpam, 0)

        # Laplace smoothing: (count + 1) / (total + 2)
        self.pFeatureGivenNoSpam = np.vstack((len(self.noSpam) - count0 + 1, count0 + 1)) / (
                len(self.noSpam) + 2)

        # Compute marginal and conditional probabilities -- case spam
        self.spam = XTrain[yTrain[:, 0] == 1]  # spam emails
        self.pSpam = len(self.spam) / len(XTrain)  # marginal probability spam
        count1 = np.count_nonzero(self.spam, 0)

        # Laplace smoothing: (count + 1) / (total + 2)
        # Should in theory be the same as adding a row of '0' and a row of '1' into X
        self.pFeatureGivenSpam = np.vstack((len(self.spam) - count1 + 1, count1 + 1)) / (
                len(self.spam) + 2)

        if verbose:
            print(f"Trained Naive Bayes: {len(self.spam)} spam, {len(self.noSpam)} no-spam emails")

        return {
            "spam_emails": len(self.spam),
            "nospam_emails": len(self.noSpam),
            "total_features": X_train.shape[1]
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Compute log-likelihood scores
        # A fancy function to get the density of X, given our calculated distribution
        scoreSpam = (np.sum(np.log(np.take_along_axis(self.pFeatureGivenSpam[:, None, :], X[None], axis=0)[0]),
                            1) + np.log(self.pSpam))[:, None]
        scoreNoSpam = (np.sum(np.log(np.take_along_axis(self.pFeatureGivenNoSpam[:, None, :], X[None], axis=0)[0]),
                              1) + np.log(self.pNoSpam))[:, None]

        return (scoreSpam > scoreNoSpam)

    def get_binary_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict(X).astype(int)