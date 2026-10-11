import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)       
        y_pred = np.clip(y_pred, a_min = 1e-7, a_max = 1 - 1e-7)
        n = y_true.shape[0]

        L = np.dot(y_true, np.log(y_pred)) + np.dot((1 - y_true), np.log(1 - y_pred))
        L = (-1.0 * L) / n
        return np.round(L, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred, a_min = 1e-7, a_max = 1 - 1e-7)
        n = y_true.shape[0]

        L = []        
        for idx, pred in enumerate(y_pred):
            L.append(np.dot(np.log(pred), y_true[idx]))
        
        L = ( (-1) * np.sum(np.asarray(L)) ) / n
        return np.round(L, 4)
