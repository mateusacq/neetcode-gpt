import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h_current = x
        h_next = None
        n = len(weights)
        for idx, w in enumerate(weights):
            h_next = h_current @ w + biases[idx]
            if idx != n - 1:
                h_next = np.maximum(0, h_next)
            h_current = h_next

        return np.round(h_next, 5)