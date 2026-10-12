import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # Considering: L = hidden_size (number of layers),
        #              I = input_size, n = output_size
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        n = y_true.shape[0]

        # Forward
        z1 = W1 @ x + b1 # L x 1
        a1 = np.maximum(0, z1) # L x 1
        z2 = W2 @ a1 + b2 # y_hat (n x 1)

        # Loss
        mse = np.mean(np.square(z2 - y_true))

        # Backward
        dL_dz2 = (2 * (z2 - y_true)) / n # n x 1
        # dz2_dW2 = a1 (L x 1)
        # dz2_da1 = W2 (n x L)
        da1_dz1 = (z1 > 0).astype(float)
        # dz1_dW1 = x (I x 1)

        # Final gradients
        dL_db2 = dL_dz2 # n x 1
        dL_dW2 = np.outer(dL_db2, a1.T) # n x L
        dL_db1 = (W2.T @ dL_dz2) * da1_dz1 # L x 1 (Hadamard product (*) with 2 numpy L x 1 arrays)
        dL_dW1 = np.outer(dL_db1, x.T) # L x I

        # Roundings
        return {
            'loss': float(np.round(mse, 4)),
            'dW1':  np.round(dL_dW1, 4).tolist(),
            'db1':  np.round(dL_db1, 4).tolist(),
            'dW2':  np.round(dL_dW2, 4).tolist(),
            'db2':  np.round(dL_db2, 4).tolist()
        }