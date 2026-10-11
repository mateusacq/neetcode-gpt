class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        if iterations == 0:
            return init

        x_old = init
        x_new = 0
        
        for i in range(iterations):
            x_new = x_old - learning_rate * 2 * x_old
            x_old = x_new

        return round(x_new, 5)
