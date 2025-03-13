import numpy as np

# Gradient descent algorithm implementation
def gradient_descent(f_grad: callable, x: np.array, beta: float, tol: float, max_iter: int):
    iter_count = 0
    history = np.array([x])
    while iter_count < max_iter and np.linalg.norm(f_grad(x)) > tol:
        iter_count += 1
        gradient = f_grad(x)
        x = x - beta * gradient
        history = np.append(history, [x], axis=0)
    
    return x, history