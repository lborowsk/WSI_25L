import numpy as np

# Gradient of function f
def gradient_f(vector: np.array):
    return np.array([40*vector[0]**3 + 9*vector[0]**2 - 60*vector[0] + 10])

# Gradient of function g
def gradient_g(vector: np.array):
    return np.array([(4*(vector[0]-2)**3 + 4*(vector[0]-2)*(vector[1]+3)**2),
                     (4*(vector[1]+3)**3 + 4*(vector[1]+3)*(vector[0]-2)**2)])