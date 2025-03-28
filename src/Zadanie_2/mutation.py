import numpy as np

def mutate(population: np.array, probability: float):
    mask = np.random.rand(*population.shape) < probability
    population[mask] = 1 - population[mask]