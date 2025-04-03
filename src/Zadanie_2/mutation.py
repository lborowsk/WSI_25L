import numpy as np

def mutate(population: np.ndarray, probability: float):
    for i in range(len(population)):
        mask = np.random.rand(len(population[i])) < probability
        population[i][mask] = 1 - population[i][mask]
    
    return population