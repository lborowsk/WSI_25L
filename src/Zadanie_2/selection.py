import numpy as np

def roulette_selection(population: np.ndarray, num_offspring: int, fitness:np.array) -> np.ndarray:
    if np.sum(fitness) == 0:
        probability = np.ones(len(population)) / len(population)
    else:
        probability = fitness/np.sum(fitness)
    parent_indices = np.random.choice(len(population), size=num_offspring, replace=True, p=probability)
    return population[parent_indices]