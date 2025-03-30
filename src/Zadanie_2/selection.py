import numpy as np
from evaluation import evaluate
from crossover import crossover
from mutation import mutate

def roulette_selection(population: np.ndarray, num_offspring: int) -> np.ndarray:
    fitness = evaluate(population)
    probability = fitness/np.sum(fitness)
    parent_indices = np.random.choice(len(population), size=num_offspring, p=probability)
    return population[parent_indices]