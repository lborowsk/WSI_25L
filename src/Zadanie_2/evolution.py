import numpy as np
from crossover import crossover
from mutation import mutate
from selection import roulette_selection
from evaluation import find_the_fittest

def evolution(population: np.ndarray, genome_size: int, max_iter:int , mutation_rate: float):
    iter = 0
    pop_size = population.shape[0]
    best_individual, best_fitness = find_the_fittest(population)
    while (iter < max_iter):
        parents = roulette_selection(population, pop_size)
        half = len(parents) // 2
        offspring = crossover(parents[:half], parents[half:], genome_size)
        mutate(offspring, mutation_rate)
        new_best_individual, new_best_fitness = find_the_fittest(offspring)
        if new_best_fitness > best_fitness:
            best_individual = new_best_individual
            best_fitness = new_best_fitness
        population = offspring
        iter += 1
    
    return best_individual, best_fitness

