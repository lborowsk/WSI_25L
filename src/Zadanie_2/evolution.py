import numpy as np
from crossover import crossover
from mutation import mutate
from selection import roulette_selection
from Zadanie_4.evaluation import find_the_fittest, evaluate

def evolution(population: np.ndarray, max_iter: int, 
              mutation_rate: float, crossover_prob: float, elite_size: int = 1):
    
    genome_size = len(population[0])
    iter = 0
    pop_size = population.shape[0]
    best_individual, best_fitness = find_the_fittest(population)
    
    while iter < max_iter:
        fitness = evaluate(population)[0]
        elite_indices = np.argsort(fitness)[-elite_size:]
        elite = population[elite_indices]
        
        parents = roulette_selection(population, pop_size - elite_size, fitness)
        
        offspring = crossover(parents, genome_size, crossover_prob)
        offspring = mutate(offspring, mutation_rate)

        new_population = np.concatenate([offspring, elite])
        
        new_best_individual, new_best_fitness = find_the_fittest(new_population)
        if new_best_fitness > best_fitness:
            best_individual = new_best_individual
            best_fitness = new_best_fitness
        
        population = new_population
        iter += 1
    
    return best_individual, best_fitness