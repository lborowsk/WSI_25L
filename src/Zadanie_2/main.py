import numpy as np
from evolution import evolution
from visualisation import visualize_board
from evaluation import find_the_fittest

population = np.random.randint(0, 2, size=(100, 400))
'''
random_fittest, random_fitness = find_the_fittest(population)
print(random_fitness)
visualize_board(random_fittest)
'''
best_individual, best_fitness = evolution(population, 100, 30000, 0.01)
print(best_individual, best_fitness)
visualize_board(best_individual)