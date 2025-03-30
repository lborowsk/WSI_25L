import numpy as np
from evolution import evolution

population = np.random.randint(0, 2, size=(100, 400))
best_individual, best_fitness = evolution(population, 100, 10000, 0.01)
print(best_individual, best_fitness)