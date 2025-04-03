import numpy as np
import random as r

def crossover(parents: np.ndarray, genome_size: int, crossover_prob: float) -> np.ndarray:
    
    children = []
    for i in range(0, len(parents)-1, 2):
        if np.random.rand() < crossover_prob:
            cross_point = np.random.randint(2, genome_size)
            children.append(np.concatenate((parents[i][:cross_point], parents[i + 1][cross_point:])))
            children.append(np.concatenate((parents[i + 1][:cross_point], parents[i][cross_point:])))
        else:
            children.append(parents[i])
            children.append(parents[i+1])
    
    return np.array(children)