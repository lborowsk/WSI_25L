import numpy as np
import random as r

def crossover(parents_a: np.ndarray, parents_b: np.ndarray, genome_size: int) -> np.ndarray:
    
    assert len(parents_a) == len(parents_b)
    
    children = []
    for parent_1, parent_2 in zip(parents_a, parents_b):
        crossing_point = np.random.randint(1, genome_size - 1)
        child_1 = np.concatenate([parent_1[:crossing_point], parent_2[crossing_point:]])
        child_2 = np.concatenate([parent_2[:crossing_point], parent_1[crossing_point:]])
        children.extend([child_1, child_2])
    
    return np.array(children)