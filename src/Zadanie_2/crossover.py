import numpy as np
import random as r

def crossover(parent_1: np.array, parent_2: np.array):
    crossing_point = r.randint(1,398)
    child_1 = np.concatenate(parent_1[:crossing_point], parent_2[crossing_point:])
    child_2 = np.concatenate(parent_2[:crossing_point], parent_1[crossing_point:])

    return child_1, child_2