import numpy as np
def shift_r(x):
    return np.concatenate((np.zeros_like(x[..., :, -1:]), x[..., :, :-1]), axis=-1)

def shift_l(x):
    return np.concatenate((x[..., :, 1:], np.zeros_like(x[..., :, :1])), axis=-1)

def shift_t(x):
    return np.concatenate((np.zeros_like(x[..., -1:, :]), x[..., :-1, :]), axis=-2)

def shift_b(x):
    return np.concatenate((x[..., 1:, :], np.zeros_like(x[..., :1, :])), axis=-2)

def evaluate(x, size=20):
    assert np.shape(x)[-1] == size * size
    grid = np.asarray(np.reshape(x, (-1, size, size)), dtype=np.int_)
    points = np.minimum(
        shift_r(grid) + shift_l(grid) + shift_t(grid) + shift_b(grid),
        1 - grid
    )
    return points.reshape(np.shape(x)).sum(-1), grid, points

def find_the_fittest(population: np.ndarray):
    fitness = evaluate(population)[0]
    best_index = np.argmax(fitness)
    return population[best_index], fitness[best_index]
