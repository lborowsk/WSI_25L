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
    grid = np.cast[np.int_](np.reshape(x, (-1, size, size)))
    points = np.minimum(
    shift_r(grid) + shift_l(grid) + shift_t(grid) + shift_b(grid),
    1 - grid
    )
    return points.reshape(np.shape(x)).sum(-1)
