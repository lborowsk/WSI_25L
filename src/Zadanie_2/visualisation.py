import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from evaluation import evaluate
from matplotlib.patches import Rectangle

def visualize_board(board, size=20):
    if board.ndim == 1:
        board = board.reshape(size, size)
    elif board.shape != (size, size):
        raise ValueError(f"Tablica musi mieć kształt ({size},{size}) lub {size*size} elementów")
    
    padded = np.pad(board, 1, mode='constant')
    neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                padded[1:-1, :-2] + padded[1:-1, 2:])
    points = (neighbors > 0) & (board == 0)
    
    cmap = mcolors.ListedColormap(['yellow', 'lightgreen', 'darkgreen'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    display_array = np.zeros_like(board)
    display_array[points] = 1
    display_array[board == 1] = 2
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(display_array, cmap=cmap, norm=norm)

    for i in range(size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels(np.arange(1, size + 1))
    ax.set_yticklabels(np.arange(1, size + 1))
    ax.tick_params(axis='both', which='both', length=0)
    
    # Legenda
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='darkgreen', label='Obiekt (1)'),
        Rectangle((0, 0), 1, 1, color='lightgreen', label='Punkt (sąsiad z obiektem)'),
        Rectangle((0, 0), 1, 1, color='yellow', label='Brak punktu')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Wizualizacja rozwiązania")
    plt.tight_layout()
    plt.show()



def generate_trivial_solution(size=20):

    board = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        if i % 4 == 0:  # Co trzeci wiersz zaczynając od 0: 101010...
            board[i, ::2] = 1
        elif i % 4 == 2:  # Co trzeci wiersz z przesunięciem: 010101...
            board[i, 1::2] = 1
        # Pozostałe wiersze pozostają zerami
    
    return board
