from two_player_games.game import Game
from two_player_games.state import State
from two_player_games.player import Player
from alphabeta import alphabeta
from evaluation import evaluate
import random

def random_best_move(lst: list[int], maximizing: bool):
    if maximizing:
        max_val = max(lst)
        max_indices = [i for i, x in enumerate(lst) if x == max_val]
        return random.choice(max_indices)
    else:
        min_val = min(lst)
        min_indices = [i for i, x in enumerate(lst) if x == min_val]
        return random.choice(min_indices)


def make_best_move(game_state: State, depth: int, maximizingPlayer: Player):
    moves = []
    values = []
    for move in game_state.get_moves():
        moves.append(move)
        values.append(alphabeta(game_state.make_move(move), depth-1, float('-inf'), float('inf'), maximizingPlayer, evaluate))
    
    if game_state._current_player is maximizingPlayer:
        return moves[random_best_move(values, True)]
    else:
        return moves[random_best_move(values, False)]
        