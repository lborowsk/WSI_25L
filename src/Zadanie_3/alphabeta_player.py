from two_player_games.game import Game
from two_player_games.state import State
from two_player_games.player import Player
from alphabeta import alphabeta
from evaluation import evaluate
import random

class AlphabetaPlayer:
    
    def __init__(self, depth: int, maximizing: bool, player: Player, evaluate: callable):
        self.depth = depth
        self.maximizing = maximizing
        self.player = player
        self.evaluate = evaluate
    
    def random_best_move(self, lst: list[int]):
        if self.maximizing:
            max_val = max(lst)
            max_indices = [i for i, x in enumerate(lst) if x == max_val]
            return random.choice(max_indices)
        else:
            min_val = min(lst)
            min_indices = [i for i, x in enumerate(lst) if x == min_val]
            return random.choice(min_indices)


    def make_best_move(self, game_state: State):
        moves = []
        values = []
        for move in game_state.get_moves():
            moves.append(move)
            values.append(alphabeta(game_state.make_move(move), self.depth, float('-inf'), float('inf'), self.player, self.evaluate))
        
        return moves[self.random_best_move(values)]
        