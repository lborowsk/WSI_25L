from two_player_games.game import Game
from two_player_games.player import Player
from two_player_games.move import Move
from two_player_games.state import State


def alphabeta(game_state: State, depth: int, a, b, maximizingPlayer: Player, evaluate: callable):
    
    if depth == 0 or game_state.is_finished:
        return evaluate(game_state, maximizingPlayer)
    
    if game_state.get_current_player() is maximizingPlayer:
        value = float('-inf')

        for move in game_state.get_moves():
            value = max(value, alphabeta(game_state.make_move(move), depth - 1, a, b, maximizingPlayer, evaluate))
            a = max(a, value)
            if value >= b:
                break
        
        return value
    else:
        value = float('inf')

        for move in game_state.get_moves():
            value = min(value, alphabeta(game_state.make_move(move), depth - 1, a, b, maximizingPlayer, evaluate))
            b = min(b, value)
            if value <= a:
                break
        
        return value


    


