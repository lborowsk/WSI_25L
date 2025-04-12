from two_player_games.games.connect_four import ConnectFour
from evaluation import evaluate
from alphabeta_player import AlphabetaPlayer
import random


game = ConnectFour()
player_1 = AlphabetaPlayer(2, True, game.first_player, evaluate)
player_2 = AlphabetaPlayer(2, False, game.second_player, evaluate)


while not game.is_finished():
    if game.state.get_current_player() is game.first_player:
        move = player_1.make_best_move(game.state)
        game.make_move(move)
    else:
        move = player_2.make_best_move(game.state)
        game.make_move(move)
    print(evaluate(game.state, player_2))
    print(game.state)

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)