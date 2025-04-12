from two_player_games.games.connect_four import ConnectFour
from evaluation import evaluate
from alphabeta_player import make_best_move
import random


game = ConnectFour()
maxPlayer = game.first_player


while not game.is_finished():
    if game.get_current_player() is maxPlayer:
        move = make_best_move(game.state, 3, maxPlayer)
        game.make_move(move)
    else:
        move = make_best_move(game.state, 3, maxPlayer)
        game.make_move(move)
    print(evaluate(game.state, maxPlayer))
    print(game.state)

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)