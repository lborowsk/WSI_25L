from two_player_games.games.connect_four import ConnectFour
from evaluation import evaluate
from best_move import make_best_move


game = ConnectFour()
maxPlayer = game.first_player


while not game.is_finished():
    if game.get_current_player() is maxPlayer:
        move = make_best_move(game.state, 2, maxPlayer)
        game.make_move(move)
    else:
        move = make_best_move(game.state, 4, maxPlayer)
        game.make_move(move)
    print(game.state)
    print(evaluate(game.state, maxPlayer))

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)