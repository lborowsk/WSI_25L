from two_player_games.games.connect_four import ConnectFour
from best_move import make_best_move
from collections import defaultdict
import time

def run_test(depth1, depth2, num_games=20):
    results = {'player1': 0, 'player2': 0, 'draw': 0}
    
    for _ in range(num_games):
        game = ConnectFour()
        maxPlayer = game.first_player
        
        while not game.is_finished():
            current_player = game.get_current_player()
            depth = depth1 if current_player is maxPlayer else depth2
            move = make_best_move(game.state, depth, maxPlayer)
            game.make_move(move)
    
        winner = game.get_winner()
        if winner is maxPlayer:
            results['player1'] += 1
        elif winner is not None:
            results['player2'] += 1
        else:
            results['draw'] += 1
    
    return results

def run_all_tests():
    depths = range(1, 6)
    total_tests = len(depths) ** 2
    current_test = 0
    
    all_results = defaultdict(dict)
    
    for depth1 in depths:
        for depth2 in depths:
            current_test += 1
            print(f"Running test {current_test}/{total_tests}: depth1={depth1}, depth2={depth2}")
            start_time = time.time()
            
            results = run_test(depth1, depth2)
            all_results[(depth1, depth2)] = results
            
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f} seconds")
    
    # Print results in a readable table
    print("\nResults:")
    print("Depth1 | Depth2 | Player1 Win % | Player2 Win % | Draw %")
    print("--------------------------------------------------------")
    
    for (depth1, depth2), results in all_results.items():
        total = sum(results.values())
        p1_win = (results['player1'] / total) * 100
        p2_win = (results['player2'] / total) * 100
        draw = (results['draw'] / total) * 100
        
        print(f"{depth1:6} | {depth2:6} | {p1_win:12.1f}% | {p2_win:12.1f}% | {draw:6.1f}%")

if __name__ == "__main__":
    run_all_tests()