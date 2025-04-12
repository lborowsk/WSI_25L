import math
from two_player_games.games.connect_four import ConnectFourState
from two_player_games.player import Player

def evaluate(state: 'ConnectFourState', maximizing_player: 'Player') -> float:
    """
    Heuristic based on the power of continuous chips in lines.
    Returns:
        - +inf if maximizing_player wins,
        - -inf if opponent wins,
        - Heuristic score (exponentially weighted by consecutive chips).
    """
    winner = state.get_winner()
    if winner == maximizing_player:
        return math.inf
    elif winner is not None:
        return -math.inf

    score = 0
    opponent = state._other_player if maximizing_player == state._current_player else state._current_player

    # Evaluate all lines (rows, columns, diagonals)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        score += _evaluate_line(state, maximizing_player, dx, dy)
        score -= _evaluate_line(state, opponent, dx, dy)

    # Bonus for center control (middle 3 columns)
    center_cols = [len(state.fields) // 2 - 1, len(state.fields) // 2, len(state.fields) // 2 + 1]
    for col in center_cols:
        if 0 <= col < len(state.fields):
            for row in range(len(state.fields[col])):
                if state.fields[col][row] == maximizing_player:
                    score += 2
                elif state.fields[col][row] == opponent:
                    score -= 2

    return score

def _evaluate_line(state: 'ConnectFourState', player: 'Player', dx: int, dy: int) -> int:
    """
    Scores a line (row/column/diagonal) exponentially based on consecutive chips.
    """
    total = 0
    cols, rows = len(state.fields), len(state.fields[0]) if state.fields else 0

    for col in range(cols):
        for row in range(rows):
            if col + 3 * dx >= cols or row + 3 * dy < 0 or row + 3 * dy >= rows:
                continue  # Skip incomplete lines

            consecutive = 0
            for i in range(4):
                cell = state.fields[col + i * dx][row + i * dy]
                if cell == player:
                    consecutive += 1
                elif cell is not None:  # Opponent's chip blocks the line
                    consecutive = 0
                    break

            if consecutive > 0:
                total += 2 ** consecutive  # Exponential scoring (e.g., 2^2=4, 2^3=8)

    return total