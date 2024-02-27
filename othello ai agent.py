import time
import json

class DuoOthelloAgent:
    def __init__(self):
        self.player = None
        self.opponent = None
        self.remaining_time = 0
        self.opponent_time = 0
        self.board = [['.' for _ in range(12)] for _ in range(12)]

    def read_input(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

            # First line: Player that your agent will play on this turn: X or O
            self.player = lines[0].strip()
            self.opponent = 'X' if self.player == 'O' else 'O'

            # Second line: Remaining play time for your agent and the opponent
            times = lines[1].strip().split()
            self.remaining_time = float(times[0])
            self.opponent_time = float(times[1])

            # Next 12 lines: Current state of the board
            self.board = [list(line.strip()) for line in lines[2:14]]

    def write_output(self, move, filename):
        # Assuming move is a tuple (column, row), e.g., ('c', 2)
        columns = 'abcdefghijkl'
        with open(filename, 'w') as file:
            # Convert row and col to the proper output format
            file.write(f"{columns[move[1]]}{move[0]+1}\n")

    def get_legal_moves(self, board, player):
        """
        Get all legal moves for a player given a board configuration. Legal definition: there exists opponent(s) between this unassigned position and another of my piece.
        
        :param board: A 12x12 matrix representing the game board.
        :param player: 'X' or 'O', representing the current player.
        :return: A list of tuples, each tuple is a legal move (row, col).
        """
        opponent = 'O' if player == 'X' else 'X'
        legal_moves = []
        
        # check surrounding 8 positions clockwise
        directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)] 

        for row in range(12):
            for col in range(12):
                if board[row][col] == '.': # Legal moves can only be made in empty cells.
                    for dr, dc in directions:  # loop through each of the 8 positions
                        r, c = row + dr, col + dc
                        pieces_to_flip = []
                        while 0 <= r < 12 and 0 <= c < 12 and board[r][c] == opponent: # continue search in this direction (horizontal/vertical/diagonal) until 1. off the board; 2. found '.'; 3. found my player.
                            pieces_to_flip.append((r, c)) # if current location legal, these opponent pieces are flipped 
                            r += dr
                            c += dc
                        if 0 <= r < 12 and 0 <= c < 12 and board[r][c] == player and pieces_to_flip: # check whether can found my player after a sequence of opponent pieces AND sandwiches opponent(s)
                            legal_moves.append((row, col)) # If do have a sandwich/legal move, add the current cell to the list of legal_moves
                            break  # Only need to find one sandwiching for each location to be a legal move
        return legal_moves

    def is_stable(self, board, r, c, player):
        """
        A disc is fully stable if:
            - It is in a corner.
            - It is in a completed row, column, or diagonal.
            - It is adjacent to discs that are fully stable and the entire line (row, column, diagonal) between the disc and the stable disc is controlled by the same player.

        A disc is unstable if:
            - It could be flipped in the opponent's next move.
        """
        # If a disc is in the corner, it is stable
        if (r, c) in [(0, 0), (0, 11), (11, 0), (11, 11)]:
            return True

        # Check if a disc is on a completed row, column, or diagonal
        if all(board[r][i] == player for i in range(12)) or all(board[i][c] == player for i in range(12)):
            return True

        # Check diagonals
        if r == c and all(board[i][i] == player for i in range(12)):
            return True
        # top right corner (0,11) to the bottom left corner (11,0) diagonal
        if r + c == 11 and all(board[i][11-i] == player for i in range(12)):
            return True

        # Check for stability through adjacency to stable discs
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            adjacent_r, adjacent_c = r + dr, c + dc
            # Check if the adjacent disc is stable and the line is filled with the player's discs
            if 0 <= adjacent_r < 12 and 0 <= adjacent_c < 12 and board[adjacent_r][adjacent_c] == player:
                stable_line = True
                while 0 <= adjacent_r < 12 and 0 <= adjacent_c < 12:
                    if board[adjacent_r][adjacent_c] != player:
                        stable_line = False
                        break
                    adjacent_r += dr
                    adjacent_c += dc
                if stable_line:
                    return True

        return False

    def can_be_flipped(self, board, r, c, player):
        """
        Checks if a disc at position (r, c) can be flipped by the opponent.
        This requires checking if there's an opponent's disc in each direction that is followed by a disc of the current player.
        """
        opponent = 'O' if player == 'X' else 'X'
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            adjacent_r, adjacent_c = r + dr, c + dc
            if 0 <= adjacent_r < 12 and 0 <= adjacent_c < 12 and board[adjacent_r][adjacent_c] == opponent:
                while 0 <= adjacent_r < 12 and 0 <= adjacent_c < 12:
                    if board[adjacent_r][adjacent_c] == '.':  # Empty space
                        break
                    if board[adjacent_r][adjacent_c] == player:
                        return True  # Found a path to flip
                    adjacent_r += dr
                    adjacent_c += dc
        return False
    
    def is_semi_stable(self, board, r, c, player):
        """
        A disc is semi-stable if it is not fully stable but cannot be flipped in the next move.
        """
        return not self.is_stable(board, r, c, player) and not self.can_be_flipped(board, r, c, player)


    def calculate_stability(self, board, player):
        stability_score = 0
        for r in range(12):
            for c in range(12):
                if board[r][c] == player:
                    if self.is_stable(board, r, c, player):
                        stability_score += 1  # Fully stable
                    elif self.is_semi_stable(board, r, c, player):
                        stability_score += 0.5  # Semi-stable
                    else:
                        stability_score -= 1  # Unstable or potentially flippable
        return stability_score



    def evaluate_board(self, board, player):
        """the minimax with alpha-beta algorithm is used to explore different game states by simulating moves for both the player and the opponent. 
           When the algorithm reaches a certain depth, it stops going deeper due to time limits, and will use this evaluate_board func to estimate the moves' result score and sort the moves with this score. 
           By exploring moves that lead to better board states first, we can improve the efficiency of the alpha-beta pruning.
        """
        # Return a heuristic value of the board

        empty_spots = sum(1 for row in board for spot in row if spot == '.')
        total_spots = 12 * 12  # Total spots on a 12x12 board
        filled_percentage = ((total_spots - empty_spots) / total_spots) * 100

        if filled_percentage < 33: # early stage
            corner_weight, edge_weight, mobility_weight, frontier_weight, tiles_weight, stability_weight = 20, 10, 5, -3, 1, 5 # Early game prioritize board control and mobility. 
        elif filled_percentage < 66: # mid stage
            corner_weight, edge_weight, mobility_weight, frontier_weight, tiles_weight, stability_weight = 25, 10, 15, -5, 1, 15 # focus towards securing stable positions and maximizing piece count. 
        else:
            corner_weight, edge_weight, mobility_weight, frontier_weight, tiles_weight, stability_weight = 30, 5, 0, -1, 2, 25  #  in endgame, the number of pieces is more important than mobility or frontier tiles. 
        
        opponent = 'O' if player == 'X' else 'X'
        player_tiles = 0
        opponent_tiles = 0
        player_frontier_tiles = 0 # tiles adjacent to empty spaces
        opponent_frontier_tiles = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # different grid locations have different desirability
        corners = [(0, 0), (0, 11), (11, 0), (11, 11)]
        edges = [(i, 0) for i in range(1, 11)] + \
                [(i, 11) for i in range(1, 11)] + \
                [(0, i) for i in range(1, 11)] + \
                [(11, i) for i in range(1, 11)]
        
        for r in range(12):
            for c in range(12):
                if board[r][c] == player:
                    player_tiles += 1
                    for dr, dc in directions:
                        if 0 <= r + dr < 12 and 0 <= c + dc < 12 and board[r + dr][c + dc] == '.':
                            player_frontier_tiles += 1
                            break # once an empty space is found -> it is a frontier
                elif board[r][c] == opponent:
                    opponent_tiles += 1
                    for dr, dc in directions:
                        if 0 <= r + dr < 12 and 0 <= c + dc < 12 and board[r + dr][c + dc] == '.':
                            opponent_frontier_tiles += 1
                            break

        # Corners captured
        player_corners = sum(1 for x, y in corners if board[x][y] == player)
        opponent_corners = sum(1 for x, y in corners if board[x][y] == opponent)

        # Edge control
        player_edges = sum(1 for x, y in edges if board[x][y] == player)
        opponent_edges = sum(1 for x, y in edges if board[x][y] == opponent)

        # Mobility
        player_mobility = len(self.get_legal_moves(board, player))
        opponent_mobility = len(self.get_legal_moves(board, opponent))

        score = ((player_tiles - opponent_tiles) * tiles_weight) + \
            ((player_corners - opponent_corners) * corner_weight) + \
            ((player_edges - opponent_edges) * edge_weight) - \
            ((player_frontier_tiles - opponent_frontier_tiles) * frontier_weight) + \
            ((player_mobility - opponent_mobility) * mobility_weight)

        stability_score = self.calculate_stability(board, player)
        score += stability_score * stability_weight


        return score
    
    def make_move(self, board, move, player):
        # make the legal move on board, flip all those to flip based on player X/O
        # Assume move is a tuple (row, col) where the player wishes to place their piece.
        new_board = [list(row) for row in board]  # Create a copy of the board to modify
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 directions

        # Place the piece on the board
        new_board[move[0]][move[1]] = player

        # Opponent's pieces to flip
        pieces_to_flip = []

        # Determine the opponent's symbol
        opponent = 'O' if player == 'X' else 'X'

        # Check all directions
        for d in directions:
            x, y = move[0], move[1]
            x += d[0]
            y += d[1]
            pieces_in_direction = []

            while 0 <= x < len(new_board) and 0 <= y < len(new_board[0]) and new_board[x][y] == opponent: # search opponents in this direction
                pieces_in_direction.append((x, y))
                x += d[0]
                y += d[1]

            if 0 <= x < len(new_board) and 0 <= y < len(new_board[0]) and new_board[x][y] == player: # confirm whether sandwich
                pieces_to_flip.extend(pieces_in_direction)

        # Flip the opponent's pieces
        for x, y in pieces_to_flip:
            new_board[x][y] = player

        return [''.join(row) for row in new_board] # Convert back to the original board format


    def alpha_beta(self, board, depth, alpha, beta, maximizing_player, start_time, time_for_this_move):
        # maximizing_player: A boolean. True if the current player is the maximizer (player me) and False if the current player is the minimizer (player opponent).

        current_time = time.time()
        if current_time - start_time >= time_for_this_move or depth == 0:
            return self.evaluate_board(board, self.player if maximizing_player else self.opponent)

        legal_moves = self.get_legal_moves(board, self.player if maximizing_player else self.opponent)
        if not legal_moves:  # If no legal moves, evaluate board
            return self.evaluate_board(board, self.player if maximizing_player else self.opponent)

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                new_board = self.make_move(board, move, self.player)
                eval = self.alpha_beta(new_board, depth - 1, alpha, beta, False, start_time, time_for_this_move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_board = self.make_move(board, move, self.opponent)
                eval = self.alpha_beta(new_board, depth - 1, alpha, beta, True,start_time, time_for_this_move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval


    def best_move(self, board, player, time_left):
        best_val = float('-inf') if (player == self.player) else float('inf') # self.player is the X/O read from read_input, player is alternating
        best_move = None
        legal_moves = self.get_legal_moves(board, player)
        depth = 1
        start_time = time.time()
        

        empty_spots = sum(1 for row in board for spot in row if spot == '.')
        estimated_moves_remaining = max(empty_spots, 1)  # Avoid division by zero

        # Allocate time for this move (more sophisticated heuristics could be used here)
        time_for_this_move = time_left / estimated_moves_remaining

        # Adjust the initial depth based on the phase of the game
        if empty_spots > 95:  # Early game
            initial_depth = 2
        elif empty_spots > 46:  # Midgame
            initial_depth = 3
        else:  # Endgame
            initial_depth = 4 # Start deeper if in the endgame

        # Iterative deepening within the allocated time for this move
        depth = initial_depth
        # Iterative deepening within the allocated time for this move
        while True:
            current_time = time.time()
            if current_time - start_time >= time_for_this_move:
                break  # Ensure we don't start a new depth if time is up

            for move in legal_moves:
                new_board = self.make_move(board, move, player)
                val = self.alpha_beta(new_board, depth, float('-inf'), float('inf'), player == self.player, start_time, time_for_this_move)
                
                
                if (player == self.player and val > best_val) or (player != self.player and val < best_val): # the algorithm is evaluating moves for myself/opponent, who is trying to max/minimize the score
                    best_val = val
                    best_move = move
            if time.time() - start_time >= time_for_this_move:
                break
                

            depth += 1 # Iterative deepening

        return best_move if best_move is not None else legal_moves[0]

        
    def play(self):
        self.read_input('input.txt')

        # Determine the best move using the remaining time
        move = self.best_move(self.board, self.player, self.remaining_time)

        # Write the best move to output.txt if a move is available
        if move:
            self.write_output(move, 'output.txt')
        else:
            # No legal moves available - this should not happen as per the project description,
            # but it's here to handle unexpected cases gracefully.
            print("No legal moves available.")



agent = DuoOthelloAgent()

# Run the agent's play method - in the actual game, there would be a condition to exit this loop?
agent.play()