import torch
from network.encode_state import StateEncoder
from network.mask_policy import mask_invalid_moves, square_move_to_index
from network.network import Network
from network.encode_dist import MCTSDist
import network.sample_policy
import chess
from mcts.mcts import MCTSEvaluator
from utils import square_to_n_n
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)

START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

MAX_TURNS = float('inf')

class GameRunner:
    def __init__(self, T, temp=1, temp_divisor=1.005, std_ucb=False, max_trials=1000, max_time_s=10,
                 device='cpu'):
        self.T = T
        self.temp = temp
        self.temp_divisor = temp_divisor
        self.std_ucb = std_ucb
        self.max_trials = max_trials
        self.max_time_s = max_time_s
        self.state_encoder = StateEncoder(T)
        self.device = device

    def play_game(self, network, start_fen=START_FEN, use_rollouts=False):
        network.eval()
        board = chess.Board(start_fen)
        fen_history = [] # Could use a deque here

        mcts_dists = []
        subtree = None # The previous MCTS subtree from which MCTS will be started
        turn = 0
        logger.info(f'Starting FEN: {board.fen()}')
        while board.outcome() is None:
            mcts_dist, subtree = self.play_turn(
                board,
                network,
                fen_history[-self.T:],
                subtree=subtree,
                turn=turn,
                use_rollouts=use_rollouts
            )
            mcts_dists.append(mcts_dist)
            fen_history.append(board.fen())

            turn += 1
            logger.info(f'FEN after turn {turn}: {board.fen()}')

            if turn >= MAX_TURNS:
                break

        # If we're not using rollouts, we need to set the values to the actual winner as opposed
        # to the estimated rollout values
        if not use_rollouts: # Set the values to 1 for the winning states, -1 for losing
            winner = board.outcome().winner

            if winner is None: # Draw
                white_val = 0
                black_val = 0
            else:
                white_val = 1 if winner == chess.WHITE else -1
                black_val = 1 if winner == chess.BLACK else -1

            white_val = float(white_val)
            black_val = float(black_val)

            for i in range(0, len(mcts_dists), 2):
                mcts_dists[i].value = white_val
            for i in range(1, len(mcts_dists), 2):
                mcts_dists[i].value = black_val

        # Build mcts_dist_histories
        if self.T > len(mcts_dists):
            return board, [mcts_dists]

        mcts_dist_histories = [mcts_dists[i:i+self.T] for i in range(len(mcts_dists)-self.T+1)]

        return board, mcts_dist_histories

    def play_turn(self, board, network, fen_history, subtree=None, turn=1, use_rollouts=True):
        '''
            fen_history is a list of FEN strings detailing the history up until
            and not including the current state.

            board is a chess.Board describing the current state.
        '''
        prior_func_builder = self._get_prior_func_builder(network, fen_history)

        mcts_evaluator = MCTSEvaluator(
            board.fen(),
            prior_func_builder=prior_func_builder,
            subtree=subtree
        )

        root = mcts_evaluator.mcts(
            std_ucb=self.std_ucb,
            max_trials=self.max_trials,
            max_time_s=self.max_time_s,
            use_rollouts=use_rollouts
        )

        effective_temp = self.temp / self.temp_divisor ** turn
        sampled_move, sampled_ind = self._sample_move(root, effective_temp)
        logger.debug(f'Sampled move: {sampled_move}')
        board.push(sampled_move)

        return MCTSDist(root, effective_temp), root.out_edges[sampled_ind].to_node

    def _sample_move(self, root, effective_temp):
        visits = torch.tensor([edge.n_visits for edge in root.out_edges])

        # In the same way as in MCTSPolicyEncoder, perform softmax over visits
        # as opposed to this harder version used in the original paper
        visits = visits ** (1 / effective_temp) # Original AlphaZero probability
        visits /= visits.sum() # Original AlphaZero formulation
        #visits = F.softmax(visits / effective_temp, dim=0)

        sampled_ind = torch.multinomial(visits, 1).item()
        sampled_edge = root.out_edges[sampled_ind]

        return chess.Move.from_uci(sampled_edge.uci), sampled_ind

    def _get_prior_func_builder(self, network, fen_history):
        '''
            fen_history is a list of FEN strings detailing the history up until
            and not including the root of the MCTS tree.
        '''
        def prior_func_builder(fens):
            '''
                fens is a list of FEN strings detailing the path from the root
                of the MCTS tree up to and INCLUDING the current state from which
                a move will be selected.
            '''
            # Build the network's masked policy
            fens = fen_history + fens # Combine the history before the root of mcts and after
            boards = (chess.Board(fen) for fen in fens[-self.T:])
            curr_state = self.state_encoder.encode_state_with_history(boards)
            with torch.no_grad():
                curr_state = curr_state.unsqueeze(0).to(self.device)
                value, net_log_probs = network(curr_state)

            net_policy = net_log_probs.exp().squeeze()
            mask_invalid_moves(net_policy, chess.Board(fens[-1]), device=self.device)

            def prior_func(move):
                row, col = square_to_n_n(move.from_square)
                index = square_move_to_index(move.from_square, move.to_square, move.promotion)
                return net_policy[row, col, index]

            return value, net_policy, prior_func

        return prior_func_builder

if __name__ == '__main__':
    pass
