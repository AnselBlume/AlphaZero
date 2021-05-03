import torch
from network.encode_state import StateEncoder
from network.mask_policy import mask_invalid_moves, square_move_to_index
from network.network import Network, to_probabilities
from network.encode_dist import MCTSDist
import network.sample_policy
import chess
from mcts.mcts import MCTSEvaluator
from utils import square_to_n_n
import logging

logger = logging.getLogger(__name__)

class GameRunner:
    def __init__(self, T, temp=2, std_ucb=False, max_trials=500, max_time_s=10,
                 device='cpu'):
        self.T = T
        self.temp = temp
        self.std_ucb = std_ucb
        self.max_trials = max_trials
        self.max_time_s = max_time_s
        self.state_encoder = StateEncoder(T)
        self.device = device

    def play_game(self, network):
        board = chess.Board()
        fen_history = [] # Could use a deque here

        mcts_dists = []
        logger.info(f'Current FEN: {board.fen()}')
        while board.outcome() is None:
            mcts_dist = self.play_turn(board, network, fen_history[-self.T:])
            mcts_dists.append(mcts_dist)
            fen_history.append(board.fen())
            logger.info(f'Current FEN: {board.fen()}')

        # Build mcts_dist_histories
        if self.T > len(mcts_dists):
            return [mcts_dists]

        mcts_dist_histories = [mcts_dists[i:i+T] for i in range(len(mcts_dists)-T+1)]

        return board, mcts_dist_histories

    def play_turn(self, board, network, fen_history):
        '''
            fen_history is a list of FEN strings detailing the history up until
            and not including the current state.

            board is a chess.Board describing the current state.
        '''
        prior_func_builder = self._get_prior_func_builder(network, fen_history)
        mcts_evaluator = MCTSEvaluator(board.fen(), prior_func_builder)
        root = mcts_evaluator.mcts(
            std_ucb=self.std_ucb,
            max_trials=self.max_trials,
            max_time_s=self.max_time_s
        )
        sampled_move = self._sample_move(root)
        logger.info(f'Sampled move: {sampled_move}')
        board.push(sampled_move)

        return MCTSDist(root)

    def _sample_move(self, root):
        visits = torch.tensor([edge.n_visits for edge in root.out_edges])
        visits = visits ** (1 / self.temp)
        visits /= visits.sum()

        sampled_ind = torch.multinomial(visits, 1).item()
        sampled_edge = root.out_edges[sampled_ind]

        return chess.Move.from_uci(sampled_edge.uci)

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
                _, net_logits = network(curr_state)

            net_policy = to_probabilities(net_logits).squeeze()
            mask_invalid_moves(net_policy, chess.Board(fens[-1]), device=self.device)

            def prior_func(move):
                row, col = square_to_n_n(move.from_square)
                index = square_move_to_index(move.from_square, move.to_square, move.promotion)
                return net_policy[row, col, index]

            return prior_func

        return prior_func_builder

if __name__ == '__main__':
    pass
