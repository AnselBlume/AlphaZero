from .mask_policy import square_move_to_index
from utils import square_to_n_n
import chess
import torch
from torch.nn import functional as F

class MCTSDist:
    '''
        The information from the MCTS distribution obtained from the root
        to be stored in the replay memory.
    '''
    def __init__(self, root, temp):
        self.fen = root.fen
        self.value = root.get_state_value()
        self.move_data = [MoveData(edge) for edge in root.out_edges]
        self.temp = temp

class MCTSPolicyEncoder:
    def get_mcts_policy(self, mcts_dist):
        '''
            Takes the MCTSDist and converts it to a gold
            distribution in the shape of the network's policy.
        '''
        policy = torch.zeros(8,8,73)

        for move in mcts_dist.move_data:
            row, col = square_to_n_n(move.from_square)
            index = square_move_to_index(move.from_square, move.to_square, move.promotion)
            # Since MCTS is very expensive on a single core machine, use softmax instead
            # of hard probabilities which are frequently zero as n_visits are sparse
            # score = move.n_visits
            score = move.n_visits ** (1 / mcts_dist.temp) # Original AlphaZero score

            policy[row,col,index] = score

        # policy = F.softmax(policy.flatten() / mcts_dist.temp, dim=0) # Softer distribution than original AlphaZero
        # policy = policy.reshape(8,8,73)
        policy /= policy.sum() # Original AlphaZero normalization

        return policy

class MoveData:
    def __init__(self, tree_edge):
        move = chess.Move.from_uci(tree_edge.uci)

        self.from_square = move.from_square
        self.to_square = move.to_square
        self.promotion = move.promotion
        self.n_visits = tree_edge.n_visits
