import chess
from network.mask_policy import mask_invalid_moves
from network.encode_state import StateEncoder
from network.sample_policy import move_from_indices, from_flattened_index
import torch

N = 8

def square_to_n_n(square):
    '''
        Returns the indices in the N x N board corresponding to square.

        N x N
        a8 -> (0, 0), a0 -> (7, 0), h8 -> (0, 7), h0 -> (7, 7)
    '''
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    return N - (rank + 1), file

def n_n_to_square(row, col):
    '''
        Inverse of square_to_n_n.
    '''
    rank = N - 1 - row
    file = col

    return chess.square(file, rank)

def top_net_moves(board, network, temp=2, k=5, device='cpu'):
    # Get the network's output policy
    state = StateEncoder().encode_state_with_history([board]).to(device)
    with torch.no_grad():
        val, log_probs = network(state.unsqueeze(0))
    val, probs = val.item(), log_probs.exp().squeeze()
    mask_invalid_moves(probs, board, device)

    # Get the top legal moves
    top_probs, top_indices = torch.topk(probs.flatten(), k)
    gt_zero = top_probs > 0 # Make sure we aren't collecting illegal moves
    top_probs, top_indices = top_probs[gt_zero], top_indices[gt_zero]

    # Output the value of the current state and the top moves with probabilities
    shape_indices = [from_flattened_index(index, (8,8,73)) for index in top_indices]
    top_ucis = [move_from_indices(indices, board).uci() for indices in shape_indices]
    top_probs = (top_prob.item() for top_prob in top_probs)

    return val, list(zip(top_ucis, top_probs))

def top_mcts_moves(hi, temp=2):
    pass
