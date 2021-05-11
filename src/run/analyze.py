from network.mask_policy import mask_invalid_moves, square_move_to_index
from network.encode_state import StateEncoder
from network.sample_policy import move_from_indices, from_flattened_index
from mcts.mcts import MCTSEvaluator
import torch
from torch.nn import functional as F
import chess
from utils import square_to_n_n

def top_net_moves(fen, network, T=8, temp=2, k=5, device='cpu'):
    # Get the network's output policy
    board = chess.Board(fen)
    state = StateEncoder(T).encode_state_with_history([board]).to(device)
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

def top_mcts_moves(fen, network, T=8, temp=1, k=5, device='cpu',
                   max_trials=1000, max_time_s=5):
    # Build inputs for MCTS
    state_encoder = StateEncoder(T)

    # Modified from GameRunner
    def prior_func_builder(fens):
        '''
            fens is a list of FEN strings detailing the path from the root
            of the MCTS tree up to and INCLUDING the current state from which
            a move will be selected.
        '''
        # Build the network's masked policy
        boards = (chess.Board(fen) for fen in fens[-network.T:])
        curr_state = state_encoder.encode_state_with_history(boards)
        with torch.no_grad():
            curr_state = curr_state.unsqueeze(0).to(device)
            value, net_log_probs = network(curr_state)

        net_policy = net_log_probs.exp().squeeze()
        mask_invalid_moves(net_policy, chess.Board(fens[-1]), device=device)

        def prior_func(move):
            row, col = square_to_n_n(move.from_square)
            index = square_move_to_index(move.from_square, move.to_square, move.promotion)
            return net_policy[row, col, index]

        return value, net_policy, prior_func

    # Run MCTS
    mcts_evaluator = MCTSEvaluator(fen, prior_func_builder)
    root = mcts_evaluator.mcts(max_trials=max_trials, max_time_s=max_time_s)

    # Get move distribution; computation analogous to GameRunner._sample_move
    # All moves are legal so no need to remove probability 0 moves like in top_net_moves
    visits = torch.tensor([edge.n_visits for edge in root.out_edges]).float()
    visits /= visits.sum()

    top_probs, top_indices = torch.topk(visits, k)
    top_probs = (top_prob.item() for top_prob in top_probs)
    top_ucis = [root.out_edges[i].uci for i in top_indices]

    return root.get_state_value(), list(zip(top_ucis, top_probs))
