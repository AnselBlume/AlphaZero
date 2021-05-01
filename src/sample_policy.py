import torch
import utils
import chess
import constants as consts

def sample_policy(masked_policy, board):
    '''
        Flatten policy tensor and sample chess.Move from the policy.
        Policy should be masked to have only legal moves for the given board state.

        Torch doesn't have an equivalent of numpy's unravel_index yet.
    '''
    sampled_ind = masked_policy.flatten().multinomial(1)
    indices = from_flattened_index(sampled_ind, masked_policy.shape)

    return move_from_indices(indices, board)

def move_from_indices(indices, board):
    '''
        Converts a triplet of indices into a policy into its corresponding
        chess.Move object.
    '''
    from_square = get_from_square(indices)
    to_square = get_to_square(from_square, indices)
    promotion = get_promotion(from_square, to_square, indices, board)

    return chess.Move(from_square, to_square, promotion)

def get_from_square(indices):
    return utils.n_n_to_square(*indices[:2])

def get_to_square(from_square, indices):
    rank_delta, file_delta = deltas_from_index(from_square, indices[-1])

    to_rank = chess.square_rank(from_square) + rank_delta
    to_file = chess.square_file(from_square) + file_delta

    return chess.square(to_file, to_rank)

def deltas_from_index(from_square, index):
    rank_delta = 0
    file_delta = 0

    # TODO make this cleaner with a data class
    if index < consts.POLICY_KNIGHT_OFFSET:
        dist = index % 7 + 1
        offset = index - dist + 1

        if offset == consts.POLICY_N_OFFSET:
            rank_delta = dist
        elif offset == consts.POLICY_NE_OFFSET:
            rank_delta = dist
            file_delta = dist
        elif offset == consts.POLICY_E_OFFSET:
            file_delta = dist
        elif offset == consts.POLICY_SE_OFFSET:
            rank_delta = -dist
            file_delta = dist
        elif offset == consts.POLICY_S_OFFSET:
            rank_delta = -dist
        elif offset == consts.POLICY_SW_OFFSET:
            rank_delta = -dist
            file_delta = -dist
        elif offset == consts.POLICY_W_OFFSET:
            file_delta = -dist
        else: # consts.POLICY_NW_OFFSET
            rank_delta = dist
            file_delta = -dist
    elif index < consts.POLICY_PROMOTION_OFFSET: # Knight movement
        if index == consts.POLICY_KNIGHT_UP_RIGHT:
            rank_delta = 2
            file_delta = 1
        elif index == consts.POLICY_KNIGHT_RIGHT_UP:
            rank_delta = 1
            file_delta = 2
        elif index == consts.POLICY_KNIGHT_RIGHT_DOWN:
            rank_delta = -1
            file_delta = 2
        elif index == consts.POLICY_KNIGHT_DOWN_RIGHT:
            rank_delta = -2
            file_delta = 1
        elif index == consts.POLICY_KNIGHT_DOWN_LEFT:
            rank_delta = -2
            file_delta = -1
        elif index == consts.POLICY_KNIGHT_LEFT_DOWN:
            rank_delta = -1
            file_delta = -2
        elif index == consts.POLICY_KNIGHT_LEFT_UP:
            rank_delta = 1
            file_delta = -2
        else: # consts.POLICY_UP_LEFT
            rank_delta = 2
            file_delta = -1
    else: # Underpromotion; doesn't check for validity
        if chess.square_rank(from_square) == 6: # Rank 7
            rank_delta = 1
        else: # Must be rank 2
            rank_delta = -1

        if index < consts.POLICY_PROMOTION_V: # West
            file_delta = -1
        elif index < consts.POLICY_PROMOTION_VE: # North/South
            file_delta = 0
        else: # East
            file_delta = 1

    return rank_delta, file_delta

def get_promotion(from_square, to_square, indices, board):
    to_rank = chess.square_rank(to_square)

    # Needs to be a pawn on one of the edge ranks
    # We assume this move is valid if it's a pawn
    if board.piece_type_at(from_square) != chess.PAWN \
       or (to_rank != 7 and to_rank != 0):
        return None

    # Queen
    if indices[-1] < consts.POLICY_PROMOTION_OFFSET:
        return chess.QUEEN

    # Underpromotion
    underpromotion_ind = (indices[-1] - consts.POLICY_PROMOTION_OFFSET) % 3
    if underpromotion_ind == consts.POLICY_PROMOTION_KNIGHT:
        return chess.KNIGHT
    elif underpromotion_ind == consts.POLICY_PROMOTION_BISHOP:
        return chess.BISHOP
    else:
        return chess.ROOK

# Can see discussion here for different implementation of from_flattened_index
# This is really just conversion to and from a number in multiple different bases
# https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987
def to_flattened_index(indices, dims):
    '''
        Converts indices indexing into an nd-array of shape dims
        into the corresponding index in the flattened array.

        Can check the result by
        a = torch.zeros(dims)
        a[indices] = 1
        a.flatten().argmax()
    '''
    flat_index = indices[-1]
    prod_so_far = dims[-1] # Product of dimensions

    for i in range(len(dims) - 2, -1, -1):
        flat_index += indices[i] * prod_so_far
        prod_so_far *= dims[i]

    return flat_index

def from_flattened_index(index, dims):
    '''
        Converts an index indexing into a one-dimensional array
        into its multi-dimensional index in the shape specified by dims.

        Can go from right to left by doing modulos as in the discuss.pytorch link
        above. This code computes indices left to right.

        Can be tested by checking that
        to_flattened_index(from_flattened_index(index, dims), dims) == index
    '''
    indices = []
    products = [1] * len(dims)

    for i in range(len(dims) - 1, 0, -1):
        products[i - 1] = dims[i] * products[i]

    for prod in products:
        ind = index // prod
        indices.append(ind)
        index -= ind * prod

    return tuple(indices)
