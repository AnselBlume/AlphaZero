import chess
import torch
import constants as consts
from utils import n_n_to_square
r'''
    Policy represented by 8x8x73
    8x8: which square to pick up a piece from
    0-55: number of squares to move a piece in a dir {1,...,7} and dir to
    move it {N, NE, E, SE, S, SW, W, NW} organized as
    {N1,...,N7, NE1,...,NE7, ... ,W1,...W7,...,NW1,...,NW7}

    56-63: Knight moves |-, __|, --|, |_, _|, |--, |__, -|

    64-72: Pawn underpromotions in each possible movement/capture direction
    \N, \B, \R, |N, |B, |R, /N, /B, /R
'''
def build_legal_move_dict(board):
    legal_moves = {} # from_square -> to_square -> set(promotion)
    for move in board.legal_moves: # Handles the current player
        legal_moves.setdefault(move.from_square, {}) \
                   .setdefault(move.to_square, set()) \
                   .add(move.promotion)

    return legal_moves

def square_move_to_index(from_square, to_square, promotion=None):
    '''
        Converts the move corresponding to from_square -> to_square and promotion
        to its relevant index. Does not check for any kind of move validity
        in terms of movement or promotion.
    '''
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)

    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)

    rank_diff = to_rank - from_rank
    file_diff = to_file - from_file

    # Check for directions
    # TODO would be better to refactor these constants to offset.N, offset.NE, etc.
    if file_diff == 0: # Movement along column
        dist = abs(rank_diff)

        if rank_diff > 0: # North
            dir_offset = consts.POLICY_N_OFFSET # 0 * 7
        else: # South
            dir_offset = consts.POLICY_S_OFFSET # 4 * 7

        index = dir_offset + dist - 1
    elif rank_diff == 0: # Movement along row
        dist = abs(file_diff)

        if file_diff > 0: # East
            dir_offset = consts.POLICY_E_OFFSET # 2 * 7
        else: # West
            dir_offset = consts.POLICY_W_OFFSET # 6 * 7

        index = dir_offset + dist - 1
    elif abs(rank_diff) == abs(file_diff): # Movement along diagonal
        dist = abs(rank_diff) # or abs(file_diff)

        if rank_diff > 0 and file_diff > 0: # Northeast
            dir_offset = consts.POLICY_NE_OFFSET # 1 * 7
        elif rank_diff < 0 and file_diff > 0: # Southeast
            dir_offset = consts.POLICY_SE_OFFSET # 3 * 7
        elif rank_diff < 0 and file_diff < 0: # Southwest
            dir_offset = consts.POLICY_SW_OFFSET # 5 * 7
        else: # Northwest
            dir_offset = consts.POLICY_NW_OFFSET # 7 * 7

        index = dir_offset + dist - 1
    else: # Knight move
        if file_diff > 0: # Movement right
            if rank_diff > 0: # Movement up
                if rank_diff > file_diff: # |-
                    index = consts.POLICY_KNIGHT_UP_RIGHT
                else: # __|
                    index = consts.POLICY_KNIGHT_RIGHT_UP
            else: # Movement down
                if file_diff > -rank_diff: # --|
                    index = consts.POLICY_KNIGHT_RIGHT_DOWN
                else: # |_
                    index = consts.POLICY_KNIGHT_DOWN_RIGHT
        else: # Movement left
            if rank_diff < 0: # Movement down
                if rank_diff < file_diff: # _|
                    index = consts.POLICY_KNIGHT_DOWN_LEFT
                else: # |--
                    index = consts.POLICY_KNIGHT_LEFT_DOWN
            else: # Movement up
                if rank_diff < -file_diff: # |__
                    index = consts.POLICY_KNIGHT_LEFT_UP
                else: # -|
                    index = consts.POLICY_KNIGHT_UP_LEFT

    # Default is to promote to Queen
    if promotion is None or promotion == chess.QUEEN:
        return index

    # Direction is one of \ | /
    # Convert dir_offset of general movements to dir_offset for promotions
    if dir_offset == consts.POLICY_NW_OFFSET \
       or dir_offset == consts.POLICY_SW_OFFSET: # Vertical west
        dir_offset = consts.POLICY_PROMOTION_VW
    elif dir_offset == consts.POLICY_N_OFFSET \
         or dir_offset == consts.POLICY_S_OFFSET: # Vertical
        dir_offset = consts.POLICY_PROMOTION_V
    elif dir_offset == consts.POLICY_NE_OFFSET \
         or dir_offset == consts.POLICY_SE_OFFSET: # Vertical east
        dir_offset = consts.POLICY_PROMOTION_VE
    else:
        raise RuntimeError('Something is wrong with the computed dir_offset')

    if promotion == chess.KNIGHT:
        return dir_offset + consts.POLICY_PROMOTION_KNIGHT
    elif promotion == chess.BISHOP:
        return dir_offset + consts.POLICY_PROMOTION_BISHOP
    elif promotion == chess.ROOK:
        return dir_offset + consts.POLICY_PROMOTION_ROOK
    else:
        raise RuntimeError('The promotion target is invalid')

def mask_position(row, col, policy, legal_move_dict, device):
    mask = torch.zeros(73).to(device)

    from_square = n_n_to_square(row, col)

    # Add legal moves to mask
    if from_square in legal_move_dict:
        for to_square, promotions in legal_move_dict[from_square].items():
            for promotion in promotions:
                move_ind = square_move_to_index(from_square, to_square, promotion)
                mask[move_ind] += 1

    policy[row,col,:] *= mask

def mask_invalid_moves(policy, board, device='cpu'):
    legal_move_dict = build_legal_move_dict(board)

    for row in range(policy.shape[0]):
        for col in range(policy.shape[1]):
            mask_position(row, col, policy, legal_move_dict, device)

    policy /= policy.sum() # Renormalize to 1
