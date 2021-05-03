import torch
import chess
from utils import square_to_n_n
# Pages 12-15 of paper https://arxiv.org/pdf/1712.01815.pdf are most relevant
N = 8 # Board dimensions
M = 14 # 6 p1 pieces, 6 p2 pieces, 2 repetition counts
L = 7

class StateEncoder:
    '''
        Dimensions N x N x (MT + L)

        N x N
        a8 -> (0, 0), a0 -> (7, 0), h8 -> (0, 7), h0 -> (7, 7)

        M array follows values of chess lib's piece types offset by one
        M: P, N, B, R, Q, K, p, n, b, r, q, k, Rep, rep

        L: 1 player color, 1 total move count, 2 p1 castling, 2 p2 castling, 1 no-progress count
    '''
    def __init__(self, T=2):
        self.T = T # Length of history

    def encode_state_with_history(self, boards):
        '''
            Returns a state encoding of the final board in boards with
            a history from the previous boards in the iterable of length T - 1.
        '''
        curr_state = self.get_empty_state()
        for board in boards:
            curr_state = self.encode_state(board, curr_state)

        return curr_state

    def piece_to_index(self, piece, board):
        '''
            Returns index of the piece in the M array (not accounting for T offset).
        '''
        # Offset for player or opponent's piece
        offset = 0 if piece.color == board.turn else 6
        return offset + piece.piece_type - 1

    def encode_state(self, board, prev_state):
        '''
            Returns the input tensor to be processed by AlphaZero.
        '''
        T = self.T
        state = torch.zeros(M*T+L, N, N)

        # Encode history, dropping the oldest state
        state[:M*(T-1), ...] = prev_state[M:M*T,...]

        # Encode pieces
        t_offset = M*(T-1)
        for square, piece in board.piece_map().items():
            piece_ind = self.piece_to_index(piece, board)
            row, col = square_to_n_n(square)
            state[t_offset + piece_ind, row, col] = 1

        # TODO Potentially encode repeat count for both players

        # Encode current player color
        state[M*T, ...] += board.turn

        # Encode total move count
        state[M*T + 1, ...] += len(board.move_stack)

        # Encode player castling
        state[M*T + 2, ...] += board.has_queenside_castling_rights(board.turn)
        state[M*T + 3, ...] += board.has_kingside_castling_rights(board.turn)

        # Encode opponent castling
        state[M*T + 4, ...] += board.has_queenside_castling_rights(not board.turn)
        state[M*T + 5, ...] += board.has_kingside_castling_rights(not board.turn)

        # TODO Potentially encode progress count

        return state

    def get_empty_state(self):
        return torch.zeros(M*(self.T)+L, N, N)
