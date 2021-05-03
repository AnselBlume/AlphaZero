import chess

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
