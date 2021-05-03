import unittest
import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
from network.sample_policy import *
import chess
import torch
import constants as consts
from utils import *

class TestSamplePolicy(unittest.TestCase):
    def test_to_flattened_index(self):
        torch.manual_seed(42)

        # Test 1
        dims = (63, 111, 30, 14)
        indices = (62, 84, 22, 11)
        index = to_flattened_index(indices, dims)

        expected = torch.zeros(dims)
        expected[indices] += 1
        expected = expected.flatten().argmax()

        self.assertEqual(index, expected)

        # Test 2
        dims = (4, 3, 3)
        indices = (2, 1, 1)
        index = to_flattened_index(indices, dims)

        expected = 22

        self.assertEqual(index, expected)

        # Test 3
        num_dims = 5
        dims = (torch.rand(num_dims) * 75).int() + 1
        indices = (torch.rand(num_dims) * dims).int()
        dims = tuple(dims.tolist())
        indices = tuple(indices.tolist())

        #dims = dims.tolist()
        #indices = indices.tolist()
        index = to_flattened_index(indices, dims)

        expected = torch.zeros(dims)
        expected[tuple(indices)] += 1
        expected = expected.flatten().argmax()

        self.assertEqual(index, expected)

    def test_from_flattened_index(self):
        # Test 1
        dims = (32, 12, 5)
        index = 100

        indices = from_flattened_index(index, dims)

        self.assertEqual(index, to_flattened_index(indices, dims))

        # Test 2
        dims = (5, 3, 4)
        index = 17

        indices = from_flattened_index(index, dims)
        expected = (1, 1, 1)

        self.assertEqual(expected, indices)

        # Test 3
        dims = (45, 32, 7, 9)
        index = 180293

        indices = from_flattened_index(index, dims)

        self.assertEqual(index, to_flattened_index(indices, dims))

    def test_get_from_square(self):
        # Test 1
        indices = [1, 2, 3]
        n_n = [1, 2]

        expected = n_n_to_square(*n_n)
        actual = get_from_square(indices)

        self.assertEqual(expected, actual)

        # Test 2
        indices = [10, 8, 25]
        n_n = [10, 8]

        expected = n_n_to_square(*n_n)
        actual = get_from_square(indices)

        self.assertEqual(expected, actual)

    def test_get_to_square(self):
        from_square = chess.E4
        indices = [*square_to_n_n(from_square), -1]

        # Test N
        indices[-1] = consts.POLICY_N_OFFSET + 3
        expected = chess.E8
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Test NE
        indices[-1] = consts.POLICY_NE_OFFSET + 2
        expected = chess.H7
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Test E
        indices[-1] = consts.POLICY_E_OFFSET
        expected = chess.F4
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Test SE
        indices[-1] = consts.POLICY_SE_OFFSET + 2
        expected = chess.H1
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Change from_square
        from_square = chess.H8
        indices = [*square_to_n_n(from_square), -1]

        # Test S
        indices[-1] = consts.POLICY_S_OFFSET + 6
        expected = chess.H1
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Test SW
        indices[-1] = consts.POLICY_SW_OFFSET + 6
        expected = chess.A1
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Test W
        indices[-1] = consts.POLICY_W_OFFSET + 5
        expected = chess.B8
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Change from_square
        from_square = chess.H1
        indices = [*square_to_n_n(from_square), -1]

        # Test NW
        indices[-1] = consts.POLICY_NW_OFFSET + 6
        expected = chess.A8
        actual = get_to_square(from_square, indices)

        # Test Knight
        # Change from_square
        from_square = chess.E4
        indices = [*square_to_n_n(from_square), -1]

        # Up right
        indices[-1] = consts.POLICY_KNIGHT_UP_RIGHT
        expected = chess.F6
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Right up
        indices[-1] = consts.POLICY_KNIGHT_RIGHT_UP
        expected = chess.G5
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Right down
        indices[-1] = consts.POLICY_KNIGHT_RIGHT_DOWN
        expected = chess.G3
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Down right
        indices[-1] = consts.POLICY_KNIGHT_DOWN_RIGHT
        expected = chess.F2
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Down left
        indices[-1] = consts.POLICY_KNIGHT_DOWN_LEFT
        expected = chess.D2
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Left down
        indices[-1] = consts.POLICY_KNIGHT_LEFT_DOWN
        expected = chess.C3
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Left up
        indices[-1] = consts.POLICY_KNIGHT_LEFT_UP
        expected = chess.C5
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Up left
        indices[-1] = consts.POLICY_KNIGHT_UP_LEFT
        expected = chess.D6
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Test underpromotion
        # Change from_square
        from_square = chess.E7
        indices = [*square_to_n_n(from_square), -1]

        # Northwest
        indices[-1] = consts.POLICY_PROMOTION_VW
        expected = chess.D8
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # North
        indices[-1] = consts.POLICY_PROMOTION_V
        expected = chess.E8
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Northeast
        indices[-1] = consts.POLICY_PROMOTION_VE
        expected = chess.F8
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Change from_square
        from_square = chess.G2
        indices = [*square_to_n_n(from_square), -1]

        # Southwest
        indices[-1] = consts.POLICY_PROMOTION_VW
        expected = chess.F1
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # South
        indices[-1] = consts.POLICY_PROMOTION_V
        expected = chess.G1
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

        # Southeast
        indices[-1] = consts.POLICY_PROMOTION_VE
        expected = chess.H1
        actual = get_to_square(from_square, indices)

        self.assertEqual(expected, actual)

    def test_get_promotion(self):
        # https://lichess.org/editor/8/6P1/8/1k6/8/5K2/8/8_w_-_-_0_1
        # Pawn at G7
        north_board = chess.Board('8/6P1/8/1k6/8/5K2/8/8 w - - 0 1')
        north_from_square = chess.G7

        # https://lichess.org/editor/8/8/8/1k6/8/5K2/1p6/8_w_-_-_0_1
        # Pawn at B2
        south_board = chess.Board('8/8/8/1k6/8/5K2/1p6/8 w - - 0 1')
        south_from_square = chess.B2

        indices = [-1, -1, -1]
        expecteds = [
            chess.QUEEN,
            chess.BISHOP,
            chess.KNIGHT,
            chess.ROOK,
        ]

        def test_promotion_dir(from_square, to_square, promotions, expected, board):
            for promotion, expected in zip(promotions, expecteds):
                indices[-1] = promotion
                actual = get_promotion(from_square, to_square, indices, board)
                self.assertEqual(expected, actual)

        # West promotions
        promotions = [
            -1,
            consts.POLICY_PROMOTION_VW + consts.POLICY_PROMOTION_BISHOP,
            consts.POLICY_PROMOTION_VW + consts.POLICY_PROMOTION_KNIGHT,
            consts.POLICY_PROMOTION_VW + consts.POLICY_PROMOTION_ROOK
        ]

        # NW
        to_square = chess.F8
        promotions[0] = consts.POLICY_NW_OFFSET

        test_promotion_dir(north_from_square, to_square,
                           promotions, expecteds, north_board)

        # SW
        to_square = chess.A1
        promotions[0] = consts.POLICY_SW_OFFSET

        test_promotion_dir(south_from_square, to_square,
                           promotions, expecteds, south_board)

        # Vertical promotions
        promotions = [
            -1,
            consts.POLICY_PROMOTION_V + consts.POLICY_PROMOTION_BISHOP,
            consts.POLICY_PROMOTION_V + consts.POLICY_PROMOTION_KNIGHT,
            consts.POLICY_PROMOTION_V + consts.POLICY_PROMOTION_ROOK
        ]

        # N
        to_square = chess.G8
        promotions[0] = consts.POLICY_N_OFFSET

        test_promotion_dir(north_from_square, to_square,
                           promotions, expecteds, north_board)

        # S
        to_square = chess.B1
        promotions[0] = consts.POLICY_S_OFFSET

        test_promotion_dir(south_from_square, to_square,
                           promotions, expecteds, south_board)

        # East promotions
        promotions = [
            -1,
            consts.POLICY_PROMOTION_VE + consts.POLICY_PROMOTION_BISHOP,
            consts.POLICY_PROMOTION_VE + consts.POLICY_PROMOTION_KNIGHT,
            consts.POLICY_PROMOTION_VE + consts.POLICY_PROMOTION_ROOK
        ]

        # NE
        to_square = chess.H8
        promotions[0] = consts.POLICY_NE_OFFSET

        test_promotion_dir(north_from_square, to_square,
                           promotions, expecteds, north_board)

        # SE
        to_square = chess.C1
        promotions[0] = consts.POLICY_SE_OFFSET

        test_promotion_dir(south_from_square, to_square,
                           promotions, expecteds, south_board)

        # Test not pawn
        north_board.set_piece_at(north_from_square, chess.Piece(chess.QUEEN, chess.WHITE))
        to_square = chess.H8
        indices[-1] = consts.POLICY_N_OFFSET

        expected = None
        actual = get_promotion(north_from_square, to_square, indices, north_board)
        self.assertEqual(expected, actual)

        south_board.set_piece_at(south_from_square, chess.Piece(chess.QUEEN, chess.BLACK))
        to_square = chess.C1
        indices[-1] = consts.POLICY_SE_OFFSET

        expected = None
        actual = get_promotion(south_from_square, to_square, indices, south_board)
        self.assertEqual(expected, actual)

        # Test not in right place
        # https://lichess.org/editor/8/8/6P1/2k5/8/5K2/8/8_w_-_-_0_1
        # Pawn at G6
        north_board = chess.Board('8/8/6P1/2k5/8/5K2/8/8 w - - 0 1')
        north_from_square = chess.G6
        to_square = chess.G7
        indices[-1] = consts.POLICY_N_OFFSET

        expected = None
        actual = get_promotion(north_from_square, to_square, indices, north_board)
        self.assertEqual(expected, actual)

        # https://lichess.org/editor/8/8/8/2k5/8/1p3K2/8/8_w_-_-_0_1
        # Pawn at B3
        south_board = chess.Board('8/8/8/2k5/8/1p3K2/8/8 w - - 0 1')
        south_from_square = chess.B3
        to_square = chess.B2
        indices[-1] = consts.POLICY_S_OFFSET

        expected = None
        actual = get_promotion(south_from_square, to_square, indices, south_board)
        self.assertEqual(expected, actual)

    def test_move_from_indices(self):
        # Knight move
        # https://lichess.org/editor/8/8/8/2k5/5N2/5K2/8/8_w_-_-_0_1
        board = chess.Board('8/8/8/2k5/5N2/5K2/8/8 w - - 0 1')
        indices = [*square_to_n_n(chess.F4), consts.POLICY_KNIGHT_LEFT_UP]

        expected = chess.Move.from_uci('f4d5')
        actual = move_from_indices(indices, board)
        self.assertEqual(expected, actual)

        # Queen move diagonal
        # https://lichess.org/editor/8/8/8/2k5/q7/5K2/8/8_b_-_-_0_1
        board = chess.Board('8/8/8/2k5/q7/5K2/8/8 b - - 0 1')
        indices = [*square_to_n_n(chess.A4), consts.POLICY_SE_OFFSET + 2]

        expected = chess.Move.from_uci('a4d1')
        actual = move_from_indices(indices, board)
        self.assertEqual(expected, actual)

        # Rook move
        # https://lichess.org/editor/8/8/8/2k5/8/5K2/2r5/8_b_-_-_0_1
        board = chess.Board('8/8/8/2k5/8/5K2/2r5/8 b - - 0 1')
        indices = [*square_to_n_n(chess.C2), consts.POLICY_S_OFFSET]

        expected = chess.Move.from_uci('c2c1')
        actual = move_from_indices(indices, board)
        self.assertEqual(expected, actual)

        # Queen promotion
        # https://lichess.org/editor/8/5P2/8/2k5/8/5K2/8/8_w_-_-_0_1
        board = chess.Board('8/5P2/8/2k5/8/5K2/8/8 w - - 0 1')
        indices = [*square_to_n_n(chess.F7), consts.POLICY_N_OFFSET]

        expected = chess.Move.from_uci('f7f8q')
        actual = move_from_indices(indices, board)
        self.assertEqual(expected, actual)

        # Knight underpromotion
        # https://lichess.org/editor/8/8/8/2k5/8/8/3p1K2/8_b_-_-_0_1
        board = chess.Board('8/8/8/2k5/8/8/3p1K2/8 b - - 0 1')
        indices = [*square_to_n_n(chess.D2), consts.POLICY_PROMOTION_V]

        expected = chess.Move.from_uci('d2d1n')
        actual = move_from_indices(indices, board)
        self.assertEqual(expected, actual)

    def test_sample_policy(self):
        # https://lichess.org/editor/8/8/8/2k5/8/8/5K2/8_b_-_-_0_1
        # Only one move with probability mass
        board = chess.Board('8/8/8/2k5/8/8/5K2/8 b - - 0 1')
        torch.manual_seed(42)
        masked_policy = torch.zeros(8,8,73)

        row, col = square_to_n_n(chess.C5)
        masked_policy[row, col, consts.POLICY_SE_OFFSET] = 1

        expected = chess.Move.from_uci('c5d4')
        actual = sample_policy(masked_policy, board)
        self.assertEqual(expected, actual)

        # Two moves with probability mass
        masked_policy[row, col, consts.POLICY_SE_OFFSET] = .25
        masked_policy[row, col, consts.POLICY_S_OFFSET] = .75

        expected = chess.Move.from_uci('c5c4')
        actual = sample_policy(masked_policy, board)
        self.assertEqual(expected, actual)
