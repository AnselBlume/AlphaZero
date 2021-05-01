import unittest
import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
from mask_policy import *
import chess
import torch
import constants as consts
import utils

class TestMaskPolicy(unittest.TestCase):
    def test_build_legal_move_dict(self):
        # https://lichess.org/editor/8/2k5/8/8/8/8/1K3p2/8_b_-_-_0_1
        board = chess.Board('8/2k5/8/8/8/8/1K3p2/8 b - - 0 1')
        move_dict = build_legal_move_dict(board)
        expected = {
            chess.C7: { # King moves
                chess.B8: {None},
                chess.C8: {None},
                chess.D8: {None},
                chess.D7: {None},
                chess.D6: {None},
                chess.C6: {None},
                chess.B6: {None},
                chess.B7: {None}
            },
            chess.F2: { # Pawn moves
                chess.F1: {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
            }
        }

        self.assertEqual(expected, move_dict)

    def test_square_move_to_index(self):
        # Move along rank
        # Move right
        from_square = chess.A6
        to_square = chess.G6

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_E_OFFSET + 5

        self.assertEqual(index, expected)

        # Move left
        from_square = chess.G6
        to_square = chess.A6

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_W_OFFSET + 5

        self.assertEqual(index, expected)

        # Move along file
        # Move up
        from_square = chess.B3
        to_square = chess.B6

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_N_OFFSET + 2

        self.assertEqual(index, expected)

        # Move down
        from_square = chess.B3
        to_square = chess.B2

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_S_OFFSET

        self.assertEqual(index, expected)

        # Move along diagonals
        # NE
        from_square = chess.C2
        to_square = chess.E4

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_NE_OFFSET + 1

        self.assertEqual(index, expected)

        # SE
        from_square = chess.D8
        to_square = chess.G5

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_SE_OFFSET + 2

        self.assertEqual(index, expected)

        # SW
        from_square = chess.H8
        to_square = chess.A1

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_SW_OFFSET + 6

        self.assertEqual(index, expected)

        # NW
        from_square = chess.F3
        to_square = chess.B7

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_NW_OFFSET + 3

        self.assertEqual(index, expected)

        # Knight move
        from_square = chess.F2
        to_square = chess.E4

        index = square_move_to_index(from_square, to_square)
        expected = consts.POLICY_KNIGHT_UP_LEFT

        self.assertEqual(index, expected)

        # Promotions
        # Queen
        from_square = chess.G7
        to_square = chess.H8
        promotion = chess.QUEEN

        index = square_move_to_index(from_square, to_square, promotion)
        expected = consts.POLICY_NE_OFFSET

        self.assertEqual(index, expected)

        # Rook
        from_square = chess.E2
        to_square = chess.D1
        promotion = chess.ROOK

        index = square_move_to_index(from_square, to_square, promotion)
        expected = consts.POLICY_PROMOTION_VW + consts.POLICY_PROMOTION_ROOK

        self.assertEqual(index, expected)

    def test_mask_position(self):
        policy = torch.ones(8, 8, 73)
        # Alekhine's defense https://lichess.org/editor/rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/5N2/PPP2PPP/RNBQKB1R_b_KQkq_-_1_4
        board = chess.Board('rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 1 4')

        # Mask the black knight's moves
        legal_move_dict = build_legal_move_dict(board)
        knight_moves = legal_move_dict[chess.D5]
        legal_indices = {square_move_to_index(chess.D5, to_square) for to_square in knight_moves}
        expected = torch.tensor([1 if i in legal_indices else 0 for i in range(73)])

        row, col = utils.square_to_n_n(chess.D5)
        mask_position(row, col, policy, legal_move_dict)

        self.assertTrue((policy[row,col] == expected).all())

    def test_mask_invalid_moves(self):
        # https://lichess.org/editor/8/3k4/8/8/8/4Q3/2K5/8_w_-_-_0_1
        board = chess.Board('8/3k4/8/8/8/4Q3/2K5/8 w - - 0 1')

        # Mask the white King and Queen's moves
        legal_move_dict = build_legal_move_dict(board)

        queen_moves = legal_move_dict[chess.E3]
        queen_indices = {square_move_to_index(chess.E3, to_square) for to_square in queen_moves}
        queen_expected = torch.tensor([1 if i in queen_indices else 0 for i in range(73)])

        king_moves = legal_move_dict[chess.C2]
        king_indices = {square_move_to_index(chess.C2, to_square) for to_square in king_moves}
        king_expected = torch.tensor([1 if i in king_indices else 0 for i in range(73)])

        expected = torch.zeros(8, 8, 73)
        expected[utils.square_to_n_n(chess.E3)] = queen_expected
        expected[utils.square_to_n_n(chess.C2)] = king_expected
        expected /= expected.sum()

        policy = torch.ones(8, 8, 73)
        mask_invalid_moves(policy, board)

        self.assertEqual(policy.sum(), 1)
        self.assertTrue((expected == policy).all())
