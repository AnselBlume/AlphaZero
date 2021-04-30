import unittest
import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
import encode_state
from encode_state import StateEncoder
import chess
import torch

class TestState(unittest.TestCase):
    def setUp(self):
        self.encoder = StateEncoder(T=2)

    def test_piece_to_index(self):
        # Test 1
        board = chess.Board() # color is True by default
        piece = chess.Piece(chess.ROOK, False)
        index = self.encoder.piece_to_index(piece, board)
        expected = 9

        self.assertEqual(index, expected)

        # Test 2
        board = chess.Board()
        piece = chess.Piece(chess.QUEEN, True)
        index = self.encoder.piece_to_index(piece, board)
        expected = 4

        self.assertEqual(index, expected)

        # Test 3
        board = chess.Board()
        board.turn = False
        piece = chess.Piece(chess.PAWN, True)
        index = self.encoder.piece_to_index(piece, board)
        expected = 6

        self.assertEqual(index, expected)

    def test_encode_state(self):
        M, N = encode_state.M, encode_state.N
        T = self.encoder.T

        # Test 1 (https://lichess.org/editor/8/8/8/1k6/2n5/5K2/2B5/8_w_-_-_0_1)
        board = chess.Board(fen='8/8/8/1k6/2n5/5K2/2B5/8 w - - 0 1')
        prev_state = self.encoder.get_empty_state()
        state = self.encoder.encode_state(board, prev_state)

        # Check history
        expected_history = 0
        history = state[:M*(T-1),...]
        self.assertTrue((history == expected_history).all())

        # Check position planes
        expected_white_pos = torch.zeros(6, N, N)
        expected_white_pos[2, 6, 2] = 1 # Bishop
        expected_white_pos[5, 5, 5] = 1 # King

        expected_black_pos = torch.zeros(6, N, N)
        expected_black_pos[5, 3, 1] = 1 # King
        expected_black_pos[1, 4, 2] = 1 # Knight

        player_pos = state[M*(T-1):M*(T-1)+6, ...]
        opponent_pos = state[M*(T-1)+6:M*(T-1)+12, ...]

        self.assertTrue((player_pos == expected_white_pos).all())
        self.assertTrue((opponent_pos == expected_black_pos).all())

        # Check L planes
        player_color = state[M*T, ...]
        expected_color = 1
        self.assertTrue((player_color == expected_color).all())

        move_count = state[M*T+1, ...]
        expected_count = 0
        self.assertTrue((move_count == expected_count).all())

        player_castling = state[M*T+2:M*T+4, ...]
        expected_castling = 0
        self.assertTrue((player_castling == expected_castling).all())

        opponent_castling = state[M*T+4:M*T+6, ...]
        expected_castling = 0
        self.assertTrue((opponent_castling == expected_castling).all())

        # Take a move (https://lichess.org/editor/8/8/8/1k6/2n5/8/2B1K3/8_w_-_-_0_1)
        ke2 = chess.Move.from_uci('f3e2')
        board.push(ke2)
        prev_state = state
        state = self.encoder.encode_state(board, prev_state)

        # Check history
        history = state[:M*(T-1), ...]
        expected_history = prev_state[M:M*T, ...]
        self.assertTrue((history == expected_history).all())

        # Check positions
        expected_white_pos[5,...] = 0 # White king moved; need to adjust expected
        expected_white_pos[5, 6, 4] = 1

        player_pos = state[M*(T-1):M*(T-1)+6, ...]
        opponent_pos = state[M*(T-1)+6:M*(T-1)+12,...]

        self.assertTrue((player_pos == expected_black_pos).all()) # Reverse of before
        self.assertTrue((opponent_pos == expected_white_pos).all())

        # Check current player color
        player_color = state[M*T, ...]
        expected_color = 0
        self.assertTrue((player_color == expected_color).all())

        # Check move count
        move_count = state[M*T+1, ...]
        expected_count = 1
        self.assertTrue((move_count == expected_count).all())
