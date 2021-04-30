
import unittest
import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
import chess
import torch
from utils import *

class TestUtils(unittest.TestCase):
    def test_square_to_n_n(self):
        # Test 1
        square = chess.parse_square('a1')
        n_n = square_to_n_n(square)
        expected = (7, 0)

        self.assertEqual(n_n, expected)

        # Test 2
        square = chess.parse_square('g3')
        n_n = square_to_n_n(square)
        expected = (5, 6)

        self.assertEqual(n_n, expected)

    def test_n_n_to_square(self):
        # Test 1
        row, col = 7, 7
        square = n_n_to_square(row, col)
        expected = chess.parse_square('h1')

        self.assertEqual(square, expected)

        # Test 2
        row, col = 4, 5
        square = n_n_to_square(row, col)
        expected = chess.parse_square('f4')

        self.assertEqual(square, expected)
