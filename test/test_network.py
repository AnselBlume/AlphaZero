import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
import chess
import unittest
from network.network import Network
from network.encode_state import StateEncoder, M, L
import torch

class TestNetwork(unittest.TestCase):
    def test_outputs_correct_shape(self):
        def test_correct_shape(T):
            encoder = StateEncoder(T)
            prev_state = encoder.get_empty_state()
            curr_state = encoder.encode_state(chess.Board(), prev_state)
            curr_state = curr_state.unsqueeze(0)

            net = Network(M*T + L)
            out = net(curr_state)

            self.assertEqual(out[0].shape, torch.Size([1]))
            self.assertEqual(out[1].shape, torch.Size((1, 8,8,73)))

        test_correct_shape(1)
        test_correct_shape(2)
