from .encode_dist import MCTSPolicyEncoder
from .encode_state import StateEncoder
import chess
import torch.nn.functional as F
import torch

class MCTSLoss:
    def __init__(self, T, device='cpu'):
        self.mcts_policy_encoder = MCTSPolicyEncoder()
        self.state_encoder = StateEncoder(T)
        self.device = device

    def get_loss(self, network, mcts_dist_histories):
        '''
            network is the neural network whose loss will be computed.

            mcts_dist_histories is a list of lists of MCTSDist objects obtained from
            running mcts a list of (mcts_dist_history). Each list in mcts_dist_histories contains the history MCTSDist
            states leading up to the current state. Each list will have length <= T.

            Each mcts_dist_history list represents a single move: the history
            is for input to the neural network, with the final MCTSDist representing
            the final state in which the move probabilities were computed by MCTS.
        '''
        # Compute network outputs
        encodings = [
            self._build_final_state_encoding(mcts_dists)
            for mcts_dists in mcts_dist_histories
        ]
        encodings = torch.stack(encodings, dim=0).to(self.device)
        net_vals, net_log_probs = network(encodings)

        # Compute mcts outputs
        mcts_vals = []
        mcts_policies = []
        for mcts_dists in mcts_dist_histories:
            # Compute MCTS value, outputs for state
            final_state = mcts_dists[-1]
            mcts_val = final_state.value
            mcts_policy = self.mcts_policy_encoder.get_mcts_policy(final_state)

            mcts_vals.append(mcts_val)
            mcts_policies.append(mcts_policy)

        mcts_vals = torch.tensor(mcts_vals).reshape(-1, 1).to(self.device)
        mcts_policies = torch.stack(mcts_policies, dim=0).to(self.device)

        assert mcts_vals.shape == net_vals.shape
        assert mcts_policies.shape == net_log_probs.shape

        mse = F.mse_loss(net_vals, mcts_vals)
        ce = -(mcts_policies * net_log_probs).sum() / len(mcts_dist_histories)

        assert not torch.isnan(mse).any() and not torch.isinf(mse).any()
        assert not torch.isnan(ce).any() and not torch.isinf(ce).any()

        # Let the optimizer handle l2 regularization

        return mse + ce

    def _build_final_state_encoding(self, mcts_dists):
        '''
            Builds the final state encoding from a list of MCTSDists representing
            a series of states.
        '''
        boards = (chess.Board(mcts_dist.fen) for mcts_dist in mcts_dists)
        return self.state_encoder.encode_state_with_history(boards)
