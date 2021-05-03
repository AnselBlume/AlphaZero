from .encode_dist import MCTSPolicyEncoder
from .encode_state import StateEncoder
from .network import to_probabilities
import chess
import torch.nn.functional as F

class MCTSLoss:
    def __init__(self, T, temp=2):
        self.temp = 2
        self.mcts_policy_encoder = MCTSPolicyEncoder(temp=temp)
        self.state_encoder = StateEncoder(T)

    def get_loss(network, mcts_dist_histories):
        '''
            network is the neural network whose loss will be computed.

            mcts_dist_histories is a list of lists of MCTSDist objects obtained from
            running mcts. Each list in mcts_dist_histories contains the history MCTSDist
            states leading up to the current state. Each list will have length T.
        '''
        # Compute network outputs
        encodings = [
            self._build_final_state_encoding(mcts_dists)
            for mcts_dists in mcts_dist_histories
        ]
        encodings = torch.stack(encodings, dim=0)
        net_vals, net_policies = network(encodings)
        net_policies = to_probabilities(net_policies)

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

        mcts_vals = torch.tensor(mcts_vals)
        mcts_policies = torch.stack(mcts_policies, dim=0)

        assert mcts_vals.shape == net_vals.shape
        assert mcts_policies.shape == net_policies.shape

        mse = F.mse_loss(net_vals, mcts_vals)
        ce = (mcts_policies * net_policies.log()).sum() / len(mcts_dist_histories)

        # Let the optimizer handle l2 regularization

        return mse + ce

    def _build_final_state_encoding(self, mcts_dists):
        '''
            Builds the final state encoding from a list of MCTSDists representing
            a series of states.
        '''
        boards = (chess.Board(mcts_dist.fen) for mcts_dist in mcts_dists)
        return self.state_encoder.encode_state_with_history(boards)
