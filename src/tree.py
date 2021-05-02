import chess
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TreeNode:
    def __init__(self, fen, fen_to_node, is_rollout=False):
        '''
            fen_to_node is a dictionary mapping FENs to TreeNodes so that
            we can reuse TreeNode statistics if multiple paths lead to the same
            TreeNode state (as the graph is a DAG, not a directed tree).

            If is_rollout, that means this node was created during a rollout and
            shouldn't be added to the fen index (as it is temporary). All expanded
            children will also have is_rollout set to true and not be added to the dict.
        '''
        self.fen = fen
        self.out_edges = []
        self.outcome = chess.Board(self.fen).outcome()

        # Add self to node index if not created from a rollout
        if not is_rollout:
            fen_to_node[fen] = self
        self.is_rollout = is_rollout
        self.fen_to_node = fen_to_node

        # Not technically needed but nice to compute state value
        self.n_visits = 0
        self.score = 0

    def is_terminal(self):
        return self.outcome is not None

    def was_expanded(self):
        return self.is_terminal() and self.is_leaf() \
               or not self.is_terminal() and not self.is_leaf()

    def expand(self, prior_func=None, is_rollout=False):
        '''
            prior_func should take a chess.Move and return the prior probability
            for the move.

            If prior_func is None or is_rollout the prior will be set to zero (for faster
            evaluation during rollouts which don't compute the UCB).

            If is_rollout, nodes created during the expansion won't be added to
            the fen dictionary and no nodes will be retrieved from the fen dictionary
            to prevent modification of the tree.
        '''
        assert not self.was_expanded()

        # If this node was created in a rollout, doesn't matter what the user inputs
        # for is_rollout; we're still in a rollout
        is_rollout = is_rollout or self.is_rollout

        # No need to compute prior if in a rollout
        if prior_func is None or is_rollout:
            prior_func = lambda x: 0

        # For each action, create a child node
        board = chess.Board(self.fen)
        for move in board.legal_moves:
            # Compute the next state from the move
            board.push(move)
            child_fen = board.fen()

            # If we're in a rollout,
            if not is_rollout and child_fen in self.fen_to_node:
                child_state = self.fen_to_node[child_fen]
            else:
                child_state = TreeNode(child_fen, self.fen_to_node, is_rollout=is_rollout)

            # Add an edge to the new state to this node
            prior = prior_func(move)
            out_edge = TreeEdge(
                move.uci(),
                prior,
                self,
                child_state
            )
            self.out_edges.append(out_edge)

            board.pop() # Reset state to expand more moves

        assert len(self.out_edges) > 0

    def unexpand(self):
        self.out_edges = []

    def is_leaf(self):
        return self.out_edges == []

    def backup_update(self, score):
        self.score += score
        self.n_visits += 1

    def get_edge_to_explore(self, trials_so_far=None):
        '''
            If trials_so_far is not None, uses the standard sqrt(ln(...))
            and multiplies it by the AlphaGo prior.
        '''
        if self.is_terminal():
            return None

        if self.out_edges == []:
            raise RuntimeError('This node has not yet been expanded')

        # Don't compute for log message unless necessary
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(list(map(lambda e: f'{e.uci}: {e.get_ucb(trials_so_far)}', self.out_edges)))

        best_edge = self.out_edges[0]
        best_ucb = best_edge.get_ucb(trials_so_far)

        for edge in self.out_edges[1:]:
            curr_ucb = edge.get_ucb(trials_so_far)
            if curr_ucb > best_ucb:
                best_edge = edge
                best_ucb = curr_ucb

        return best_edge

    def get_random_edge(self):
        if self.is_terminal():
            return None

        if self.out_edges == []:
            raise RuntimeError('This node has not yet been expanded')

        rand_ind = int(np.random.rand() * len(self.out_edges))
        return self.out_edges[rand_ind]

    def get_best_move(self):
        if self.is_terminal():
            return None

        if self.out_edges == []:
            raise RuntimeError('This node has not yet been expanded')

        best_edge = self.out_edges[0]
        best_q_val = best_edge.get_action_value()

        for edge in self.out_edges[1:]:
            curr_q_val = edge.get_action_value()
            if curr_q_val > best_q_val:
                best_edge = edge
                best_q_val = curr_q_val

        return best_edge

    def get_state_value(self):
        return self.score / self.n_visits if self.n_visits > 0 else 0

    def __str__(self):
        return (
            f'FEN: {self.fen}\n'
            + f'Score: {self.score}\n'
            + f'Visits: {self.n_visits}\n'
            + f'State value: {self.get_state_value()}\n'
            + f'Best move: {self.get_best_move().uci}\n'
        )

    def __repr__(self):
        return str(self)

class TreeEdge:
    UCB_FACTOR = 5

    def __init__(self, uci, prior, from_node, to_node):
        self.uci = uci
        self.prior = prior
        self.score = 0
        self.n_visits = 0

        self.from_node = from_node
        self.to_node = to_node

    def get_ucb(self, trials_so_far=None):
        q = self.get_action_value()

        # Use default AlphaGo
        if trials_so_far is None:
            u = 1 / (1 + self.n_visits)
        else: # Or standard UCB
            u = np.sqrt(np.log(1 + trials_so_far) / (1 + self.n_visits))
        u *= TreeEdge.UCB_FACTOR * self.prior

        return q + u

    def backup_update(self, score):
        self.score += score
        self.n_visits += 1

    def get_action_value(self):
        return self.score / self.n_visits if self.n_visits > 0 else 0

    def __str__(self):
        return (
            f'UCI: {self.uci}\n'
            + f'Prior: {self.prior}\n'
            + f'Score: {self.score}\n'
            + f'Visits: {self.n_visits}\n'
            + f'Action value: {self.get_action_value()}\n'
        )

    def __repr__(self):
        return str(self)
