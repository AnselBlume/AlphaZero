import chess
from .tree import TreeNode
import logging
from time import time

logger = logging.getLogger(__name__)

# Fantastic explanation of MCTS:
# https://www.youtube.com/watch?v=UXW2yZndl7U

class MCTSEvaluator:
    def __init__(self, root_fen, prior_func_builder):
        # TODO take in fen history of root_fen to pass into prior_func_builder
        self.root_fen = root_fen
        self.curr_player = chess.Board(root_fen).turn # For terminal state eval
        self.prior_func_builder = prior_func_builder

    def mcts(self, std_ucb=True, max_trials=500, max_time_s=-1):
        '''
            max_time_s < 0 indicates all the time necessary to finish
            max_trials.
        '''
        logger.debug(f'Starting MCTS in state {self.root_fen}')
        fen_to_node = {} # Index of all TreeNodes
        root = TreeNode(self.root_fen, fen_to_node)
        root.expand(self.prior_func_builder([self.root_fen]))

        start_time = time()
        out_of_time = False

        i = 0
        while i < max_trials and not out_of_time:
            self.evaluate(root, fen_to_node, i if std_ucb else None)
            i += 1

            if max_time_s > 0 and time() - start_time > max_time_s:
                out_of_time = True
                logger.debug(f'Out of time for MCTS evaluation')

        logger.debug(f'Completed {i} rounds of MCTS')

        return root

    def evaluate(self, root, fen_to_node, trials_so_far=None):
        # Path will contain nodes and edges so we can update state values
        # though we technically only care about action values
        path = [root]
        fen_path = [root.fen] # Keep track of just node path to compute priors

        # Descend until we find a leaf (a node which hasn't been expanded)
        curr = root
        while not curr.is_leaf():
            edge = curr.get_edge_to_explore(trials_so_far)
            curr = edge.to_node

            logger.debug(f'Exploring: {edge.uci}')

            path.append(edge)
            path.append(curr)
            fen_path.append(curr.fen)

        if curr.is_terminal():
            value = self.get_terminal_value(curr)
        else:
            if curr.n_visits > 0:
                curr.expand(self.prior_func_builder(fen_path))

                edge = curr.get_edge_to_explore(trials_so_far)
                curr = edge.to_node

                logger.debug(f'Exploring: {edge.uci}')

                path.append(edge)
                path.append(curr)
                fen_path.append(curr)

            value = self.rollout(curr)

        self.backup(path, value)

    def backup(self, path, value):
        for node_or_edge in path:
            node_or_edge.backup_update(value)

    def rollout(self, to_rollout):
        # Attempt to trim the rollout subtree at the highest possible point
        # May not be possible to trim as the rollout subtree could have already been
        # created as the MCTS is not a true tree (it's a DAG)
        to_unexpand = None

        curr = to_rollout
        while not curr.is_terminal():
            if not curr.was_expanded():
                # Find the highest node to trim the rollout subtree from
                if to_unexpand is None:
                    to_unexpand = curr

                curr.expand(is_rollout=True) # No need to compute prior for state value evaluation

            curr = curr.get_random_edge().to_node

        value = self.get_terminal_value(curr)

        # If the nodes were already expanded before the rollout, don't unexpand them
        # Otherwise, can let go of the rollout subtree
        if to_unexpand is not None:
            to_unexpand.unexpand()

        return value

    def get_terminal_value(self, terminal):
        winner = terminal.outcome.winner
        if winner is None:
            return 0

        return 1 if self.curr_player == winner else -1
