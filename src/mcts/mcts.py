import chess
from .tree import TreeNode
import logging
from time import time
from numpy.random import randint
import ctypes
import glob
import os

logger = logging.getLogger(__name__)

USE_CPP_ROLLOUT = True

if USE_CPP_ROLLOUT:
    # Note that logs and prints don't occur during import on Google Colab

    # Hack to set up the C++ rollout library on any system
    # Refer to test/test_lib.py for the code
    orig_wd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(dir_path, 'c_rollout'))

    # Google Colab is okay with the script version for some reason
    # os.system returns the exit code; exit code != 0 implies an error occurred
    if not os.path.exists('./build'):
        if os.system('chmod u+x ./build_lib.sh && ./build_lib.sh'):
            raise RuntimeError('A problem occurred building the C++ rollout library')
        logger.info('C++ rollout library built successfully')

    libfile = glob.glob('build/*/*.so')[0]
    rollout_lib = ctypes.CDLL(libfile)

    '''
    # Google Colab doesn't seem to like the g++ version
    LIB_FILE = 'rollout.so'

    if not os.path.exists(LIB_FILE):
        if os.system(f'g++ -O3 -o {LIB_FILE} -w rollout.cpp thc.cpp'):
            raise RuntimeError('A problem occurred building the C++ rollout library')
        logger.info('C++ rollout library built successfully')

    rollout_lib = ctypes.CDLL(LIB_FILE)
    '''
    rollout_lib.rollout.restype = ctypes.c_int
    rollout_lib.rollout.argtypes = [ctypes.c_char_p]

    os.chdir(orig_wd)

# Fantastic explanation of MCTS:
# https://www.youtube.com/watch?v=UXW2yZndl7U

class MCTSEvaluator:
    def __init__(self, root_fen, prior_func_builder, subtree=None):
        # TODO take in fen history of root_fen to pass into prior_func_builder
        self.root_fen = root_fen
        self.curr_player = chess.Board(root_fen).turn # For terminal state eval
        self.prior_func_builder = prior_func_builder
        self.subtree = subtree # Whether to initialize the MCTS tree to this tree

    def mcts(self, std_ucb=False, max_trials=500, max_time_s=float('inf')):
        '''
            max_time_s < 0 indicates all the time necessary to finish
            max_trials.
        '''
        logger.debug(f'Starting MCTS in state:\n{chess.Board(self.root_fen)}')
        fen_to_node = {} # Index of all TreeNodes
        if self.subtree is None:
            root = TreeNode(self.root_fen, fen_to_node)
            root.expand(self.prior_func_builder([self.root_fen]))
        else:
            root = self.subtree
            assert root.was_expanded()

        start_time = time()
        out_of_time = False

        i = 0
        while i < max_trials and not out_of_time:
            self.evaluate(root, fen_to_node, i, std_ucb=std_ucb)
            i += 1

            if max_time_s > 0 and time() - start_time > max_time_s:
                out_of_time = True
                logger.debug(f'Out of time for MCTS evaluation')

        logger.debug(f'Completed {i} rounds of MCTS')

        return root

    def evaluate(self, root, fen_to_node, trials_so_far, std_ucb=False):
        # Path will contain nodes and edges so we can update state values
        # though we technically only care about action values
        path = [root]
        fen_path = [root.fen] # Keep track of just node path to compute priors

        # Descend until we find a leaf (a node which hasn't been expanded)
        curr = root
        while not curr.is_leaf():
            edge = curr.get_edge_to_explore(trials_so_far, std_ucb=std_ucb)
            curr = edge.to_node

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

                path.append(edge)
                path.append(curr)
                fen_path.append(curr)

            if USE_CPP_ROLLOUT:
                value = rollout_lib.rollout(curr.fen.encode('ascii'))
            else:
                value = self.rollout_fast(curr)

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

    def rollout_fast(self, to_rollout):
        '''
            Much faster rollout version that doesn't construct the tree, since
            we're trimming it off anyways.

            Just simulates with chess.Board
        '''
        board = chess.Board(to_rollout.fen)
        curr_player = board.turn
        outcome = board.outcome()

        while outcome is None:
            moves = list(board.legal_moves)
            rand_move = moves[randint(0, len(moves))]
            board.push(rand_move)
            outcome = board.outcome()

        winner = outcome.winner
        if winner is None:
            return 0

        return 1 if curr_player == winner else -1

    def get_terminal_value(self, terminal):
        winner = terminal.outcome.winner
        if winner is None:
            return 0

        return 1 if self.curr_player == winner else -1
