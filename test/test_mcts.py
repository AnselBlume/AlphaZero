import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
import chess
from mcts.mcts import MCTSEvaluator
import coloredlogs

#coloredlogs.DEFAULT_LEVEL_STYLES['info'] = 'blue'
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

if __name__ == '__main__':

    prior_func_builder = lambda l: lambda m: .5

    # Black checkmates white on white turn; should return -1
    # https://lichess.org/editor/8/8/8/8/8/8/5kq1/7K_w_-_-_0_1
    #fen = '8/8/8/8/8/8/5kq1/7K w - - 0 1'
    #eval = MCTSEvaluator(fen, prior_func_builder)
    #root = eval.mcts()

    # Easy black checkmate on black turn; should return close to 1
    # https://lichess.org/editor/8/8/8/8/8/6p1/5kr1/7K_b_-_-_0_1
    fen = '8/8/8/8/8/6p1/5kr1/7K b - - 0 1'
    eval = MCTSEvaluator(fen, prior_func_builder)
    root = eval.mcts(max_trials=50)

    # Endgames https://lichess.org/study/aHKg4c4e
    # https://lichess.org/editor/8/8/p5r1/1p6/1P1R4/8/5K1p/7k_w_-_-_0_1
    '''
    fen = '8/8/p5r1/1p6/1P1R4/8/5K1p/7k w - - 0 1'
    eval = MCTSEvaluator(fen, prior_func_builder)
    root = eval.mcts(max_trials=1000)
    '''

    # Starting position
    '''
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    eval = MCTSEvaluator(fen, prior_func_builder)
    root = eval.mcts(max_trials=2)
    '''

    # White wins in four moves
    # https://lichess.org/editor/k5r1/pp6/8/1N2Q3/8/8/8/3K4_w_-_-_0_1
    '''
    fen = 'k5r1/pp6/8/1N2Q3/8/8/8/3K4 w - - 0 1'
    eval = MCTSEvaluator(fen, prior_func_builder)
    root = eval.mcts(max_trials=2000)
    '''

    # White wins in two moves
    # https://lichess.org/editor/k7/8/1K2p3/8/2Q5/8/8/8_w_-_-_0_1
    fen = 'k6r/4p3/1K6/2Q5/8/8/8/8 w - - 0 1'
    eval = MCTSEvaluator(fen, prior_func_builder)
    root = eval.mcts(max_trials=800)

# exec(open('test_mcts.py').read())
