import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
import chess
from mcts import MCTSEvaluator
import coloredlogs

#coloredlogs.DEFAULT_LEVEL_STYLES['info'] = 'blue'
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

if __name__ == '__main__':

    prior_func = lambda x: .5

    # Black checkmates white on white turn; should return -1
    # https://lichess.org/editor/8/8/8/8/8/8/5kq1/7K_w_-_-_0_1
    #board_fen = '8/8/8/8/8/8/5kq1/7K w - - 0 1'
    #eval = MCTSEvaluator(board_fen, prior_func)
    #root = eval.mcts()

    # Easy black checkmate on black turn; should return close to 1
    # https://lichess.org/editor/8/8/8/8/8/6p1/5kr1/7K_b_-_-_0_1
    board_fen = '8/8/8/8/8/6p1/5kr1/7K b - - 0 1'
    eval = MCTSEvaluator(board_fen, prior_func)
    root = eval.mcts(trials=50)

    # Endgames https://lichess.org/study/aHKg4c4e
    # https://lichess.org/editor/8/8/p5r1/1p6/1P1R4/8/5K1p/7k_w_-_-_0_1
    board_fen = '8/8/p5r1/1p6/1P1R4/8/5K1p/7k w - - 0 1'
    eval = MCTSEvaluator(board_fen, prior_func)
    root = eval.mcts(trials=500)

    # Starting position
    '''
    board_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    eval = MCTSEvaluator(board_fen, prior_func)
    root = eval.mcts(trials=2)
    '''

# exec(open('test_mcts.py').read())
