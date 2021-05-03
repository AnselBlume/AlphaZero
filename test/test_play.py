import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
from play import GameRunner
from network.network import Network
import coloredlogs
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

if __name__ == '__main__':
    device = 'cpu'
    net = Network(1).to(device)
    game_runner = GameRunner(1, device=device)
    board, mcts_dists_histories = game_runner.play_game(net)
