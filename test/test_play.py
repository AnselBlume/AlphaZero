import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
from play import GameRunner
from network.network import Network
import coloredlogs
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s %(levelname)s %(message)s')
from network.encode_state import M, L

if __name__ == '__main__':
    net = Network(1)
    game_runner = GameRunner(1)
    game_runner.play_game(net)
