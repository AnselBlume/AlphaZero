import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
from network.network import Network
from run.train import train
import coloredlogs
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

if __name__ == '__main__':
    device = 'cpu'
    T = 2
    max_time_s = 10
    start_fen = 'k7/8/1K6/2Q5/8/8/8/8 w - - 0 1'

    train(T, device=device, start_fen=start_fen, max_time_s=max_time_s)
