import os
import sys
sys.path.insert(1, os.path.realpath('../src'))
import torch
from run.train import load_state, MODEL_KEY

if __name__ == '__main__':
    T = 8
    checkpoint_path = '../checkpoints/latest_chkpt.tar'
    state_dict_path = '../checkpoints/net_dict.tar'

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    torch.save(checkpoint[MODEL_KEY], state_dict_path)
