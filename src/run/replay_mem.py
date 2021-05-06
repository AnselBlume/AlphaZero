import torch
from collections import deque

class ReplayMemory:
    def __init__(self, last_k=10):
        '''
            last_k is the number of games to save in the memory.
        '''
        self.last_k = last_k
        self.memory = deque()

    def save(self, mcts_dist_histories):
        if len(self.memory) == self.last_k:
            self.memory.popleft()

        self.memory.append(mcts_dist_histories)

    def sample(self, num_to_sample=50):
        '''
            Returns a list of mcts_dist_history lists of length num_to_sample.
        '''
        # Flatten the memory of all previous games
        flattened = [
            mcts_dist_history for mcts_dist_histories in self.memory
            for mcts_dist_history in mcts_dist_histories
        ]
        probs = torch.ones(len(flattened))
        probs /= probs.sum() # Uniform
        samples = torch.multinomial(probs, num_to_sample, replacement=True)

        return [flattened[i] for i in samples]
