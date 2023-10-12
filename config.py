import torch

NUM_EPISODES = 1000      #Number of episodes
MAX_EPSILON = 1      #Initial exploration probability
MIN_EPSILON = 0.01      #Final exploration probability
ALPHA = 0.001      #Learning rate for the Q-Table
EPSILON_DECAY = 0.99
LR = 0.0005      #Learning Rate of the Neural Network
BUFFER_SIZE = 100000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
