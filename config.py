import torch

NUM_EPISODES = 1000      #Number of episodes
MAX_EPSILON = 1      #Initial exploration probability
MIN_EPSILON = 0.01      #Final exploration probability
ALPHA = 0.001      #Learning rate for the Q-Table
EPSILON_DECAY = 0.99
LR = 0.0005      #Learning Rate of the Neural Network
BUFFER_SIZE = 100000
BATCH_SIZE = 64
TARGET_FREQ_UPDATE = 10

USE_QTABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Manage Checkpoints
def save_model(model, optimizer, episode):  
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CHECKPOINT)

def load_model(file, model, optimizer):
    model_check = torch.load(file, map_location=DEVICE)
    model.load_state_dict(model_check["state_dict"])
    optimizer.load_state_dict(model_check["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = LR
