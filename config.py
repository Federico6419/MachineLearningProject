import torch

NUM_EPISODES = 500      #Number of episodes
MAX_EPSILON = 1      #Initial exploration probability
MIN_EPSILON = 0.1            #Prima era 0.01      #Final exploration probability
ALPHA = 0.001      #Learning rate for the Q-Table
EPSILON_DECAY = 0.99        #Si può provare anche 0.9999 
GAMMA = 0.99            #Si può provare anche 0.95     #Discount factor
LR = 0.0005     #Si può provare anche 0.001 #Learning Rate of the Neural Network
BUFFER_SIZE = 5000 #100000
BATCH_SIZE = 64
TARGET_FREQ_UPDATE = 5

USE_QTABLE = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Manage Checkpoints
#Load
LOAD_CHECKPOINT = False
LOAD_CHECKPOINT_FOLDER = "Checkpoints/CheckpointOpt200"
LOADED_EPSILON = 0.13263987810938213

#Save Checkpoints
CHECKPOINT_FOLDER = "Checkpoints/checkpoint.pth.tar"

def save_model(model, optimizer, episode):  
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CHECKPOINT_FOLDER)

def load_model(file, model, optimizer):
    model_check = torch.load(file, map_location=DEVICE)
    model.load_state_dict(model_check["state_dict"])
    optimizer.load_state_dict(model_check["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = LR
