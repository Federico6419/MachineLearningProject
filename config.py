import torch

NUM_EPISODES = 500      #Number of episodes
MAX_EPSILON = 1         #Initial exploration probability
MIN_EPSILON = 0.1       #Final exploration probability    
ALPHA = 0.001           #Learning rate for the Q-Table
EPSILON_DECAY = 0.99    #It is also possible to use 0.9999 
GAMMA = 0.99            #Discount factor (It is also possible to use 0.95)   
LR = 0.0005             #Learning Rate of the Neural Network (#It is also possible to use 0.001)
BUFFER_SIZE = 5000      
BATCH_SIZE = 64         #Batch Size
TARGET_FREQ_UPDATE = 5

USE_QTABLE = True       #Decide if using Q-Table or Neural Network for training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"    #Choose the Device

#Checkpoints managenent
#Load pre-trained weights
LOAD_CHECKPOINT = False           #If it is true pre-trained weights will be loaded
LOAD_CHECKPOINT_FOLDER = "Checkpoints/CheckpointOpt200"        #Checkpoints file
LOADED_EPSILON = 0.13263987810938213            #This is the current Epsilon of the loaded checkpoints

#Save Checkpoints
CHECKPOINT_FOLDER = "Checkpoints/checkpoint.pth.tar"            #This is the file in which the weights will be saved

#Function that saves current weights
def save_model(model, optimizer, episode):  
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CHECKPOINT_FOLDER)

#Function that loades pre-trained weights
def load_model(file, model, optimizer):
    model_check = torch.load(file, map_location=DEVICE)
    model.load_state_dict(model_check["state_dict"])
    optimizer.load_state_dict(model_check["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = LR
