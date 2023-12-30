import torch

NUM_EPISODES = 10      #Number of episodes
LR = 0.0005             #Learning Rate of the Neural Network

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"        #Choose the Device

#Checkpoints managenent
CHECKPOINT_FOLDER = "Checkpoints/Checkpoint400"            #This is the file that contains the weights to use in testing

#Function that loades pre-trained weights
def load_model(file, model, optimizer):
    model_check = torch.load(file, map_location=DEVICE)
    model.load_state_dict(model_check["state_dict"])
    optimizer.load_state_dict(model_check["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = LR
