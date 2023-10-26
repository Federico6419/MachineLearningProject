import torch

NUM_EPISODES = 100      #Number of episodes
LR = 0.0005      #Learning Rate of the Neural Network

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Manage Checkpoints
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
