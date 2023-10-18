import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, stride=3),
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4),
    
        self.dense1 = nn.Linear(432, 216)
        self.dense2 = nn.Linear(216, 12)
    
        #self.flatten = torch.flatten(start_dim=2)
         
        self.maxpool = nn.MaxPool2d((2, 2))


    def forward(self, state):
        x0 = self.maxpool(F.relu(self.conv1(state)))
        x1 = self.maxpool(F.relu(self.conv2(x0)))
        x2 = torch.flatten(x1)
        x3 = F.relu(self.dense1(x2))
        x4 = self.dense2(x3)
        return x4 







    

    
