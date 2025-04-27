import torchvision.transforms as transforms
import torch
from torch import nn
transform = transforms.Compose([transforms.Resize((256,256)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5,0.5,0.5])
                               ])

transform1 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

batch_size = 16
epochs = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam