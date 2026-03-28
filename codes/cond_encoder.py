import torch
from torch import nn

from tqdm import tqdm
import random

from dataloader import Loaders

device = torch.device("cuda:1")

n_channels=32

class Swish(nn.Module):
    """
    ### Swish 激活函数
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Land_Encoder(nn.Module):
    def __init__(self,n_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(10,n_channels,(3,3),(2,2),(1,1))
        self.act1 = Swish()
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels,2*n_channels,(3,3),(2,2),(1,1))
        self.act2 = Swish()
        self.norm2 = nn.BatchNorm2d(2*n_channels)
        self.conv3 = nn.Conv2d(2*n_channels,n_channels,(3,3),(2,2),(1,1))
        self.act3 = Swish()
        self.norm3 = nn.BatchNorm2d(n_channels)
    
    def forward(self,x):
        x = x.to(torch.long)
        b,c,w,h= x.shape
        x = x.view(b,w,h)
        x = nn.functional.one_hot(x,num_classes=10).float().permute(0,3,1,2)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        return x
    
class Popdem_Encoder(nn.Module):
    def __init__(self,n_channels) -> None:
        super().__init__()
        self.c_project = nn.Sequential(
        nn.Conv2d(2,n_channels,(3,3),stride=(2,2),padding=(1,1)),
        Swish(),
        nn.BatchNorm2d(n_channels),
        nn.Conv2d(n_channels,2*n_channels,(3,3),stride=(2,2),padding=(1,1)),
        Swish(),
        nn.BatchNorm2d(2*n_channels),
        nn.Conv2d(2*n_channels,n_channels,(3,3),stride=(2,2),padding=(1,1)),
        Swish(),
        nn.BatchNorm2d(n_channels)
        )

    def forward(self,x):
        return self.c_project(x)
    
class Cond_Encoder(nn.Module):
    def __init__(self,n_channels) -> None:
        super().__init__()
        self.land_encoder = Land_Encoder(n_channels)
        self.c_project = Popdem_Encoder(n_channels)

    def forward(self,condition):
        pop,dem,lan = torch.chunk(condition,chunks=3,dim=1)
        lan = self.land_encoder(lan)
        condition = torch.cat([pop,dem],dim=1)
        condition=self.c_project(condition)
        condition = torch.cat([condition,lan],dim=1)

        return condition

if __name__ == '__main__':
    loaders = Loaders(batch_size=4,data_type="RGB")
    train_data=loaders.train_loader
    test_data=loaders.test_loader
    databar = tqdm(train_data)

    cond_encoder = Cond_Encoder(n_channels).to(device)

    #for (pop, dem, lan),data in dataIter:
    for i, samples in enumerate(databar):
        conditions=samples[0]
        pop=conditions[0]
        dem=conditions[1]
        lan=conditions[2]

        data=samples[1]

        p = random.random()
        condition = torch.cat([pop,dem,lan],dim=1).to(device)
        data = data.to(device)
        condition_code=cond_encoder(condition)




