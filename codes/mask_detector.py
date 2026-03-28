import random
import torch
import torch.nn.functional as F
from torch import nn
from dataloader import Loaders
from torchvision.utils import save_image
import os
import torch.nn as nn
import torch
from torchvision import models
from tqdm import tqdm
import numpy as np

import config

device=torch.device("cuda:1")
batch_size: int = 4

def tensor_to_png(tensor,samples):
    color_map = torch.tensor([
        [0, 0, 0],         # no use
        [255, 255, 255],   # residential
        [0, 0, 255],     # tertiary
        [0, 255, 0],     # primary secondary
        [255,0,0] ,    #trunk motorway
    ], dtype=torch.uint8)

    new_tensor = torch.zeros((samples, 512, 512,3), dtype=torch.uint8)
    for i in range(5):
        mask = tensor == i
        new_tensor[ mask,:] = color_map[i]

    return new_tensor.permute(0,3,1,2)

def tensor2png(data,samples):
    data = torch.argmax(data,dim=1)
    data = tensor_to_png(data,samples)/255
    return data

def tensor_to_one_channel(tensor,samples):
    color_map = torch.tensor([
        [0],         # no use
        [255],   # residential
        [255],     # tertiary
        [255],     # primary secondary
        [255] ,    #trunk motorway
    ], dtype=torch.uint8)

    new_tensor = torch.zeros((samples, 512, 512,1), dtype=torch.uint8,device=device)
    for i in range(5):
        mask = tensor == i
        new_tensor[mask,:] = color_map[i]

    return new_tensor.permute(0,3,1,2)

def tensor2one_channel(data,samples):
    data = torch.argmax(data,dim=1)
    data = tensor_to_one_channel(data,samples)/255
    return data

def two_to_one_channel(tensor,samples):
    color_map = torch.tensor([
        [0],         # no use
        [255],   # residential
    ], dtype=torch.uint8)

    new_tensor = torch.zeros((samples, 512, 512,1), dtype=torch.uint8,device=device)
    for i in range(2):
        mask = tensor == i
        new_tensor[mask,:] = color_map[i]

    return new_tensor.permute(0,3,1,2)

def two2one_channel(data,samples):
    data = torch.argmax(data,dim=1)
    data = two_to_one_channel(data,samples)/255
    return data


palette = [[0], [1]]

def mask_to_onehot(mask):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=None):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

#     def forward(self, preds, labels):
#         if self.ignore_index is not None:
#             mask = labels != self.ignore
#             labels = labels[mask]
#             preds = preds[mask]

#         logpt = -self.bce_fn(preds, labels)
#         pt = torch.exp(logpt)
#         loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
#         return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2,num_classes = 2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            
        labels = target.clone().long()
        labels = labels.view(labels.size(0),labels.size(1),-1)
        labels = labels.transpose(1,2)
        labels = labels.contiguous().view(-1,labels.size(2))
        labels = torch.argmax(labels,dim = 1).unsqueeze(1)

        logpt =  F.log_softmax(input,dim = 1)
        pt = torch.exp(logpt)
        
        logpt = logpt.gather(1,labels).squeeze()
        pt = pt.gather(1,labels).squeeze()
        alpha = self.alpha.to(labels.device).unsqueeze(0)
        alpha = alpha.repeat(labels.shape[0], 1)
        alpha = alpha.gather(0,labels).squeeze()
        loss = -torch.mul(torch.pow((1-pt), self.gamma), logpt.t())
        loss = torch.mul(alpha, loss.t())
        if self.size_average: return loss.mean()
        else: return loss.sum()

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
# 构建 InceptionV3 编码器
class Encoder(nn.Module):
    def __init__(self,img_channels,latent_channels):
        super(Encoder, self).__init__()
        new_conv = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model = models.resnet101(pretrained=False)
        self.model.conv1 = new_conv
        encoder = list(self.model.children())[:-4]
        self.encoder_layers = nn.Sequential(*encoder)
        self.bn = nn.BatchNorm2d(512)
        self.active = nn.ReLU()
        self.output = nn.Sequential(
            nn.ConvTranspose2d(512, latent_channels*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(latent_channels*2),
            nn.ReLU(),
        )
        
    def forward(self, x):
        #print(x.shape)
        #x1=torch.rand(4, 5, 512, 512).to(config.device)
        #print(x1.shape)
        features = self.encoder_layers(x)
        #print(features.shape)
        features = self.bn(features)
        features = self.active(features)
        features = self.output(features)
        #print(features.shape)
        return features

# 构建解码器
class Decoder(nn.Module):
    def __init__(self,img_channels,latent_channels):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            UpNet(latent_channels,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            UpNet(256,128),
            ResidualBlock(128,64),
            UpNet(64,64),
            ResidualBlock(64,64),
            ResidualBlock(64,32),
            nn.GroupNorm(8,32),
            Swish(),
            nn.Conv2d(32, img_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.Tanh()
        )
        
    def forward(self, x):
        reconstructed_image = self.decoder_layers(x)
        return reconstructed_image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 16):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))

        return h + self.shortcut(x)
    
class UpNet(nn.Module):
    def  __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res = ResidualBlock(out_channels,out_channels)
        
    def forward(self,x):
        return self.res(self.transconv(x))


    
# 构建完整的自编码器
class Autoencoder(nn.Module):
    def __init__(self,img_channels,latent_channels,masked=False,loss = "Focal"):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(img_channels,latent_channels)
        self.decoder = Decoder(2,latent_channels)
        self.masked = masked
        if loss == "Focal":
            self.loss_func = FocalLoss(gamma=2,alpha=[0.1,2.0])
        elif loss == "MSE":
            self.loss_func = torch.nn.MSELoss()
        self.loss_name=loss
        
    def encode(self,x):
        x = self.encoder(x)
        mean,z_log_var = torch.chunk(x,chunks=2,dim=1)
        eps = torch.randn_like(mean)
        return mean + torch.exp(0.5 * z_log_var) * eps
    
    def decode(self,x):
        return self.decoder(x)

    def loss(self,x):
        #orig = x.clone()
        data=x.clone()
        bs,channels,height,width = x.size()
        #random_box_x,random_box_y = random.randint(10,100),random.randint(10,100)
        orig_mask = torch.ones(bs,1,512,512).to(x.device)
        random_box_num = random.randint(100,200)
        for i in range(bs):
            for _ in range(random_box_num):
                random_box_x,random_box_y = random.randint(10,30),random.randint(10,30)
                center_x,center_y = random.randint(1,height-random_box_x-1),random.randint(1,height-random_box_y-1)
                #center_x,center_y = random.randint(random_box_x+1,height-random_box_x-1),random.randint(random_box_y+1,height-random_box_y-1)
                one_hot_tensor = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32,device=x.device).view(1, 5, 1, 1).expand(1,5,random_box_x,random_box_y)
                data[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = one_hot_tensor
                orig_mask[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = 0
        
        data_one_channel = tensor2one_channel(x,batch_size).to(x.device)
        orig_mask_not = torch.ones_like(orig_mask)-orig_mask

        #mask=orig_mask_not
        mask=orig_mask_not*data_one_channel
        mask_npy=mask.cpu().numpy().transpose(0,2,3,1)

        mask_onehot = mask_to_onehot(mask_npy)
        mask_onehot = torch.tensor(mask_onehot.transpose(0,3,1,2)).to(device)
        
        h = self.encoder(data)
        #print(h.shape)

        mean,z_log_var = torch.chunk(h,chunks=2,dim=1)
        KL_loss = -0.5 * torch.mean(1 + z_log_var - mean.pow(2) - z_log_var.exp())
        eps = torch.randn_like(mean)
        h = mean + torch.exp(0.5 * z_log_var) * eps
        
        h = self.decode(h)

        recon_loss = self.loss_func(h,mask_onehot)
        return recon_loss,KL_loss

    def forward(self, x):
        encoded_features = self.encode(x)
        reconstructed_image = self.decode(encoded_features)
        return reconstructed_image
    

def main():
    epochs: int = 100
    loaders = Loaders(batch_size,data_type="numpy")
    train_data=loaders.train_loader
    test_data = loaders.test_loader
    it = iter(test_data)

    autoencoder = Autoencoder(img_channels=5,latent_channels=4,masked=True,loss="Focal").to(device=device)

    #autoencoder.load_state_dict(torch.load("/home/lli/refine_ddpm/resnet_VAE/RGB/model/epoch60.pkl",map_location=device))
    optimizer = torch.optim.Adam([{'params': autoencoder.parameters(), 'initial_lr': 1e-5}],lr=1e-5)
    print("model is prepared")
    i=0
    for epoch in range(0,epochs):
        # Train the model
        t = random.randint(0,200)
        i= 0
        print("epoch: %i/%i" % (int(epoch), int(epochs)))
        databar = tqdm(train_data)
        #for (pop, dem, lan),data in train_data:
        for i, samples in enumerate(databar):
            conditions=samples[0]
            pop=conditions[0]
            dem=conditions[1]
            lan=conditions[2]

            data=samples[1]

            data = data.to(device)
            optimizer.zero_grad()
            # fake_data = autoencoder(data)
            # loss = nn.functional.mse_loss(data,fake_data)
            recon_loss,KL_loss = autoencoder.loss(data)
            loss = recon_loss + 0.0001*KL_loss
            databar.set_description('now_epoch=%d,recon_loss=%.3f,KL_loss=%.3f,total_loss=%.3f' % (epoch,recon_loss,KL_loss,loss))
            #print("now_epoch={},recon_loss={},KL_loss={},total_loss={}".format(epoch,recon_loss,KL_loss,loss))
            loss.backward()
            optimizer.step()

            if ((epoch+1) % 10 == 0 or epoch==0)and i==t:
                with torch.no_grad():
                    break_data=data.clone()
                    bs,channels,height,width = data.size()
                    #random_box_x,random_box_y = random.randint(10,100),random.randint(10,100)
                    orig_mask = torch.ones(bs,1,512,512).to(data.device)
                    random_box_num = random.randint(100,200)
                    for i in range(bs):
                        for _ in range(random_box_num):
                            random_box_x,random_box_y = random.randint(10,30),random.randint(10,30)
                            center_x,center_y = random.randint(1,height-random_box_x-1),random.randint(1,height-random_box_y-1)
                            #center_x,center_y = random.randint(random_box_x+1,height-random_box_x-1),random.randint(random_box_y+1,height-random_box_y-1)
                            one_hot_tensor = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32,device=data.device).view(1, 5, 1, 1).expand(1,5,random_box_x,random_box_y)
                            break_data[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = one_hot_tensor
                            orig_mask[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = 0

                    data_one_channel = tensor2one_channel(data,batch_size).to(data.device)
                    orig_mask_not = torch.ones_like(orig_mask)-orig_mask

                    mask=orig_mask_not*data_one_channel

                    fake_data = autoencoder(break_data)
                    
                    fake_data = two2one_channel(fake_data,batch_size)
                    fake_real=torch.cat([mask.cpu(),fake_data.cpu()],dim=0)
                    data_break = torch.cat([data,break_data],dim=0)
                    data_break = tensor2png(data_break,2*batch_size)

                    save_image(fake_real, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0/img", 'train_real_fake_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128,cmap='gary')
                    save_image(data_break, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0/img", 'train_data_break_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128)
                    
                    (pop, dem, lan),data = next(it)
                    data = data.to(device)
                    break_data=data.clone()
                    bs,channels,height,width = data.size()
                    #random_box_x,random_box_y = random.randint(10,100),random.randint(10,100)
                    orig_mask = torch.ones(bs,1,512,512).to(data.device)
                    random_box_num = random.randint(50,70)
                    for i in range(bs):
                        for _ in range(random_box_num):
                            random_box_x,random_box_y = random.randint(10,30),random.randint(10,30)
                            center_x,center_y = random.randint(1,height-random_box_x-1),random.randint(1,height-random_box_y-1)
                            #center_x,center_y = random.randint(random_box_x+1,height-random_box_x-1),random.randint(random_box_y+1,height-random_box_y-1)
                            one_hot_tensor = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32,device=data.device).view(1, 5, 1, 1).expand(1,5,random_box_x,random_box_y)
                            break_data[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = one_hot_tensor
                            orig_mask[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = 0

                    data_one_channel = tensor2one_channel(data,1).to(data.device)
                    orig_mask_not = torch.ones_like(orig_mask)-orig_mask

                    mask=orig_mask_not*data_one_channel

                    fake_data=autoencoder(break_data)
                    fake_data = two2one_channel(fake_data,1)
                    mask=mask.cpu()
                    fake_data=fake_data.cpu()

                    break_data = break_data.cpu()                    
                    data = data.cpu()

                    fake_real=torch.cat([mask,fake_data],dim=0)

                    data_break = torch.cat([data,break_data],dim=0)
                    data_break = tensor2png(data_break,2)


                    save_image(fake_real, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0/img", 'test_real_fake_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128,cmap='gary')
                    save_image(data_break, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0/img", 'data_break_fake_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128)
                    torch.save(autoencoder.state_dict(),"/home/sadong/refine_ddpm/mask_detector/focal_cuda0/model/epoch{}.pkl".format(epoch+1))
                    print("successfully saved")
            i+=1

if __name__ == '__main__':
    main()