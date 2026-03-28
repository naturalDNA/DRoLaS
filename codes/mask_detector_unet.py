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

import math
from typing import Optional, Tuple, Union, List


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
    


    
# 构建完整的自编码器
class Autoencoder(nn.Module):
    def __init__(self,img_channels,latent_channels,masked=False,loss = "Focal"):
        super(Autoencoder, self).__init__()

        channel_multipliers: List[int] = [1, 2, 2, 4]
        is_attention: List[int] = [False, False, False, True]
        self.unet = UNet(image_channels=img_channels, n_channels=latent_channels,ch_mults=channel_multipliers,is_attn=is_attention,)

        self.masked = masked
        if loss == "Focal":
            self.loss_func = FocalLoss(gamma=2,alpha=[0.1,2.0])
        elif loss == "MSE":
            self.loss_func = torch.nn.MSELoss()
        self.loss_name=loss
        

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
        
        h = self.unet(data)
        #print(h.shape)

        recon_loss = self.loss_func(h,mask_onehot)
        return recon_loss

    def forward(self, x):
        reconstructed_image = self.unet(x)
        return reconstructed_image
    


class Swish(nn.Module):
    """
    ### Swish 激活函数
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 4):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings


    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings

        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 8):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        return: size 不变  输入 x：(b,c,w,h) t:(b,)  输出 x:(b,c,w,h)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        return self.conv(x)
    


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 5, n_channels: int = 4,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()
        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        out_channels = in_channels = n_channels

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels,  is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(4, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, 2, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        
        # Get time-step embeddings

        # Get image projection
        x = self.image_proj(x)
        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x)
        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()

                #print("x:{},s:{}".format(x.shape,s.shape))
                x = torch.cat((x, s), dim=1)
                x = m(x)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))




def main():
    epochs: int = 100
    loaders = Loaders(batch_size,data_type="numpy")
    train_data=loaders.train_loader
    test_data = loaders.test_loader
    it = iter(test_data)

    autoencoder = Autoencoder(img_channels=5,latent_channels=16,masked=True,loss="Focal").to(device=device)

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
            recon_loss = autoencoder.loss(data)
            loss = recon_loss
            databar.set_description('now_epoch=%d,recon_loss=%.3f' % (epoch,recon_loss))
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

                    save_image(fake_real, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0_unet/img", 'train_real_fake_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128,cmap='gary')
                    save_image(data_break, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0_unet/img", 'train_data_break_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128)

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


                    save_image(fake_real, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0_unet/img", 'test_real_fake_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128,cmap='gary')
                    save_image(data_break, os.path.join("/home/sadong/refine_ddpm/mask_detector/focal_cuda0_unet/img", 'data_break_fake_{}.png'.format(epoch+1)), nrow=4,padding=10,pad_value=128)
                    torch.save(autoencoder.state_dict(),"/home/sadong/refine_ddpm/mask_detector/focal_cuda0_unet/model/epoch{}.pkl".format(epoch+1))
                    print("successfully saved")
            i+=1

if __name__ == '__main__':
    main()