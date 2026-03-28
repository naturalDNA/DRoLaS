# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch.autograd import Variable
import random

class FocalLoss(nn.Module):
    def __init__(self, gamma=0,num_classes = 5, alpha=None, size_average=True):
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
    
    
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        return x*torch.sigmoid(x)
    
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout()
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinearAttention(in_channels)



class Autoencoder(nn.Module):
    def __init__(self,in_channels,out_channels,z_channels,channels,mult_resolution=[1,2,2,4],attn_res=[False,False,False,True],res_block_num = 2,masked=False,loss = "Focal"):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels,z_channels,channels,mult_resolution,res_block_num,attn_res)
        self.decoder = Decoder(out_channels,z_channels,channels,mult_resolution,res_block_num,attn_res)
        if loss == "Focal":
            self.loss_func = FocalLoss(gamma=2,alpha=[0.1,0.3,0.6,1.0,2.0])
        elif loss == "MSE":
            self.loss_func = torch.nn.MSELoss()
        
        self.masked = masked
    
    def encode(self,x):
        x = self.encoder(x)
        mean,z_log_var = torch.chunk(x,chunks=2,dim=1)
        eps = torch.randn_like(mean)
        return mean + torch.exp(0.5 * z_log_var) * eps
    
    def decode(self,x):
        return self.decoder(x)
        

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return  x
    
    def loss(self,x):
        orig = x.clone()
        if self.masked:
            bs,channels,height,width = x.size()
            random_box_x,random_box_y = random.randint(10,100),random.randint(10,100)
            for i in range(bs):
                for _ in range(10):
                    center_x,center_y = random.randint(random_box_x+1,height-random_box_x-1),random.randint(random_box_y+1,height-random_box_y-1)
                    if self.loss == "Focal":
                        one_hot_tensor = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32,device=x.device).view(1, 5, 1, 1).expand(1,5,random_box_x,random_box_y)
                        x[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = one_hot_tensor
                    elif self.loss == "MSE":
                        masked_tensor = torch.tensor([0, 0, 0], dtype=torch.float32,device=x.device).view(1, 3, 1, 1).expand(1,3,random_box_x,random_box_y)
                        x[i,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = masked_tensor
        
                
        h = self.encoder(x)
        mean,z_log_var = torch.chunk(h,chunks=2,dim=1)
        KL_loss = -0.5 * torch.mean(1 + z_log_var - mean.pow(2) - z_log_var.exp())
        eps = torch.randn_like(mean)
        h = mean + torch.exp(0.5 * z_log_var) * eps
        
        h = self.decode(h)
        recon_loss = self.loss_func(h,orig)
        return recon_loss,KL_loss



class Encoder(nn.Module):
    def __init__(self, in_channels, z_channels,channels = 64,mult_resolution=[1,2,2,4],res_block_num = 2 , attn_res=[False,False,False,True],is_VQ = False):
        super(Encoder, self).__init__()
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels,channels,(3,3),1,1),
            Normalize(channels),
            Swish()
        )
        
        in_channel = channels
        moudules = []
        for i,mult in enumerate(mult_resolution):
            out_channel = in_channel*mult
            moudules.append(ResnetBlock(in_channels=in_channel,out_channels=out_channel,conv_shortcut=True))
            for _ in range(res_block_num-1):
                moudules.append(ResnetBlock(in_channels=out_channel,out_channels=out_channel,conv_shortcut=True))
            moudules.append(Downsample(out_channel,with_conv=True))
            if attn_res[i]:
                attn = make_attn(out_channel,attn_type="linear")
                moudules.append(attn)
            in_channel = out_channel
            
        self.moudules =  nn.Sequential(*moudules)
        if is_VQ:
            self.proj_out = nn.Sequential(
            nn.Conv2d(out_channel,z_channels,(3,3),1,1),
            Normalize(z_channels,4),
            Swish()
        )
        else:
            self.proj_out = nn.Sequential(
                nn.Conv2d(out_channel,2*z_channels,(3,3),1,1),
                Normalize(2*z_channels,4),
                Swish()
            )

    def forward(self, x):
        x = self.proj_in(x)
        x = self.moudules(x)
        x = self.proj_out(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, out_channels, z_channels,channels = 64,mult_resolution=[1,2,2,4],res_block_num = 2 , attn_res=[False,False,False,True]):
        super(Decoder, self).__init__()
        in_channel = channels
        for mult in mult_resolution:
            in_channel = in_channel *mult
            
        self.proj_in = nn.Sequential(
            nn.Conv2d(z_channels,in_channel,(3,3),1,1),
            Normalize(in_channel),
            Swish()
        )
        
        moudules = []
        for i,mult in enumerate(mult_resolution):
            out_channel = in_channel//mult
            moudules.append(ResnetBlock(in_channels=in_channel,out_channels=out_channel,conv_shortcut=True))
            for _ in range(res_block_num-1):
                moudules.append(ResnetBlock(in_channels=out_channel,out_channels=out_channel,conv_shortcut=True))
            moudules.append(Upsample(out_channel,with_conv=True))
            if attn_res[i]:
                attn = make_attn(out_channel,attn_type="linear")
                moudules.append(attn)
            in_channel = out_channel
        self.moudules =  nn.Sequential(*moudules)
        self.proj_out = nn.Sequential(
            nn.Conv2d(out_channel,out_channels,(3,3),1,1),
            Normalize(out_channels,1),
            Swish()
        )
        
    def forward(self, x):
        x = self.proj_in(x)
        x = self.moudules(x)
        x = self.proj_out(x)
        return x
    
    
    
    


# model = Autoencoder(5,5,z_channels=512,channels=32).to(device = torch.device("cuda:1"))
# model.load_state_dict(torch.load("/home/lli/refine_ddpm/autoencoder/model/11.pkl",map_location=torch.device("cuda:1")))
# x = torch.rand((4,5,512,512),device = torch.device("cuda:1") )
# print(model.encode(x).shape)
# print(x)