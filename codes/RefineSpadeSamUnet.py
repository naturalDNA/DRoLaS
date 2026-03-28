'''
在原始Unet的基础上修改
'''

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
import torch.nn.functional as F

import config




class Swish(nn.Module):
    """
    ### Swish 激活函数
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            scale = torch.cat([avg_out, max_out], dim=1)
            scale = self.conv(scale)
            out = x * self.sigmoid(scale)
        except Exception as e:
            print(e)
            out = x

        return out
    
    
class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps = 1e-5):
        super().__init__()

        self.norm = nn.GroupNorm(32, norm_nc, affine=False) # 32/16
        self.act = nn.ReLU()

        self.eps = eps
        #nhidden = 128
        nhidden=label_nc
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.spatialAtt = SpatialAttention()

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        #actv = self.spatialAtt(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        x1 = x * (1 + gamma) + beta


        x2 = self.spatialAtt(self.act(x1))

        # apply scale and bias
        return x2


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

class TimeEmbedding(nn.Module):
    """
    t:(b,)
    return:(b,n_channels)
    """

    def __init__(self, n_channels: int):
        """
        n_channels: 决定了返回的维度 (b,n_channels)  要>=4
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # (channels//8)
        emb = t[:, None] * emb[None, :]  # (b,channels//8)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)  # (b,channels//4)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int = 1, n_groups: int = 16):
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
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)
    

class SDMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int = 1, n_groups: int = 16, c_channels: int = 16):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        #self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.norm1 = SPADEGroupNorm(in_channels, c_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        #self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.norm2 = SPADEGroupNorm(out_channels, c_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x,cond)))
        # Add time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h,cond)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
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

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
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
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        #self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.res = SDMResidualBlock(in_channels + out_channels, out_channels, time_channels,c_channels=out_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        x = self.res(x, t, cond)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        #self.res1 = SDMResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)
        #self.res2 = SDMResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)

class c_residule(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3),padding=(1,1))
        self.act1 = Swish()
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3),padding=(1,1))
        self.act2 = Swish()
        self.norm2 = nn.BatchNorm2d(out_channels)
        if in_channels!=out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (3, 3),padding=(1,1))
        else:
            self.shortcut = nn.Identity()
    
    def forward(self,x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)
    


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 1, n_channels: int = 64,
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
        self.image_proj = nn.Conv2d(image_channels*2, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.land_encoder = Land_Encoder(n_channels//2)
        #self.land_encoder = land_ae.encoder
        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        self.c_project = nn.Sequential(
                    nn.Conv2d(2,n_channels//2,(3,3),stride=(2,2),padding=(1,1)),
                    Swish(),
                    nn.BatchNorm2d(n_channels//2),
                    nn.Conv2d(n_channels//2,n_channels,(3,3),stride=(2,2),padding=(1,1)),
                    Swish(),
                    nn.BatchNorm2d(n_channels),
                    nn.Conv2d(n_channels,n_channels//2,(3,3),stride=(2,2),padding=(1,1)),
                    Swish(),
                    nn.BatchNorm2d(n_channels//2)
                    )
        #self.c_project = dempop_ae.encoder

        c_encoder=[]
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                c_encoder.append(
                    c_residule(in_channels,out_channels)
                )
                in_channels = out_channels
            if i != n_resolutions-1:
                c_encoder.append(nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1)))
        self.c_encoder = nn.ModuleList(c_encoder)

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
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor,x0, t: torch.Tensor,condition=None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        if condition!=None:
            pop,dem,lan = torch.chunk(condition,chunks=3,dim=1)
            lan = self.land_encoder(lan)
            cond = torch.cat([pop,dem],dim=1)
            cond=self.c_project(cond)
            cond = torch.cat([cond,lan],dim=1)
            c = [cond]
            for encoder in self.c_encoder:
                cond = encoder(cond)
                c.append(cond)
        else:
            c=[torch.tensor([0],device=x.device)]
            for encoder in self.c_encoder:
                c.append(torch.tensor([0],device=x.device))
        
        # Get time-step embeddings
        t = self.time_emb(t)


        x=torch.cat((x, x0), dim=1)

        x = self.image_proj(x)
        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)
        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                #cond = c.pop()
                #x = torch.cat((x, s+condition), dim=1)
                x = torch.cat((x, s), dim=1)
                if condition==None:
                    cond=torch.zeros_like(s)
                else:
                    cond = c.pop()
                #x = m(x, t)
                x = m(x, t, cond)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

# x=torch.rand((4,1,128,128))
# t=torch.rand((4,))
# cond=torch.rand((4,2,512,512))
# model=UNet(image_channels=1,n_channels=32)
# print(model(x,t,cond).shape)
