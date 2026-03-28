import torch
from typing import List


is_numpy = True
image_channels: int = 5 if is_numpy else 3
n_samples: int = 4
n_steps: int = 1000
image_size: int = 512
epochs : int = 100
z_channels : int = 4
n_channels : int = 64
channel_multipliers: List[int] = [1, 2, 2, 4]
is_attention: List[int] = [False, False, True, True]

sample_type='ddim'
epochs : int = 100

batch_size: int = 32
learning_rate = 2e-5

class_free=False
sample_or_train='train'
device = torch.device("cuda:0")
vae_path="/home/sadong/refine_ddpm/resnet_VAE/focal_cuda0/model/epoch50.pkl"
unet_path=None
#unet_path="/home/sadong/refine_ddpm/resnet_diffusion/focal_cuda0_spadecrossnew1/model/2700.pkl"
train_path="/home/sadong/refine_ddpm/resnet_diffusion/focal_cuda0_spadecross1_regionloss[1.0,1.2,1.4,1.4,1.4]"
#sample_path="/home/sadong/refine_ddpm/results/focal_padecrossnew1"
#data_path='/home/sadong/refine_ddpm/results/focal_spadecross/condition'   #仅在细化时使用

