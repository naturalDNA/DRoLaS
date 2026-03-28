import torch
import config
import math
import os
import time

from torchvision.utils import save_image
from diffusion import DenoiseDiffusion
from typing import Optional, Tuple, Union

from diffusion import *
from Unet import *
from resnet_vae import Autoencoder
from dataloader import Loaders

def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    consts:(time_steps,)
    t:(b,)
    return: (b,1,1,1)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = gather(self.alpha_bar,timestep)
    alpha_prod_t_prev = gather(self.alpha_bar,prev_timestep)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def ddpm_step_with_logprob(self: DenoiseDiffusion, model_output: torch.FloatTensor, timestep: int,
                           sample: torch.FloatTensor, eta: float = 1.0,prev_sample: Optional[torch.FloatTensor] = None, ):
    if(timestep==0):
        timestep+=1
    prev_timestep = (timestep - 1)
    prev_timestep = torch.clamp(prev_timestep, 0, 1000 - 1)
    alpha_prod_t = gather(self.alpha_bar,timestep)
    alpha_prod_t_prev = gather(self.alpha_bar,prev_timestep)
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (
                                   sample - beta_prod_t ** (0.5) * model_output
                           ) / alpha_prod_t ** (0.5)
    pred_epsilon = model_output
    pred_original_sample = pred_original_sample.clamp(
        -1.0, 1.0
    )
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)
    #print(std_dev_t)
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon
    prev_sample_mean = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is None:
        variance_noise = torch.randn(model_output.shape, device=model_output.device)
        prev_sample = prev_sample_mean + std_dev_t * variance_noise


    log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
            - torch.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    #print(torch.log(std_dev_t))
    #print(log_prob)
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    #print(timestep,log_prob)

    return prev_sample.type(sample.dtype), log_prob

def make_timesteps(num_steps: int = 1000):
    timesteps = []
    for i in range(num_steps-1, -1, -1):
        timesteps.append(i)
    timesteps = torch.tensor(timesteps)
    return timesteps


def latent_sample(model,autoencoder,condition,z_channels,latent_size):
    with torch.no_grad():
        x = torch.randn([config.n_samples, z_channels, latent_size, latent_size],device=config.device)
        for t in range(config.n_steps-1,-1,-1):
            x = model.p_sample(x, x.new_full((config.n_samples,),fill_value=t, dtype=torch.long),condition=condition)
        return autoencoder.decode(x)

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
        new_tensor[mask,:] = color_map[i]

    return new_tensor.permute(0,3,1,2)


def tensor2png(data,samples):
    data = torch.argmax(data,dim=1)
    data = tensor_to_png(data,samples)/255
    return data


def sample_uncon(model, autoencoder, save_path):
    for i in range(2000):
        begin_time = time.time()
        img = latent_sample(model,autoencoder,None,4,64)
        if config.is_numpy:
            img = tensor2png(img,1)
        save_image(img, os.path.join(save_path, '{}.png'.format(i)), nrow=1)
        end_time = time.time()
        print("total_time = {},i={}".format(end_time-begin_time,i))
        print(i)

def sample_con(model,autoencoder,test_data,save_path):
    i = 0
    for (pop, dem, lan),data in test_data:
        condition = torch.cat([pop,dem,lan],dim=1).to(config.device)
        if config.is_numpy:
            data = tensor2png(data,1)
        for j in range(4):
            begin_time = time.time()
            img = latent_sample(model,autoencoder,condition,4,64)
            if config.is_numpy:
                img = tensor2png(img,1)
            save_image(img, os.path.join(save_path, '{}_{}.png'.format(i,j)), nrow=1)
            end_time = time.time()
            print("total_time = {},i={}".format(end_time-begin_time,i))
        i += 1

def init_model():
    # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
    # autoencoder = Autoencoder(image_channels,image_channels,z_channels=4,channels=64,mult_resolution=[1,2,4],attn_res=[False,False,True]).to(device)

    # autoencoder = Autoencoder(image_channels,image_channels,z_channels=z_channels,channels=64,mult_resolution=[1,2,4],attn_res=[False,False,True],res_block_num=2).to(device)
    image_channels: int = 5 if config.is_numpy else 3
    autoencoder = Autoencoder(img_channels=image_channels,latent_channels=config.z_channels,masked=True).to(device=config.device)
    autoencoder.load_state_dict(torch.load("/home/lli/sadong/refine_ddpm/resnet_VAE_dong/numpy/model/epoch50.pkl",map_location=config.device))
    for name, parameter in autoencoder.named_parameters():
        parameter.requires_grad = False

    eps_model = UNet(
        image_channels=config.z_channels,
        n_channels=config.n_channels,
        ch_mults=config.channel_multipliers,
        is_attn=config.is_attention,
    ).to(config.device)
    eps_model.load_state_dict(torch.load("/home/lli/sadong/refine_ddpm/resnet_diffusion_dong/model/1900.pkl",map_location=config.device))
    # Create [DDPM class](index.html)
    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=config.n_steps,
        device = config.device
    )
    return diffusion,autoencoder

if __name__ == '__main__':
    model,autoencoder = init_model()
    save_path='/home/lli/sadong/refine_ddpm/results/baseline_orig/con'
    data_type = "numpy" if config.is_numpy else "RGB"
    loaders = Loaders(1,data_type=data_type)
    test_data=loaders.test_loader
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sample_con(model, autoencoder, test_data, save_path)