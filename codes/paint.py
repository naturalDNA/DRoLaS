import os
import torch.utils.data
from diffusion import *
from Unet4 import *
import torch
from dataloader import *
from torchvision.utils import save_image
from models import Autoencoder
import time

device = torch.device("cuda:1")
diffusion: DenoiseDiffusion
image_channels: int = 5
image_size: int = 512
n_channels: int = 64
channel_multipliers: List[int] = [1, 2, 2, 4]
is_attention: List[int] = [False, False, True, True]
z_channels = 16
n_steps: int = 1000
batch_size: int = 4
n_samples: int =4


def init_model():
    autoencoder = Autoencoder(image_channels,image_channels,z_channels=z_channels,channels=n_channels,mult_resolution=[2,4],attn_res=[False,False],res_block_num=2,masked=True).to(device)
    autoencoder.load_state_dict(torch.load("/home/lli/refine_ddpm/MASK_VAE/decoder_d_2_z_16/model/95.pkl",map_location=device))
    for name, parameter in autoencoder.named_parameters():
        parameter.requires_grad = False

    eps_model = UNet(
        image_channels=z_channels,
        n_channels=n_channels,
        ch_mults=channel_multipliers,
        is_attn=is_attention,
    ).to(device)
    eps_model.load_state_dict(torch.load("/home/lli/refine_ddpm/diffusion/masked_4_sample_diffusion_16/model/490_263.pkl",map_location=device))
    for name, parameter in autoencoder.named_parameters():
        parameter.requires_grad = False
        
    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=n_steps,
        device = device
    )
    return diffusion,autoencoder

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


def sample(model,condition):
    """
    ### Sample images
    """
    begin_time = time.time()
    with torch.no_grad():
        x = torch.randn([n_samples, image_channels, image_size, image_size],
                        device=device)

        for t in range(n_steps-1, -1, -1):
            x = model.p_sample(x, x.new_full((n_samples,),fill_value=t, dtype=torch.long),condition=condition)
    end_time = time.time()
    print(f"total_time_cost = {end_time-begin_time}")
    return x


def model_paint(model,autoencoder,condition,orig,mask):
    begin_time = time.time()
    with torch.no_grad():
        orig_latent = autoencoder.encode(orig)
        mask = F.interpolate(mask, size=(512//4,512//4), mode='nearest')
        ts = orig_latent.new_full((1,),fill_value=n_steps-1, dtype=torch.long)
        x = torch.randn((1,16,512//4,512//4),device = condition.device)
        
        for t in range(n_steps-1, -1, -1):
            ts = x.new_full((1,),fill_value=t, dtype=torch.long)
            x_fg = model.p_sample(x,ts,condition)
            x_bg = model.q_sample(orig_latent,ts)
            x = x_fg * mask + (1-mask)*x_bg
        x = autoencoder.decode(x)

    end_time = time.time()
    print(f"total_time_cost = {end_time-begin_time}")
    return x
    


   
def paint(model,autoencoder,condition,orig,mask_pixel,mask_latent,i):
    with torch.no_grad():
        print("begin paint")
        orig_b = orig*mask_pixel

        x = model_paint(model,autoencoder,condition,orig,mask_latent)
        
        x = orig*mask_pixel + (1-mask_pixel) * x
        print("successful paint")
        orig_b=orig_b.cpu()
        orig=orig.cpu()
        x = x.cpu()
        img = torch.cat([orig,orig_b,x],dim=0)
        img = tensor2png(img,3)    
        save_image(img ,os.path.join("/home/lli/refine_ddpm/paint/masked_2_16/local", 'repaint_{}.jpg'.format(i)), nrow=3)
        

def main():
    model,autoencoder = init_model()
    loader = Loaders(batch_size)
    with torch.no_grad():
        i=0
        for (pop, dem, lan),data in loader.test_loader:
            condition = torch.cat([pop,dem,lan],dim=1).to(device)
            data=data.to(device)
            mask_pixel = torch.ones((1,5,512,512),device=device)
            mask_pixel[:,:,50:200,50:200]=0
            
            mask_latent = torch.ones((1,16,512,512),device=device)
            mask_latent[:,:,50:200,50:200]=0
            paint(model,autoencoder,condition,data,mask_pixel,mask_latent,i)
            i+=1


if __name__ == '__main__':
    main()




