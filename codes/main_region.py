import os
from typing import List
import torch.utils.data
from diffusion_region import *
from SpadeCrossUnet import *
import torch
from dataloader_split import Loaders
from torchvision.utils import save_image
import random
from resnet_vae import Autoencoder
import time
import shutil
from tqdm import tqdm

import cv2
from skimage import morphology

is_numpy = True

device = config.device
diffusion: DenoiseDiffusion
image_channels: int = 5 if is_numpy else 3
image_size: int = 512
n_channels = config.n_channels
channel_multipliers: List[int] = [1, 2, 2, 4]
is_attention: List[int] = [False, False, True, True]
z_channels = 4

n_steps: int = 1000
batch_size = config.batch_size
if config.sample_or_train=='train':
    n_samples = config.n_samples
else:
    n_samples=1
    config.n_samples=n_samples
learning_rate: float = 2e-5
epochs: int = 3000

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

def tensor_to_region(tensor,samples):
    color_map = torch.tensor([
        [0, 0, 0, 0, 0],         # no use
        [1,1,1,1,1],   # residential
        [1,1,1,1,1],     # tertiary
        [1,1,1,1,1],     # primary secondary
        [1,1,1,1,1] ,    #trunk motorway
    ], dtype=torch.float32,device=device)

    new_tensor = torch.zeros((samples, 512, 512,5), dtype=torch.float32,device=device)
    for i in range(5):
        mask = tensor == i
        new_tensor[mask,:] = color_map[i]

    return new_tensor.permute(0,3,1,2)


def tensor2region(data,samples):
    data = torch.argmax(data,dim=1)
    data = tensor_to_region(data,samples)
    return data


def init_model():
    # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
    # autoencoder = Autoencoder(image_channels,image_channels,z_channels=4,channels=64,mult_resolution=[1,2,4],attn_res=[False,False,True]).to(device)

    # autoencoder = Autoencoder(image_channels,image_channels,z_channels=z_channels,channels=64,mult_resolution=[1,2,4],attn_res=[False,False,True],res_block_num=2).to(device)
    autoencoder = Autoencoder(img_channels=image_channels,latent_channels=z_channels,masked=True).to(device=device)
    if config.vae_path:
        autoencoder.load_state_dict(torch.load(config.vae_path,map_location=device))
    for name, parameter in autoencoder.named_parameters():
        parameter.requires_grad = False

    eps_model = UNet(
        image_channels=z_channels,
        n_channels=n_channels,
        ch_mults=channel_multipliers,
        is_attn=is_attention,
    ).to(device)

    if config.unet_path:
        eps_model.load_state_dict(torch.load(config.unet_path,map_location=device))


    # Create [DDPM class](index.html)
    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=n_steps,
        device = device
    )

    return diffusion,autoencoder


def sample(model,condition):
    with torch.no_grad():
        x = torch.randn([n_samples, image_channels, image_size, image_size],device=device)

        for t in range(n_steps-1, -1, -1):
            x = model.p_sample(x, x.new_full((n_samples,),fill_value=t, dtype=torch.long),condition=condition)
        return x

def latent_sample(model,autoencoder,condition,z_channels,latent_size):
    with torch.no_grad():
        x = torch.randn([n_samples, z_channels, latent_size, latent_size],device=device)
        if config.sample_type=='ddpm':
            for t in range(config.n_steps-1,-1,-1):
                x = model.p_sample(x, x.new_full((config.n_samples,),fill_value=t, dtype=torch.long),condition=condition)
        elif config.sample_type=='ddim':
            if config.class_free:
                x=model.classifire_p_sample_ddim(x,condition=condition,scale=1.5)
            else:
                x=model.p_sample_ddim(x,condition=condition)
        return autoencoder.decode(x)


def train_downsample(model,autoencoder,dataIter, optimizer,epoch,iterator,save_path,data_type):
    c_img_path =  os.path.join(save_path, 'c_img')
    uc_img_path =  os.path.join(save_path, 'uc_img')
    model_path = os.path.join(save_path, 'model')

    if not os.path.exists(c_img_path):
        os.makedirs(c_img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(uc_img_path):
        os.makedirs(uc_img_path)

    # Iterate through the dataset
    t,i=random.randint(0,50), 0

    databar = tqdm(dataIter)

    #for (pop, dem, lan),data in dataIter:
    for i, samples in enumerate(databar):
        conditions=samples[0]
        pop=conditions[0]
        dem=conditions[1]
        lan=conditions[2]

        data=samples[1]

        p = random.random()
        condition = torch.cat([pop,dem,lan],dim=1).to(device)
        if p>0.5:
            condition = None
        rotate = random.randint(0,3)
        data = torch.rot90(data,rotate,dims=(2,3))
        if p<=0.5:
            condition = torch.rot90(condition,rotate,dims=(2,3))

        data = data.to(device)
        code = autoencoder.encode(data)

        # region = tensor2region(data,batch_size)
        # region_code=autoencoder.encode(region)
        region = data.clone()
        region_code=torch.zeros([4,batch_size,code.size()[1],code.size()[2],code.size()[3]],dtype=torch.float32)
        for b in range(batch_size):
            region_1b=region[b,:,:,:]
            for k in range(region_1b.size()[0]):
                mask=region_1b[k,:,:]
                mask_npy=mask.cpu().numpy()
                kernel = np.ones((3, 3), np.float32)
                mask_npy = cv2.dilate(mask_npy, kernel, iterations = 1)
                mask=torch.tensor(mask_npy)
                region_1b[k,:,:]=mask
            region[b,:,:,:]=region_1b
        region = F.interpolate(region, size=code.size()[2:], mode='bilinear')
        for k in range(region.size()[1]):
            if k!=0:
                region_1c=region[:,k,:,:].unsqueeze(1).repeat(1,code.size()[1],1,1)
                region_code[k-1,:,:,:,:]=region_1c
        region_code = region_code.to(device)

        optimizer.zero_grad()
        loss = model.loss(code,condition=condition,region=region_code)
        #print("now_epoch={},now_loss={}".format(epoch,loss))
        databar.set_description('now_epoch=%d,loss=%.3f' % (epoch+1,loss))

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if (epoch+1)%20==0 and epoch!=0 and i==t:
                print("begin generate")
                (pop_t, dem_t, lan_t),data_t =next(iterator)
                condition_t = torch.cat([pop_t,dem_t,lan_t],dim=1).to(device)
                data=data[:n_samples-1].cpu()
                data=torch.cat([data,data_t],dim=0)

                if condition !=None:
                    condition = condition[:n_samples-1]
                    condition = torch.cat([condition,condition_t],dim=0)
                if data_type == "numpy":
                    data = tensor2png(data,n_samples)
                    img = latent_sample(model,autoencoder,condition,z_channels=z_channels,latent_size=image_size//8).cpu()
                    img = tensor2png(img,n_samples)
                    img = torch.cat([data,img],dim = 0).view(n_samples*2,3,512,512)
                
                else:
                    img = latent_sample(model,autoencoder,condition,z_channels=z_channels,latent_size=image_size//8).cpu()
                    img = torch.cat([data,img],dim = 0).view(n_samples*2,3,512,512)

                if condition == None:
                    save_image(img, os.path.join(save_path, 'uc_img/{}.png'.format(epoch+1)), nrow=n_samples)
                else:
                    save_image(img, os.path.join(save_path, 'c_img/{}.png'.format(epoch+1)), nrow=n_samples)
                torch.save(model.eps_model.state_dict(),os.path.join(save_path, 'model/{}.pkl'.format(epoch+1)).format(epoch+1))
                print("successfully saved")
        i+=1    


def train_encode(model,autoencoder,dataIter, optimizer,epoch,iterator,save_path,data_type):
    c_img_path =  os.path.join(save_path, 'c_img')
    uc_img_path =  os.path.join(save_path, 'uc_img')
    model_path = os.path.join(save_path, 'model')

    if not os.path.exists(c_img_path):
        os.makedirs(c_img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(uc_img_path):
        os.makedirs(uc_img_path)

    # Iterate through the dataset
    t,i=random.randint(0,50), 0

    databar = tqdm(dataIter)

    #for (pop, dem, lan),data in dataIter:
    for i, samples in enumerate(databar):
        conditions=samples[0]
        pop=conditions[0]
        dem=conditions[1]
        lan=conditions[2]

        data=samples[1]

        p = random.random()
        condition = torch.cat([pop,dem,lan],dim=1).to(device)
        if p>0.5:
            condition = None
        rotate = random.randint(0,3)
        data = torch.rot90(data,rotate,dims=(2,3))
        if p<=0.5:
            condition = torch.rot90(condition,rotate,dims=(2,3))

        # condition = torch.cat([pop,dem,lan],dim=1).to(device)

        # rotate = random.randint(0,3)
        # data = torch.rot90(data,rotate,dims=(2,3))
        # condition = torch.rot90(condition,rotate,dims=(2,3))

        data = data.to(device)
        #with torch.no_grad():
        code = autoencoder.encode(data)

        # region = tensor2region(data,batch_size)
        # region_code=autoencoder.encode(region)
        data1 = data.clone()
        # region=torch.zeros([batch_size*4,data.size()[1],data.size()[2],data.size()[3]],dtype=torch.float32)
        # for k in range(4):
        #     region_1c = data1[:,k+1,:,:].unsqueeze(1).repeat(1,5,1,1)
        #     region[32*k:32*(k+1),:,:,:]=region_1c
        region=torch.zeros([batch_size*2,data.size()[1],data.size()[2],data.size()[3]],dtype=torch.float32).to(device)
        # for k in range(4):
        #     region_1c = data1[:,k+1,:,:].unsqueeze(1).repeat(1,5,1,1)
        #     region[32*k:32*(k+1)]=region_1c       
        for k in range(4):
            region_1c = data1[:,k+1,:,:].unsqueeze(1).repeat(1,5,1,1)
            if k==0:
                region[0:32]=region_1c    
            else:
                region[32:64]+=region_1c            

        region = region.to(device)
        #region_code=torch.zeros([batch_size*4,code.size()[1],code.size()[2],code.size()[3]],dtype=torch.float32).to(device)
        #for k in range(4):
            #region_code[32*k:32*(k+1)]=autoencoder.encode(region[32*k:32*(k+1)])
        region_code=autoencoder.encode(region)
        #region_code=region_code.detach()

        optimizer.zero_grad()
        loss = model.loss(code,condition=condition,region=region_code)
        #loss = model.loss(code,condition=condition)
        #print("now_epoch={},now_loss={}".format(epoch,loss))
        databar.set_description('now_epoch=%d,loss=%.3f' % (epoch+1,loss))

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if ((epoch+1)%20==0 or epoch==0) and i==t:
                print("begin generate")
                (pop_t, dem_t, lan_t),data_t =next(iterator)
                condition_t = torch.cat([pop_t,dem_t,lan_t],dim=1).to(device)
                data=data[:n_samples-1].cpu()
                data=torch.cat([data,data_t],dim=0)

                if condition !=None:
                    condition = condition[:n_samples-1]
                    condition = torch.cat([condition,condition_t],dim=0)
                # con_zero=torch.zeros_like(condition[0])
                # if condition !=None:
                #     condition_tr=torch.zeros_like(condition[:n_samples-1])
                #     data_tr=torch.zeros_like(data[:n_samples-1])
                #     l=0
                #     for k in range(batch_size):
                #         if not condition[k].equal(con_zero):
                #             condition_tr[l] = condition[k]
                #             data_tr[l]=data[k]
                #             l+=1
                #             if l>=3:
                #                 break
                #     condition = torch.cat([condition_tr,condition_t],dim=0)
                #     data=torch.cat([data_tr.cpu(),data_t],dim=0)


                if data_type == "numpy":
                    data = tensor2png(data,n_samples)
                    img = latent_sample(model,autoencoder,condition,z_channels=z_channels,latent_size=image_size//8).cpu()
                    img = tensor2png(img,n_samples)
                    img = torch.cat([data,img],dim = 0).view(n_samples*2,3,512,512)
                
                else:
                    img = latent_sample(model,autoencoder,condition,z_channels=z_channels,latent_size=image_size//8).cpu()
                    img = torch.cat([data,img],dim = 0).view(n_samples*2,3,512,512)

                if condition == None:
                    save_image(img, os.path.join(save_path, 'uc_img/{}.png'.format(epoch+1)), nrow=n_samples)
                else:
                    save_image(img, os.path.join(save_path, 'c_img/{}.png'.format(epoch+1)), nrow=n_samples)
                torch.save(model.eps_model.state_dict(),os.path.join(save_path, 'model/{}.pkl'.format(epoch+1)).format(epoch+1))
                print("successfully saved")
        i+=1           
       


def model_sample(test_data,model,autoencoder,output_file):
    if not os.path.exists(os.path.join(output_file,"condition")):
        os.makedirs(os.path.join(output_file,"condition"))
    if not os.path.exists(os.path.join(output_file,"uncondition")):
        os.makedirs(os.path.join(output_file,"uncondition"))

    i = 0
    for (pop, dem, lan),data in test_data:
        # if i<=636:
        #     i+=1
        #     continue
        condition = torch.cat([pop,dem,lan],dim=1).to(device)
        if is_numpy:
            data = tensor2png(data,1)
        # save_image(data, os.path.join(os.path.join(os.path.join(output_file,"condition")), '{}_orig.png'.format(i)), nrow=n_samples)
        for j in range(4):
            begin_time = time.time()
            img = latent_sample(model,autoencoder,condition,4,64)
            if is_numpy:
                img = tensor2png(img,1)
            save_image(img, os.path.join(os.path.join(output_file,"condition"), '{}_{}.png'.format(i,j)), nrow=n_samples)
            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1


def main():
    data_type = "numpy" if is_numpy else "RGB"
    loaders = Loaders(batch_size,data_type=data_type)
    train_data=loaders.train_loader
    test_data=loaders.test_loader
    model,autoencoder = init_model()
    print("model is prepared")

    #model_paint(test_data,model,autoencoder)
    # for i in range(2000):
    #     img = latent_sample(model,autoencoder,None,4,64)
    #     if is_numpy:
    #         img = tensor2png(img,1)
    #     save_image(img, os.path.join("//home/lli/refine_ddpm/paint/resnet_RGB/uncondition", '{}.png'.format(i)), nrow=n_samples)
    #     print(i)


    if config.sample_or_train=='sample':
        model_sample(test_data,model,autoencoder,config.sample_path)

    else:
        iterator = iter(test_data)
        optimizer = torch.optim.Adam(model.eps_model.parameters(), lr=learning_rate)
        for epoch in range(0,epochs):
        #     # Train the model
            print("epoch: %i/%i" % (int(epoch), int(epochs)))
            train_encode(model,autoencoder,train_data, optimizer,epoch,iterator,save_path=config.train_path,data_type="numpy")

#
if __name__ == '__main__':
    main()




