import time
import cv2
import os
import torch
import random
import config
import numpy as np
import torch.nn.functional as F
from dataloader import Loaders
from torchvision.utils import save_image

batch_size = config.batch_size
device = config.device
is_numpy = True
n_steps: int = 1000

palette = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0]]

def mask_to_onehot(mask, palette):
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

def paint(model,autoencoder,condition,orig,mask):
    begin_time = time.time()
    with torch.no_grad():
        orig_latent = autoencoder.encode(orig)
        x = torch.randn((1,4,512//8,512//8),device = condition.device)
        if config.sample_type=='ddpm':
            for t in range(n_steps-1, -1, -1):
                ts = x.new_full((1,),fill_value=t, dtype=torch.long)
                x_fg = model.p_sample(x,ts,condition)
                x_bg = model.q_sample(orig_latent,ts)
                x = x_bg * mask + (1-mask)*x_fg
        elif config.sample_type=='ddim':      
            x=model.paint_ddim(x,condition,orig_latent,mask)
        x = autoencoder.decode(x)

    end_time = time.time()
    print(f"total_time_cost = {end_time-begin_time}")
    return x

def random_mask(test_data,data_path,save_path):
    i = 0
    if not os.path.exists(os.path.join(save_path,"all")):
        os.makedirs(os.path.join(save_path,"all"))   
    if not os.path.exists(os.path.join(save_path,"mask")):
        os.makedirs(os.path.join(save_path,"mask"))  
    if not os.path.exists(os.path.join(save_path,"img_mask")):
        os.makedirs(os.path.join(save_path,"img_mask"))  
    for (pop, dem, lan),d in test_data:
        #print(d.shape)
        # if i>1:
        #     break
        for j in range(1):
            begin_time = time.time()
            datafile_path=os.path.join(data_path,'{}_{}.png'.format(i,j))
            #print(datafile_path)
            # img1=cv2.imread(datafile_path)  
            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            # img_seg=mask_to_onehot(img1, palette)
            # img_seg = img_seg.astype(np.float32)
            # img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=d
            img_tensor = img_tensor.to(device)

            data=img_tensor

            orig_mask = torch.ones(512,512).to(device)

            random_box_num = random.randint(50,70)

            bs,channels,height,width = data.size()
            for _ in range(random_box_num):
                random_box_x,random_box_y = random.randint(10,30),random.randint(10,30)
                center_x,center_y = random.randint(1,height-random_box_x-1),random.randint(1,height-random_box_y-1)
                #print(center_x,center_y)
                orig_mask[center_x:center_x+random_box_x,center_y:center_y+random_box_y] = 0
            
            img_mask = orig_mask.unsqueeze(0).unsqueeze(0).repeat(1,5,1,1)
            masked_img = img_mask * data

            data_one_channel = tensor2one_channel(data,1).to(device)
            orig_mask_not = torch.ones_like(orig_mask)-orig_mask

            mask=orig_mask_not*data_one_channel
            #print(mask.shape)

            condition = torch.cat([pop,dem,lan],dim=1).to(device)

            latent_mask = F.interpolate(orig_mask.unsqueeze(0).unsqueeze(0).repeat(1,4,1,1), (64,64))

            img_all = torch.cat([masked_img,data],dim = 0)

            if is_numpy:
                img_mask = tensor2png(masked_img,1)
            save_image(img_mask, os.path.join(save_path, 'img_mask','{}.png'.format(i,j)), nrow=1)

            if is_numpy:
                img_all = tensor2png(img_all,2)
            save_image(img_all, os.path.join(save_path, 'all','{}.png'.format(i)), nrow=2)

            save_image(mask, os.path.join(os.path.join(save_path,"mask"), '{}.png'.format(i)), nrow=1,cmap='gary')
            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1    



if __name__ == '__main__':
    data_type = "numpy" if is_numpy else "RGB"
    loaders = Loaders(batch_size,data_type=data_type)
    train_data=loaders.train_loader
    test_data=loaders.test_loader

    data_path='/home/sadong/refine_ddpm/results/mse_spade/condition'    
    save_path='/home/sadong/refine_ddpm/paint/mask_test'
    random_mask(test_data,data_path,save_path)