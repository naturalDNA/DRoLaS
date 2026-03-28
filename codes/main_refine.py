import config
from mask_vae import Autoencoder
import torch
import os
import time
import cv2
import numpy as np
from torchvision.utils import save_image
from dataloader import Loaders
from diffusion_refine import *
from RefineSpadeSamUnet import *
import random
from skimage import morphology
import torchvision.transforms as transforms
import mask_detector_unet

is_numpy = True

device = config.device
image_channels: int = 5 if is_numpy else 3
z_channels = 4
if config.sample_or_train=='train':
    n_samples = config.n_samples
else:
    n_samples=1
    config.n_samples=n_samples
batch_size=config.batch_size
n_channels = config.n_channels
channel_multipliers: List[int] = [1, 2, 2, 4]
is_attention: List[int] = [False, False, True, True]
n_steps: int = 1000
image_size: int = 512
learning_rate: float = 2e-5
epochs: int = 3000


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

def init_model():
    autoencoder = Autoencoder(img_channels=image_channels,latent_channels=z_channels,masked=True).to(device=device)

    if config.vae_path:
        autoencoder.load_state_dict(torch.load(config.vae_path,map_location=device))

    eps_model = UNet(
        image_channels=z_channels,
        n_channels=n_channels,
        ch_mults=channel_multipliers,
        is_attn=is_attention,
    ).to(device)

    if config.unet_path:
        eps_model.load_state_dict(torch.load(config.unet_path,map_location=device))

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=n_steps,
        device = device
    )

    for name, parameter in autoencoder.named_parameters():
        parameter.requires_grad = False

    return diffusion,autoencoder

def init_model_vae():
    autoencoder = Autoencoder(img_channels=image_channels,latent_channels=z_channels,masked=True).to(device=device)

    vae_path="/home/sadong/refine_ddpm/resnet_VAE/mask_cuda0/model/epoch100.pkl"
    autoencoder.load_state_dict(torch.load(vae_path,map_location=device))

    for name, parameter in autoencoder.named_parameters():
        parameter.requires_grad = False

    return autoencoder

def refine_vae(test_data,autoencoder,data_path,save_path):
    if not os.path.exists(os.path.join(save_path, 'result')):
        os.makedirs(os.path.join(save_path, 'result'))
    if not os.path.exists(os.path.join(save_path, 'all')):
        os.makedirs(os.path.join(save_path, 'all'))   
    i = 0
    for (pop, dem, lan),d in test_data:
        for j in range(4):
            begin_time = time.time()
            datafile_path=os.path.join(data_path,'{}_{}.png'.format(i,j))
            #print(datafile_path)
            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            data=img_tensor
            with torch.no_grad():
                img = autoencoder(data)


            img_all = torch.cat([img, data],dim = 0)

            if is_numpy:
                img = tensor2png(img,1)
            save_image(img, os.path.join(save_path, 'result','{}_{}.png'.format(i,j)), nrow=3)

            if is_numpy:
                img_all = tensor2png(img_all,2)
            save_image(img_all, os.path.join(save_path, 'all','{}_{}.png'.format(i,j)), nrow=3,padding=10,pad_value=128)
            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1    

def latent_sample(model,autoencoder,data_break,condition,z_channels,latent_size):
    with torch.no_grad():
        data_break=data_break.to(device)
        x_break = autoencoder.encode(data_break)
        x = torch.randn([n_samples, z_channels, latent_size, latent_size],device=device)
        if config.sample_type=='ddpm':
            for t in range(config.n_steps-1,-1,-1):
                x = model.p_sample(x,x_break, x.new_full((config.n_samples,),fill_value=t, dtype=torch.long),condition=condition)
        elif config.sample_type=='ddim':
            if config.class_free:
                x=model.classifire_p_sample_ddim(x,x_break,condition=condition,scale=1.5)
            else:
                x=model.p_sample_ddim(x,x_break,condition=condition)
        return autoencoder.decode(x)
    
#对原始路网做细化
def model_sample(test_data,model,autoencoder,output_file,data_path):
    if not os.path.exists(os.path.join(output_file,"all")):
        os.makedirs(os.path.join(output_file,"all"))
    if not os.path.exists(os.path.join(output_file,"result")):
        os.makedirs(os.path.join(output_file,"result"))

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

            datafile_path=os.path.join(data_path,'{}_{}.png'.format(i,j))
            #print(datafile_path)
            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            data_break=img_tensor

            img = latent_sample(model,autoencoder,data_break,condition,4,64)
            img_all = torch.cat([img,data_break],dim = 0)
            if is_numpy:
                img = tensor2png(img,1)
            if is_numpy:
                img_all = tensor2png(img_all,2)
            save_image(img, os.path.join(os.path.join(output_file,"result_womask"), '{}_{}.png'.format(i,j)), nrow=n_samples)
            save_image(img_all, os.path.join(os.path.join(output_file,"all"), 'refine_break_{}_{}.png'.format(i,j)), nrow=4,padding=10,pad_value=128)

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1

#对去除小块的结果做细化
def model_sample_removesmall(test_data,model,autoencoder,output_file,data_path):
    if not os.path.exists(os.path.join(output_file,"all")):
        os.makedirs(os.path.join(output_file,"all"))
    if not os.path.exists(os.path.join(output_file,"result_womask")):
        os.makedirs(os.path.join(output_file,"result_womask"))

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

            orig_path=os.path.join(data_path,'{}_{}.png'.format(i,j))
            img1=cv2.imread(orig_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            data_orig=img_tensor            

            datafile_path=os.path.join(output_file,'masked','{}_{}.png'.format(i,j))
            #print(datafile_path)
            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            data_break=img_tensor

            img = latent_sample(model,autoencoder,data_break,condition,4,64)
            img_all = torch.cat([img,data_break,data_orig],dim = 0)
            if is_numpy:
                img = tensor2png(img,1)
            if is_numpy:
                img_all = tensor2png(img_all,3)
            save_image(img, os.path.join(os.path.join(output_file,"result_womask"), '{}_{}.png'.format(i,j)), nrow=n_samples)
            save_image(img_all, os.path.join(os.path.join(output_file,"all"), 'refine_mask_break{}_{}.png'.format(i,j)), nrow=4,padding=10,pad_value=128)

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1

#对原始结果去除小块
def remove_small(data_path,save_path,test_data):
    if not os.path.exists(os.path.join(save_path,"masked")):
        os.makedirs(os.path.join(save_path,"masked"))

    i = 0
    for (pop, dem, lan),data in test_data:
        for j in range(4):
            begin_time = time.time()
            datafile_path=os.path.join(data_path,'{}_{}.png'.format(i,j))

            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            img_tensor = torch.squeeze(img_tensor,dim=0)

            img_npy = img_tensor.cpu().numpy()
            img_npy = img_npy > 0
            #print(region_npy)

            for k in range(5):
                img_npy_channelk = img_npy[k]
                img_npy_channelk = morphology.remove_small_objects(img_npy_channelk, 80)
                img_npy[k]=img_npy_channelk
            
            img_npy = img_npy.astype(np.float32)
            img_tensor = torch.from_numpy(img_npy)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            #print(img_tensor.shape)

            output_path=os.path.join(save_path,'masked','{}_{}.png'.format(i,j))
            img = tensor2png(img_tensor,1)
            save_image(img, output_path, nrow=1)

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1
    
#将细化后的结果再去除小块
def remove_small_final(data_path,save_path,test_data):
    if not os.path.exists(os.path.join(save_path,"result_removesmall")):
        os.makedirs(os.path.join(save_path,"result_removesmall"))

    i = 0
    for (pop, dem, lan),data in test_data:
        for j in range(4):
            begin_time = time.time()
            datafile_path=os.path.join(data_path,'{}_{}.png'.format(i,j))

            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            img_tensor = torch.squeeze(img_tensor,dim=0)

            img_npy = img_tensor.cpu().numpy()
            img_npy = img_npy > 0
            #print(region_npy)

            for k in range(5):
                img_npy_channelk = img_npy[k]
                img_npy_channelk = morphology.remove_small_objects(img_npy_channelk, 30)
                img_npy[k]=img_npy_channelk
            
            img_npy = img_npy.astype(np.float32)
            img_tensor = torch.from_numpy(img_npy)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            #print(img_tensor.shape)

            output_path=os.path.join(save_path,'result_removesmall','{}_{}.png'.format(i,j))
            img = tensor2png(img_tensor,1)
            save_image(img, output_path, nrow=1)

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1

def train(model,autoencoder,dataIter, optimizer,epoch,iterator,save_path,data_type):
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

        data_break=data.clone()
        bs,channels,height,width = data_break.size()
        #random_box_x,random_box_y = random.randint(10,100),random.randint(10,100)
        random_box_num = random.randint(50,70)
        for j in range(bs):
            for _ in range(random_box_num):
                random_box_x,random_box_y = random.randint(10,30),random.randint(10,30)
                center_x,center_y = random.randint(1,height-random_box_x-1),random.randint(1,height-random_box_y-1)
                #center_x,center_y = random.randint(random_box_x+1,height-random_box_x-1),random.randint(random_box_y+1,height-random_box_y-1)
                one_hot_tensor = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32,device=data_break.device).view(1, 5, 1, 1).expand(1,5,random_box_x,random_box_y)
                data_break[j,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = one_hot_tensor

        code = autoencoder.encode(data)        
        code_break = autoencoder.encode(data_break)


        optimizer.zero_grad()
        loss = model.loss(code,code_break,condition=condition)
        #print("now_epoch={},now_loss={}".format(epoch,loss))
        databar.set_description('now_epoch=%d,loss=%.3f' % (epoch+1,loss))

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if (epoch+1)%5==0 and epoch!=0 and i==t:
                print("begin generate")
                (pop_t, dem_t, lan_t),data_t =next(iterator)
                condition_t = torch.cat([pop_t,dem_t,lan_t],dim=1).to(device)
                data=data[:n_samples-1].cpu()
                data=torch.cat([data,data_t],dim=0)

                data_break=data.clone()
                bs,channels,height,width = data_break.size()
                #random_box_x,random_box_y = random.randint(10,100),random.randint(10,100)
                random_box_num = random.randint(50,70)
                for j in range(bs):
                    for _ in range(random_box_num):
                        random_box_x,random_box_y = random.randint(10,30),random.randint(10,30)
                        center_x,center_y = random.randint(1,height-random_box_x-1),random.randint(1,height-random_box_y-1)
                        #center_x,center_y = random.randint(random_box_x+1,height-random_box_x-1),random.randint(random_box_y+1,height-random_box_y-1)
                        one_hot_tensor = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32,device=data_break.device).view(1, 5, 1, 1).expand(1,5,random_box_x,random_box_y)
                        data_break[j,:,center_x:center_x+random_box_x,center_y:center_y+random_box_y] = one_hot_tensor


                if condition !=None:
                    condition = condition[:n_samples-1]
                    condition = torch.cat([condition,condition_t],dim=0)
                if data_type == "numpy":
                    data = tensor2png(data,n_samples)
                    img = latent_sample(model,autoencoder,data_break,condition,z_channels=z_channels,latent_size=image_size//8).cpu()
                    img = tensor2png(img,n_samples)
                    data_break = tensor2png(data_break,n_samples)
                    img = torch.cat([data,data_break,img],dim = 0).view(n_samples*3,3,512,512)
                
                else:
                    img = latent_sample(model,autoencoder,data_break,condition,z_channels=z_channels,latent_size=image_size//8).cpu()
                    img = torch.cat([data,img],dim = 0).view(n_samples*3,3,512,512)

                if condition == None:
                    save_image(img, os.path.join(save_path, 'uc_img/{}.png'.format(epoch+1)), nrow=n_samples)
                else:
                    save_image(img, os.path.join(save_path, 'c_img/{}.png'.format(epoch+1)), nrow=n_samples,padding=10,pad_value=128)
                torch.save(model.eps_model.state_dict(),os.path.join(save_path, 'model/{}.pkl'.format(epoch+1)).format(epoch+1))
                print("successfully saved")
        i+=1    

#得到去除小块后的mask
def make_mask(data_path,save_path,test_data):
    if not os.path.exists(os.path.join(save_path,"mask_removesmall")):
        os.makedirs(os.path.join(save_path,"mask_removesmall"))

    i = 0
    for (pop, dem, lan),data in test_data:
        for j in range(4):
            begin_time = time.time()
            datafile_path=os.path.join(data_path,'{}_{}.png'.format(i,j))

            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            img_one_channel1=tensor2one_channel(img_tensor,1)

            datafile_path=os.path.join(save_path,'masked','{}_{}.png'.format(i,j))

            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)           

            img_one_channel2=tensor2one_channel(img_tensor,1) 

            img_mask=img_one_channel1-img_one_channel2


            output_path=os.path.join(save_path,'mask_removesmall','{}_{}.png'.format(i,j))
            save_image(img_mask, output_path, nrow=1,cmap='gary')

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1

#将两个mask合并
def make_prob_mask(mask_path1,mask_path2,save_path,test_data):
    if not os.path.exists(os.path.join(save_path,"mask")):
        os.makedirs(os.path.join(save_path,"mask"))

    i = 0
    for (pop, dem, lan),data in test_data:
        for j in range(4):
            begin_time = time.time()
            mask1_path=os.path.join(mask_path1,'mask_detector_dilate2','{}_{}.png'.format(i,j))
            mask1=cv2.imread(mask1_path,cv2.IMREAD_GRAYSCALE)  
            #print(mask1.shape)
            transf = transforms.ToTensor()
            mask1_tensor = transf(mask1)

            mask2_path=os.path.join(mask_path2,'mask_removesmall_dilate2','{}_{}.png'.format(i,j))
            mask2=cv2.imread(mask2_path,cv2.IMREAD_GRAYSCALE)  
            #print(mask1.shape)
            transf = transforms.ToTensor()
            mask2_tensor = transf(mask2)

            mask=mask1_tensor+mask2_tensor
            mask=torch.where(mask>0,torch.ones_like(mask),torch.zeros_like(mask))
            mask=mask.unsqueeze(0)

            output_path=os.path.join(save_path,'mask','{}_{}.png'.format(i,j))
            save_image(mask, output_path, nrow=1,cmap='gary')

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1

#获得断路检测器检测的mask
def make_detector_mask(data_path,save_path,test_data):
    if not os.path.exists(os.path.join(save_path,"mask_detector")):
        os.makedirs(os.path.join(save_path,"mask_detector"))

    detector_path="/home/sadong/refine_ddpm/mask_detector/focal_cuda1_unet/model/epoch100.pkl"
    detector = mask_detector_unet.Autoencoder(img_channels=5,latent_channels=16,masked=True,loss="Focal").to(device)
    detector.load_state_dict(torch.load(detector_path,map_location=device))
    for name, parameter in detector.named_parameters():
        parameter.requires_grad = False

    i = 0
    for (pop, dem, lan),data in test_data:
        for j in range(4):
            begin_time = time.time()
            datafile_path=os.path.join(data_path,'{}_{}.png'.format(i,j))

            img1=cv2.imread(datafile_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                mask = detector(img_tensor)

            mask_onechannel = two2one_channel(mask,1)
        
            output_path=os.path.join(save_path,'mask_detector','{}_{}.png'.format(i,j))
            save_image(mask_onechannel, output_path, nrow=1,cmap='gary')

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1

#获得加mask后的最终结果
def make_mask_result(data_path,save_path,test_data):
    if not os.path.exists(os.path.join(save_path,"result")):
        os.makedirs(os.path.join(save_path,"result"))

    if not os.path.exists(os.path.join(save_path,"all_mask")):
        os.makedirs(os.path.join(save_path,"all_mask"))

    i = 0
    for (pop, dem, lan),data in test_data:
        for j in range(4):
            begin_time = time.time()
            mask_path=os.path.join(save_path,'mask','{}_{}.png'.format(i,j))
            mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)  
            #print(mask1.shape)
            transf = transforms.ToTensor()
            mask_tensor = transf(mask)
            mask_tensor = mask_tensor.to(device)                 
            mask=mask_tensor
            mask_not=torch.ones_like(mask)-mask

            orig_path=os.path.join(data_path,'{}_{}.png'.format(i,j))
            img1=cv2.imread(orig_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)     
            break_tensor=img_tensor

            orig_path=os.path.join(save_path,'masked','{}_{}.png'.format(i,j))
            img1=cv2.imread(orig_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)     
            orig_tensor=img_tensor
            
            orig_path=os.path.join(save_path,'result_womask','{}_{}.png'.format(i,j))
            img1=cv2.imread(orig_path)  
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg=mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2,0,1)
            img_tensor=img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)     
            refine_tensor=img_tensor        

            orig_mask=torch.ones_like(orig_tensor)  

            result=torch.ones_like(orig_tensor)
            for k in range(5):
                result[:,k,:,:]=mask*refine_tensor[:,k,:,:]+mask_not*orig_tensor[:,k,:,:]
                orig_mask[:,k,:,:]=mask_not*orig_tensor[:,k,:,:]

            img_all = torch.cat([result,orig_mask,break_tensor],dim = 0)            
            result = tensor2png(result,1)
            result_all=tensor2png(img_all,3)

            save_image(result, os.path.join(os.path.join(save_path,"result"), '{}_{}.png'.format(i,j)), nrow=n_samples)
            save_image(result_all, os.path.join(os.path.join(save_path,"all_mask"), 'refine_mask_break{}_{}.png'.format(i,j)), nrow=4,padding=10,pad_value=128)

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1

#对mask做膨胀操作   
def dilate_mask(data_path,save_path,test_data):
    if not os.path.exists(os.path.join(save_path,"mask_removesmall_dilate2")):
        os.makedirs(os.path.join(save_path,"mask_removesmall_dilate2"))
    i = 0
    for (pop, dem, lan),data in test_data:
        for j in range(4):
            begin_time = time.time()
            mask_path=os.path.join(save_path,'mask_removesmall','{}_{}.png'.format(i,j))
            mask_npy=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)  

            kernel = np.ones((3, 3), np.float32)
            mask_npy = cv2.dilate(mask_npy, kernel, iterations = 2)

            #print(mask1.shape)
            transf = transforms.ToTensor()
            mask_tensor = transf(mask_npy)
            mask_tensor = mask_tensor.to(device)   

            output_path=os.path.join(save_path,'mask_removesmall_dilate2','{}_{}.png'.format(i,j))
            save_image(mask_tensor, output_path, nrow=1,cmap='gary')

            end_time = time.time()
            print("i = {}, total_time = {}".format(i, end_time-begin_time))
        i += 1
    
# def main():
#     data_type = "numpy" if is_numpy else "RGB"
#     loaders = Loaders(batch_size,data_type=data_type)
#     train_data=loaders.train_loader
#     test_data=loaders.test_loader
#     model,autoencoder = init_model()
#     print("model is prepared")

#     #model_sample(test_data,model,autoencoder,config.sample_path)

#     if config.sample_or_train=='sample':
#         model_sample_removesmall(test_data,model,autoencoder,config.sample_path,config.data_path)

#     else:
#         iterator = iter(test_data)
#         optimizer = torch.optim.Adam(model.eps_model.parameters(), lr=learning_rate)
#         for epoch in range(0,epochs):
#             #     # Train the model
#             print("epoch: %i/%i" % (int(epoch), int(epochs)))
#             train(model,autoencoder,train_data, optimizer,epoch,iterator,save_path=config.train_path,data_type="numpy")




# if __name__ == '__main__':
#     autoencoder = init_model_vae()

#     data_type = "numpy" if is_numpy else "RGB"
#     loaders = Loaders(batch_size,data_type=data_type)
#     train_data=loaders.train_loader
#     test_data=loaders.test_loader
#     # # # autoencoder = init_model()

#     #data_path='/home/sadong/refine_ddpm/paint/spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900_refine-spadesam_2000_removesmall80_detectormask/result'
#     data_path='/home/sadong/refine_ddpm/results/focal_spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900/condition'
#     #data_path='/home/sadong/refine_ddpm/paint/focal_spadesam_prob0.45_minsize10_dilate/sample'   
#     #mask1_path='/home/sadong/refine_ddpm/paint/focal_spadesam_prob0.45_minsize10_dilate'
#     # mask2_path='/home/sadong/refine_ddpm/paint/spadesam_refine_2000_removesmall'
#     save_path='/home/sadong/refine_ddpm/paint/spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900_maskvae'
#     #refine_vae(test_data,autoencoder,data_path,save_path)
#     main()
#     # remove_small(data_path,save_path,test_data)
#     # make_mask(data_path,save_path,test_data)
#     # make_prob_mask(save_path,save_path,save_path,test_data)
#     # make_mask_result(data_path,save_path,test_data)
#     # remove_small_final(data_path,save_path,test_data)
#     # make_detector_mask(data_path,save_path,test_data)
#     # dilate_mask(data_path,save_path,test_data)


def dilate_detector_mask(save_path, test_data):
    """对检测器掩码进行膨胀，扩大修复区域"""
    if not os.path.exists(os.path.join(save_path, "mask_detector_dilate2")):
        os.makedirs(os.path.join(save_path, "mask_detector_dilate2"))
    
    i = 0
    for (pop, dem, lan), data in test_data:
        for j in range(4):
            mask_path = os.path.join(save_path, 'mask_detector', '{}_{}.png'.format(i, j))
            mask_npy = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            kernel = np.ones((3, 3), np.float32)
            mask_npy = cv2.dilate(mask_npy, kernel, iterations=2)
            
            transf = transforms.ToTensor()
            mask_tensor = transf(mask_npy)
            mask_tensor = mask_tensor.to(device)
            
            output_path = os.path.join(save_path, 'mask_detector_dilate2', '{}_{}.png'.format(i, j))
            save_image(mask_tensor, output_path, nrow=1)
        i += 1


def create_masked_image(data_path, save_path, test_data):
    """
    根据检测器掩码创建破损图像
    𝑥¯ = 𝑥˜ ⊙ (1 − 𝑚˜)
    将需要修复的区域置为背景
    """
    if not os.path.exists(os.path.join(save_path, "masked")):
        os.makedirs(os.path.join(save_path, "masked"))
    
    i = 0
    for (pop, dem, lan), data in test_data:
        for j in range(4):
            # 读取原始图像
            orig_path = os.path.join(data_path, '{}_{}.png'.format(i, j))
            img1 = cv2.imread(orig_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg = mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            # 读取检测器掩码
            mask_path = os.path.join(save_path, 'mask_detector_dilate2', '{}_{}.png'.format(i, j))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            transf = transforms.ToTensor()
            mask_tensor = transf(mask)
            mask_tensor = mask_tensor.unsqueeze(0).to(device)
            
            # 创建破损图像：将需要修复的区域（掩码为1的区域）置为背景
            masked_image = img_tensor.clone()
            for k in range(5):
                masked_image[:, k, :, :] = masked_image[:, k, :, :] * (1 - mask_tensor)
            
            # 保存破损图像
            masked_img = tensor2png(masked_image, 1)
            save_image(masked_img, os.path.join(save_path, 'masked', '{}_{}.png'.format(i, j)), nrow=1)
        i += 1


def refine_with_detector_mask(test_data, model, autoencoder, save_path, data_path):
    """
    使用检测器掩码进行细化修复
    读取破损图像，用扩散模型修复
    """
    if not os.path.exists(os.path.join(save_path, "result_womask")):
        os.makedirs(os.path.join(save_path, "result_womask"))
    
    i = 0
    for (pop, dem, lan), data in test_data:
        condition = torch.cat([pop, dem, lan], dim=1).to(device)
        
        for j in range(4):
            # 读取破损图像
            masked_path = os.path.join(save_path, 'masked', '{}_{}.png'.format(i, j))
            img1 = cv2.imread(masked_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg = mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            img_tensor = torch.from_numpy(img_seg).permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            data_break = img_tensor
            
            # 扩散模型修复
            img = latent_sample(model, autoencoder, data_break, condition, 4, 64)
            
            # 保存修复结果
            if is_numpy:
                img = tensor2png(img, 1)
            save_image(img, os.path.join(save_path, 'result_womask', '{}_{}.png'.format(i, j)), nrow=1)
            
            print("Refined: i={}, j={}".format(i, j))
        i += 1


def merge_with_mask(data_path, save_path, test_data):
    """
    图像融合：𝑥 = 𝑥ˆ ⊙ 𝑚˜ + 𝑥˜ ⊙ (1 − 𝑚˜)
    用检测器掩码融合修复图像和原始图像
    """
    if not os.path.exists(os.path.join(save_path, "final_result")):
        os.makedirs(os.path.join(save_path, "final_result"))
    
    i = 0
    for (pop, dem, lan), data in test_data:
        for j in range(4):
            # 读取原始图像
            orig_path = os.path.join(data_path, '{}_{}.png'.format(i, j))
            img1 = cv2.imread(orig_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg = mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            orig_tensor = torch.from_numpy(img_seg).permute(2, 0, 1)
            orig_tensor = orig_tensor.unsqueeze(0).to(device)
            
            # 读取修复图像
            refine_path = os.path.join(save_path, 'result_womask', '{}_{}.png'.format(i, j))
            img1 = cv2.imread(refine_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_seg = mask_to_onehot(img1, palette)
            img_seg = img_seg.astype(np.float32)
            refine_tensor = torch.from_numpy(img_seg).permute(2, 0, 1)
            refine_tensor = refine_tensor.unsqueeze(0).to(device)
            
            # 读取检测器掩码
            mask_path = os.path.join(save_path, 'mask_detector_dilate2', '{}_{}.png'.format(i, j))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            transf = transforms.ToTensor()
            mask_tensor = transf(mask)
            mask_tensor = mask_tensor.unsqueeze(0).to(device)
            
            # 图像融合
            result = torch.zeros_like(orig_tensor)
            for k in range(5):
                result[:, k, :, :] = refine_tensor[:, k, :, :] * mask_tensor + \
                                     orig_tensor[:, k, :, :] * (1 - mask_tensor)
            
            # 保存最终结果
            result_img = tensor2png(result, 1)
            save_image(result_img, os.path.join(save_path, 'final_result', '{}_{}.png'.format(i, j)), nrow=1)
        i += 1


def post_process(data_path, save_path, test_data, model=None, autoencoder=None):
    """
    后处理主函数 - 完整流程
    根据论文描述：
    1. 检测器生成掩码 𝑚˜
    2. 膨胀掩码
    3. 创建破损图像 𝑥¯ = 𝑥˜ ⊙ (1 − 𝑚˜)
    4. 扩散模型修复得到 𝑥ˆ
    5. 图像融合 𝑥 = 𝑥ˆ ⊙ 𝑚˜ + 𝑥˜ ⊙ (1 − 𝑚˜)
    """
    print("=" * 50)
    print("Starting connectivity refinement...")
    print("=" * 50)
    
    # 步骤1: 检测器生成掩码
    print("\n[Step 1] Detecting disconnections...")
    make_detector_mask(data_path, save_path, test_data)
    
    # 步骤2: 膨胀检测器掩码
    print("\n[Step 2] Dilating detector mask...")
    dilate_detector_mask(save_path, test_data)
    
    # 步骤3: 创建破损图像
    print("\n[Step 3] Creating masked image...")
    create_masked_image(data_path, save_path, test_data)
    
    # 步骤4: 扩散模型修复
    if model is not None and autoencoder is not None:
        print("\n[Step 4] Refining with diffusion model...")
        refine_with_detector_mask(test_data, model, autoencoder, save_path, data_path)
    
    # 步骤5: 图像融合
    print("\n[Step 5] Merging with mask...")
    merge_with_mask(data_path, save_path, test_data)
    
    print("\n" + "=" * 50)
    print("Connectivity refinement completed!")
    print("=" * 50)


if __name__ == '__main__':
    data_type = "numpy" if is_numpy else "RGB"
    loaders = Loaders(batch_size,data_type=data_type)
    train_data=loaders.train_loader
    test_data=loaders.test_loader
    model,autoencoder = init_model()
    print("model is prepared")

    # 配置路径
    data_path = config.data_path
    save_path = config.sample_path

    if config.sample_or_train=='sample':
        post_process(data_path, save_path, test_data, model=model, autoencoder=autoencoder)

    else:
        iterator = iter(test_data)
        optimizer = torch.optim.Adam(model.eps_model.parameters(), lr=learning_rate)
        for epoch in range(0,epochs):
            #     # Train the model
            print("epoch: %i/%i" % (int(epoch), int(epochs)))
            train(model,autoencoder,train_data, optimizer,epoch,iterator,save_path=config.train_path,data_type="numpy")
