from scipy import linalg
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
#from pytorch_fid import fid_score
import cv2 as cv
import glob
import os
import torch.nn as nn
# 构建 InceptionV3 编码器
from torchvision import models
from scipy.linalg import sqrtm
import numpy as np
import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from cleanfid import fid
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import evalution
import math

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class InceptionEncoder(nn.Module):
    def __init__(self):
        super(InceptionEncoder, self).__init__()
        self.inception = models.inception_v3(pretrained=False, aux_logits=False)
        encoder = list(self.inception.children())[:-10]
        self.encoder_layers = torch.nn.Sequential(*encoder)
        self.bn = nn.BatchNorm2d(768)
        self.active = nn.ReLU()
        
    def forward(self, x):
        features = self.encoder_layers(x)
        features = self.bn(features)
        features = self.active(features)
        return features

# 构建解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            UpNet(512,512),
            UpNet(512,256),
            ResidualBlock(256,256),
            ResidualBlock(256,128),
            UpNet(128,64),
            ResidualBlock(64,64),
            UpNet(64,64),
            nn.GroupNorm(16,64),
            Swish(),
            nn.Conv2d(64, 3, kernel_size=(3, 3), padding=(1, 1)),
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
class InceptionAutoencoder(nn.Module):
    def __init__(self):
        super(InceptionAutoencoder, self).__init__()
        self.encoder = InceptionEncoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        encoded_features = self.encoder(x)
        reconstructed_image = self.decoder(encoded_features)
        return reconstructed_image


class ImageDateset(Dataset):
    def __init__(self,img_path) -> None:
        super().__init__()
        self.png_file_paths = glob.glob(os.path.join(img_path, "*.png"))
        # self.png_file_paths = glob.glob(f'{img_path}/**[0123].png', recursive=True)
        self.length = len(self.png_file_paths) 

    def __getitem__(self, index):
        imgs = self.png_file_paths[index]
        imgs = cv.imread(imgs)
        imgs = transforms.ToTensor()(imgs)
        return imgs
    
    def __len__(self):
        return len(self.png_file_paths)    

def get_activations(dataloader, model, dims=2048):
    model.eval()
    pred_arr = np.empty((dataloader.dataset.length, dims))
    start_idx = 0
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr

def calculate_activation_statistics(dataloader, model, dims=2048):
    act = get_activations(dataloader, model, dims)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)
    
def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


device = torch.device("cuda:0")

def cal_fid(gen_folder,real_folder):
    dataset_real = ImageDateset(real_folder)
    dataset_fake = ImageDateset(gen_folder)
    dataloader_real = DataLoader(dataset_real, batch_size=4,shuffle = True,drop_last=True)
    dataloader_generated = DataLoader(dataset_fake, batch_size=4, shuffle = True,drop_last=True)
    model = InceptionAutoencoder()
    model.load_state_dict(torch.load("/home/sadong/refine_ddpm/AE/epoch35.pkl"))
    model = model.encoder
    model = model.to(device)
    feat1 = get_activations(dataloader_real,model,dims = 768)
    feat2 = get_activations(dataloader_generated,model,dims = 768)
    mu1 = np.mean(feat1, axis=0)
    mu2 = np.mean(feat2, axis=0)
    sigma1 = np.cov(feat1, rowvar=False)
    sigma2 = np.cov(feat2, rowvar=False)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(fid_value)

def cal_kid(gen_folder,real_folder):
    dataset_real = ImageDateset(real_folder)
    dataset_fake = ImageDateset(gen_folder)
    dataloader_real = DataLoader(dataset_real, batch_size=4,shuffle = True,drop_last=True)
    dataloader_generated = DataLoader(dataset_fake, batch_size=4, shuffle = True,drop_last=True)
    model = InceptionAutoencoder()
    model.load_state_dict(torch.load("/home/sadong/refine_ddpm/AE/epoch35.pkl"))
    model = model.encoder
    model = model.to(device)
    feat1 = get_activations(dataloader_real,model,dims = 768)
    feat2 = get_activations(dataloader_generated,model,dims = 768)
    kid_value = kernel_distance(feat1,feat2)
    print(kid_value)

def cal_ssim(gen_folder,real_folder,begin_index = 0):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    for i in range(744):
        real_img = cv.imread(f"{real_folder}/{i}.png")
        real_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])(real_img).unsqueeze(0).to(device)
        for j in range(begin_index,begin_index + 4):
            fake_img = cv.imread(f"{gen_folder}/{i+begin_index}_{j}.png")
            fake_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])(fake_img).unsqueeze(0).to(device)
            ssim.update(fake_img,real_img)
    ssim_score = ssim.compute()
    print(ssim_score)

def cal_is_score(gen_folder):
    model_is = InceptionScore(normalize=True).to(device)
    gen_imgs = os.listdir(gen_folder)
    for gen_img in gen_imgs:
        gen_img = gen_folder + "/" + gen_img
        gen_img = cv.imread(gen_img)
        gen_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])(gen_img).unsqueeze(0).to(device)
        model_is.update(gen_img)
    is_score, is_std = model_is.compute()
    print(is_score, is_std)


def cal_lpips(gen_folder,real_folder,begin_index = 0):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    for i in range(744):
        real_img = cv.imread(f"{real_folder}/{i}.png")
        real_img = transforms.Compose([transforms.ToTensor()])(real_img).unsqueeze(0).to(device)
        for j in range(begin_index,begin_index + 4):
            fake_img = cv.imread(f"{gen_folder}/{i+begin_index}_{j}.png")
            fake_img = transforms.Compose([transforms.ToTensor()])(fake_img).unsqueeze(0).to(device)
            lpips.update(fake_img,real_img)
    lpips_score = lpips.compute()
    print(lpips_score)

def cal_cl(gen_folder):
    gen_imgs = os.listdir(gen_folder)
    #233号、743号数据有问题，跳过
    # gen_imgs.remove("233_0.png")
    # gen_imgs.remove("233_1.png")
    # gen_imgs.remove("233_2.png")
    # gen_imgs.remove("233_3.png")    
    # gen_imgs.remove("743_0.png")
    # gen_imgs.remove("743_1.png")
    # gen_imgs.remove("743_2.png")
    # gen_imgs.remove("743_3.png")    
    #num=len(gen_imgs)
    cl=0
    num=0
    remove=["233","241","264","265","271","272","273","337","340","348","349","350","457","485","743"]
    #remove=[]
    for gen_img in gen_imgs:
        for no in remove:
            if no in gen_img:
                break
        else:
            num+=1
            gen_img = gen_folder + "/" + gen_img
            # if gen_img!="/home/sadong/refine_ddpm/results/focal_spadecross/condition/272_0.png":
            #     continue
            print("gen_img:",gen_img)
            gen_img = cv.imread(gen_img,2)
            gen_img=evalution.centerline_extraction(gen_img)
            _, _,other=evalution.raster2vector_norefine(gen_img)
            cl+=other[2]

    cl=cl/num
    print(cl)
    return cl
        
def cal_rls(real_folder,gen_folder,begin_index = 0):
    sum=0
    #remove=[233,241,264,265,271,272,273,337,340,348,349,350,457,485,743]
    remove=[]
    for i in range(744):
        if i in remove:
            continue
        real_img = cv.imread(f"{real_folder}/{i}.png",2)
        real_img=evalution.centerline_extraction(real_img)
        _, _,real_other=evalution.raster2vector(real_img)
        real_total_roads=real_other[1]
        print(i)
        for j in range(begin_index,begin_index + 4):
            gen_roads=0
            fake_img = cv.imread(f"{gen_folder}/{i+begin_index}_{j}.png",2)
            fake_img=evalution.centerline_extraction(fake_img)
            _, _,fake_other=evalution.raster2vector(fake_img)
            gen_roads+=fake_other[1]
            gen_total_roads=gen_roads
            sum+=abs(gen_total_roads-real_total_roads)
            #print(gen_total_roads,real_total_roads)
            #print(sum/(j+1))
            rls=sum/(i*4+j+1)
            print(rls)
    rls=sum/((744-len(remove))*4)
    print(rls)
    return rls
            
def cal_conv(gen_folder):
    gen_imgs = os.listdir(gen_folder)
    num=len(gen_imgs)
    covn=0
    i=0
    num=0
    #remove=["233","241","264","265","271","272","273","337","340","348","349","350","457","485","743"]
    remove=[]
    for gen_img in gen_imgs:
        for no in remove:
            if no in gen_img:
                break
        else:
            num+=1
            i+=1
            print(i)
            gen_img = gen_folder + "/" + gen_img
            gen_img = cv.imread(gen_img,2)
            gen_img=evalution.centerline_extraction(gen_img)
            _, _,other=evalution.raster2vector_norefine(gen_img)
            covn+=other[5]
    covn=covn/num
    print("covn:",covn)
    return covn

if __name__ == '__main__':
    #gen_folder = "/home/sadong/refine_ddpm/paint/spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900_refine-spadesam_2000_removesmall80_detectormask/result"
    gen_folder = "/home/sadong/refine_ddpm/results/focal_spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900/condition"
    #gen_folder = "/home/sadong/color_DDPM/sample"
    #gen_folder ="/home/sadong/refine_ddpm/paint/spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900_maskvae/result"
    #gen_folder = "/home/sadong/refine_ddpm/results/focal_spadecross_2500/condition"
    #gen_folder = "/home/sadong/refine_ddpm/paint/spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900_refine-spadesam_2000_removesmall80_detectormask/result_removesmall20"
    # real_folder = "/home/lli/control_npy_DDDPM/data/fake_data/class1"
    real_folder = "/home/sadong/refine_ddpm/test_orig"
    #gen_folder=r"D:\work\road\eval\gen\focal_ddim\condition"
    #real_folder=r"D:\work\road\eval\real\test_orig"
    #cal_kid(gen_folder,real_folder)

    # cal_fid(gen_folder,real_folder)

    cal_cl(gen_folder)
    #cal_rls(real_folder,gen_folder)
    #cal_conv(gen_folder)
    #cal_ssim(gen_folder,real_folder)
    # cal_is_score(gen_folder)
    #cal_lpips(gen_folder,real_folder,begin_index = 0)
    #cal_ssim(gen_folder,real_folder,begin_index = 0)
    #datasets_real=ImageDateset(real_folder)
