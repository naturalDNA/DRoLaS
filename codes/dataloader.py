from numpy import source
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import os
import cv2
import matplotlib.pyplot as plt
import torch
import csv
import random
import numpy as np
import random


class RGBDataset(Dataset):
    def __init__(self,  data_path:list):
        self.data_path = data_path # [[net, pop],[],[],...]

    def __getitem__(self, index):
        net = cv2.imread(self.data_path[index][0])
        net = cv2.cvtColor(net, cv2.COLOR_BGR2RGB)
        pop = cv2.imread(self.data_path[index][1], 2)  # torch.Size([1, 512, 512])
        dem = cv2.imread(self.data_path[index][2], 2)  # torch.Size([1, 512, 512])
        lan = cv2.imread(self.data_path[index][3], 2) # torch.Size([1, 512, 512])

        pop = torch.tensor(pop).float().unsqueeze(0)
        dem = torch.tensor(dem).float().unsqueeze(0)
        lan = torch.tensor(lan).float().unsqueeze(0)

        net = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])(net)
        pop = (pop-pop.mean())/pop.std()
        dem = (dem-dem.mean())/dem.std()

        #source = torch.cat([pop, dem, lan], dim=0)
        return (pop, dem, lan), net


    def __len__(self):
        return len(self.data_path)


class NumpyDataset(Dataset):
    def __init__(self,  data_path:list, mode='train',p=0.8, transform=None):
        self.data_path = data_path # [[net, pop],[],[],...]
        self.transform = transform
        self.mode=mode
        self.p=p

    def __getitem__(self, index):
        net = np.load(self.data_path[index][0])
        pop = cv2.imread(self.data_path[index][1], 2)  # torch.Size([1, 512, 512])
        dem = cv2.imread(self.data_path[index][2], 2)  # torch.Size([1, 512, 512])
        lan = cv2.imread(self.data_path[index][3], 2) # torch.Size([1, 512, 512])

        pop = torch.tensor(pop).float().unsqueeze(0)
        dem = torch.tensor(dem).float().unsqueeze(0)
        lan = torch.tensor(lan).float().unsqueeze(0)

        pop = (pop-pop.mean())/pop.std()
        dem = (dem-dem.mean())/dem.std()
        net = net.astype(np.float32)
        net = torch.from_numpy(net).permute(2,0,1)
        #source = torch.cat([pop, dem, lan], dim=0)
        # if self.mode=='train':
        #     if random.random()>self.p:
        #         pop=torch.zeros_like(pop)
        #         dem=torch.zeros_like(dem)  
        #         lan=torch.zeros_like(lan)              

        return (pop, dem, lan), net


    def __len__(self):
        return len(self.data_path)

class Loaders:
    '''
    Initialize dataloaders
    '''
    def __init__(self, batch_size ,ratio = 0.7,data_type = "numpy"):
        self.batch_size = batch_size
        data_train, data_test= self.getpathlist(shuffle=True, ratio=ratio,data_type =data_type) # type options: 'GAN' or 'V'
        #print(data_test)
        if data_type == "numpy":
            self.train_loader = DataLoader(NumpyDataset(data_train,mode='train',p=0.8),batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=4)
            self.test_loader = DataLoader(NumpyDataset(data_test,mode='test',p=1.0), batch_size=1, shuffle=False, drop_last=True)

        if data_type == "RGB":
            self.train_loader = DataLoader(RGBDataset(data_train),batch_size=self.batch_size, shuffle=False, drop_last=True,num_workers=4)
            self.test_loader = DataLoader(RGBDataset(data_test), batch_size=1, shuffle=False, drop_last=True)


    # def getpathlist(self, ratio=0.7, shuffle=False,data_type = "numpy",seed=666):
    #     root="/home/lli/DATASET"
    #     path_list = []
    #     for city in ["ME","SY","BR"]:
    #         csv_file = "/home/lli/DATASET/{}.csv".format(city) # choose a dataset
    #         with open(csv_file) as f:
    #             reader = csv.reader(f)
    #             head = next(reader)
    #             for row in reader:
    #                 no = int(row[0])
    #                 now_root=os.path.join(root,city)
    #                 if data_type == "numpy":
    #                     net = "/home/lli/npy_ds_5/" +city+ '/npy/_NET_{}.npy'.format(no)
    #                 if data_type == "RGB":
    #                     net = "/home/lli/npy_ds_5/" +city+ '/img/_NET_{}.png'.format(no)
    #                 pop = now_root + '/POP/'+ city+'_POP_{}.TIF'.format(no)
    #                 dem = now_root + '/DEM/'+ city+'_DEM_{}.TIF'.format(no)
    #                 lan = now_root + '/LAN/'+ city+'_LAN_{}.TIF'.format(no)
    #                 # path_list.append([net, pop, dem, lan]) #  net, pop, dem, lan
    #                 path_list.append([net, pop, dem, lan]) #  net, pop, dem, lan
    #     if shuffle:
    #         random.seed(seed)
    #         random.shuffle(path_list)
    #     train_list,test_list = self.train_test_split(path_list,ratio)
    #     print(f"train data num:{len(train_list)},test date num:{len(test_list)}")
        # return train_list,test_list

    def train_test_split(self, path_list, ratio):
        split = int(len(path_list) * ratio)
        return path_list[:split], path_list[split:]



    def getpathlist(self, ratio=0.7, shuffle=False,data_type = "numpy",seed=666):
        root="/home/sadong/DATASET"
        train_list = []
        test_list = []
        for city in ["ME","SY","BR"]:
            csv_file = "/home/sadong/DATASET/{}.csv".format(city) # choose a dataset
            with open(csv_file) as f:
                reader = csv.reader(f)
                head = next(reader)
                for row in reader:
                    no = int(row[0])
                    now_root=os.path.join(root,city)
                    if data_type == "numpy":
                        net = "/home/sadong/DATASET/npy_ds/" +city+ '/npy/_NET_{}.npy'.format(no)
                    if data_type == "RGB":
                        net = "/home/sadong/DATASET/npy_ds/" +city+ '/img/_NET_{}.png'.format(no)
                    pop = now_root + '/POP/'+ city+'_POP_{}.TIF'.format(no)
                    dem = now_root + '/DEM/'+ city+'_DEM_{}.TIF'.format(no)
                    lan = now_root + '/LAN/'+ city+'_LAN_{}.TIF'.format(no)
                    # path_list.append([net, pop, dem, lan]) #  net, pop, dem, lan
                    if city in ["ME","SY"]:
                        train_list.append([net, pop, dem, lan]) #  net, pop, dem, lan
                    else:
                        test_list.append([net, pop, dem, lan])
        print(f"train data num:{len(train_list)},test date num:{len(test_list)}")
        return train_list,test_list


#print(con.shape)
