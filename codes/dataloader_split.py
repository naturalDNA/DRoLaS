from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import os
import cv2
import torch
import csv
import random
import numpy as np

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
    def __init__(self,  data_path:list, transform=None):
        self.data_path = data_path # [[net, pop],[],[],...]
        self.transform = transform

    def __getitem__(self, index):
        net = np.load(self.data_path[index][0])
        pop = cv2.imread(self.data_path[index][1], 2)  # torch.Size([1, 512, 512])
        dem = cv2.imread(self.data_path[index][2], 2)  # torch.Size([1, 512, 512])
        lan = cv2.imread(self.data_path[index][3], 2) # torch.Size([1, 512, 512])

        city = self.data_path[index][4]
        no = self.data_path[index][5]

        pop = torch.tensor(pop).float().unsqueeze(0)
        dem = torch.tensor(dem).float().unsqueeze(0)
        lan = torch.tensor(lan).float().unsqueeze(0)

        pop = (pop-pop.mean())/pop.std()
        # 将dem按照最大值最小值归一化
        dem = (dem-dem.mean())/dem.std()
        net = net.astype(np.float32)
        net = torch.from_numpy(net).permute(2,0,1)
        #source = torch.cat([pop, dem, lan], dim=0)
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
            self.train_loader = DataLoader(NumpyDataset(data_train),batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
            self.test_loader = DataLoader(NumpyDataset(data_test), batch_size=1, shuffle=False, drop_last=True)

        if data_type == "RGB":
            self.train_loader = DataLoader(RGBDataset(data_train),batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
            self.test_loader = DataLoader(RGBDataset(data_test), batch_size=1, shuffle=False, drop_last=True)

    def getpathlist(self, ratio=0.7, shuffle=False, seed=666,data_type = "numpy"):
        cities=["BR","ME","SY"]
        root="/home/sadong/graduate_paper/DATASET"
        path_list = []
        for city in cities:
            csv_file = "/home/sadong/graduate_paper/DATASET/{}.csv".format(city) # choose a dataset
            with open(csv_file) as f:
                reader = csv.reader(f)
                head = next(reader)
                for row in reader:
                    no = int(row[0])
                    now_root=os.path.join(root,city)
                    if data_type == "numpy":
                        net = "/home/sadong/graduate_paper/DATASET/npy_ds/" +city+ '/npy/_NET_{}.npy'.format(no)
                    elif data_type == "RGB":
                        net = "/home/sadong/DATASET/npy_ds/"+ city+'/img/_NET_{}.png'.format(no)
                    pop = now_root + '/POP/'+ city+'_POP_{}.TIF'.format(no)
                    dem = now_root + '/DEM/'+ city+'_DEM_{}.TIF'.format(no)
                    lan = now_root + '/LAN/'+ city+'_LAN_{}.TIF'.format(no)
                    # path_list.append([net, pop, dem, lan]) #  net, pop, dem, lan
                    path_list.append([net, pop, dem, lan,city,no]) #  net, pop, dem, lan
        print(len(path_list))
        
        import json
        with open("indices.json", "r") as f:
            indices = json.load(f)
        path_list = [path_list[i] for i in indices]
        return self.train_test_split(path_list, ratio)
    
    def train_test_split(self, path_list, ratio):
        split = int(len(path_list) * ratio)
        return path_list[:split], path_list[split:]




if __name__ == '__main__':
    loaders = Loaders(1, data_type = "numpy")
    for i, ((pop, dem, lan), net) in enumerate(loaders.train_loader):
        print(pop.shape, dem.shape, lan.shape, net.shape)
        break
    for i, ((pop, dem, lan), net) in enumerate(loaders.test_loader):
        print(pop.shape, dem.shape, lan.shape, net.shape)
        break
    print("Done")