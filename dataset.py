from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random
from torchvision.transforms import functional as F

class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 256
        self.height = 256
        self.train_path = data_path
        self.datas = OrderedDict()
        self.prob = 0.5
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.*'))
                self.datas[data_name]['image'].sort()

    def __getitem__(self, index):
        
        # load image1
        input = cv2.imread(self.datas['input']['image'][index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])
        
        # load image2
        gt = cv2.imread(self.datas['gt']['image'][index])
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt.astype(dtype=np.float32)
        gt = (gt / 127.5) - 1.0
        gt = np.transpose(gt, [2, 0, 1])
        
        # convert to tensor
        input_tensor = torch.tensor(input)
        gt_tensor = torch.tensor(gt)
        #print("&&&&&&&&&", index, input_tensor.shape)
        
        # random flip
        if random.random() < self.prob:
            input_tensor = F.hflip(input_tensor)
            gt_tensor = F.hflip(gt_tensor)
            #input_tensor = input_tensor[:, :, ::-1]
            #gt_tensor = gt_tensor[:, :, ::-1]
        
        return (input_tensor, gt_tensor)
        

    def __len__(self):

        return len(self.datas['input']['image'])

class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 256
        self.height = 256
        self.test_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.*'))
                self.datas[data_name]['image'].sort()

    def __getitem__(self, index):
        
        # load image1
        input = cv2.imread(self.datas['input']['image'][index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])
        
        # load image2
        gt = cv2.imread(self.datas['gt']['image'][index])
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt.astype(dtype=np.float32)
        gt = (gt / 127.5) - 1.0
        gt = np.transpose(gt, [2, 0, 1])
        
        # convert to tensor
        input_tensor = torch.tensor(input)
        gt_tensor = torch.tensor(gt)

        return (input_tensor, gt_tensor)

    def __len__(self):

        return len(self.datas['input']['image'])


