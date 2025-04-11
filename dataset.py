import time
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

import raw_solve
from config import config

class LLIE_Dataset(Dataset):
    def __init__(self, data_list):
        self.img = []
        self.target = []
        self.ratio = []
        for data in data_list:
            self.img.append(np.array(data[0]))
            self.target.append(np.array(data[1]))
            self.ratio.append(torch.full((config['picture_size'][1], config['picture_size'][0], 1), data[2]))
        del data_list
        self.img = torch.Tensor(np.array(self.img))
        self.target = torch.Tensor(np.array(self.target))
        self.ratio = torch.Tensor(np.array(self.ratio))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        # return torch.tensor(np.array([self.img[index], self.target[index]]))
        return self.img[index], self.target[index], self.ratio[index]

def get_dataset(dataset_name, data_type, lines):

    time.sleep(0.1)
    data = []
    for line in tqdm(lines, desc=dataset_name + ' ' + data_type + ' data loading...'):
        line_list = []
        img_path = config['data_path'] + line.split(' ')[0][2:]
        target_path = config['data_path'] + line.split(' ')[1][2:]
        img, target, ratio = raw_solve.raw2np(dataset_name, img_path, target_path)
        img = cv2.resize(img, config['picture_size'], interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
        target = cv2.resize(target, config['picture_size'], interpolation=cv2.INTER_NEAREST)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)/255
        # img = cv2.resize(img, config['picture_size'], interpolation=cv2.INTER_NEAREST)
        # target = cv2.resize(target, config['picture_size'], interpolation=cv2.INTER_NEAREST)
        ratio = ratio/100
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        line_list.append(img)
        line_list.append(target)
        line_list.append(ratio)
        data.append(line_list)
        del img, target, ratio
    dataset = LLIE_Dataset(data)
    del data
    if data_type == 'train':
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader

def get_dataset_LOL(data_type, debug):
    LOL_root = '../data/LOL'
    if data_type == 'train':
        target_root_path = LOL_root + '/our485/high/'
        img_root_path = LOL_root + '/our485/low/'
    elif data_type == 'val':
        target_root_path = LOL_root + '/eval15/high/'
        img_root_path = LOL_root + '/eval15/low/'
    time.sleep(0.1)
    data = []
    filename_list = os.listdir(target_root_path)
    if debug == 'Y':
        filename_list = filename_list[0:config['batch_size']*2]
    for filename in tqdm(filename_list, desc='LOL ' + data_type + ' data loading...'):
        if filename != '.DS_Store':
            line_list = []
            img_path = img_root_path + filename
            target_path = target_root_path + filename
            # if debug == 'Y':
            #     print(img_path, target_path)
            img = cv2.imread(img_path)
            target = cv2.imread(target_path)
            # if debug == 'Y':
            #     print(img.shape, target.shape)
            ratio = adp_ratio(img, target)
            if debug == 'Y':
                print(ratio)
            img = cv2.resize(img, config['picture_size'], interpolation=cv2.INTER_NEAREST)/255
            target = cv2.resize(target, config['picture_size'], interpolation=cv2.INTER_NEAREST)/255
            line_list.append(img)
            line_list.append(target)
            line_list.append(ratio)
            data.append(line_list)
            del img, target, ratio
    dataset = LLIE_Dataset(data)
    del data
    if data_type == 'train':
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader

def adp_ratio(img, target):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # img = img.astype(float)
    # target = target.astype(float)
    target[target == 0] = 1
    img[img == 0] = 1
    ratio = np.divide(target-img, target)
    ratio_num = np.mean(ratio)
    # print(ratio_num)
    return ratio_num
