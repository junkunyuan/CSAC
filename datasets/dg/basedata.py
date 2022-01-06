from datasets.basedataset import BaseDataset
import os.path as osp
import numpy as np
import random


class DGData(object):
    def __init__(self, config, train_transform, test_transform, rtn_pth=False):
        self.config = config
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.rtn_pth = rtn_pth
        self.root_dir = config['dataset']['root_dir']

    def split_dset(self, pth2label):
        train_datas = []
        test_datas = []
        for c in range(self.config["dataset"]["class_num"]):
            index = np.where(pth2label[:, 1] == str(c))
            random.shuffle(index[0])
            split = int(len(index[0]) * self.config["process"]["scales"][0])
            train_datas.append(pth2label[index[0][:split]])
            test_datas.append(pth2label[index[0][split:]])
        return np.vstack(train_datas), np.vstack(test_datas)

    def train_dataset(self, domain, split=None):
        pth2label = self.__load_list__(osp.join(self.config["dataset"]["list_dir"], self.config["dataset"]['domains'][domain]+".txt"))

        if split is None:
            return BaseDataset(pth2label, self.train_transform, self.root_dir, self.rtn_pth)
        else:
            train_pth2label, test_pth2label = self.split_dset(pth2label)
            return BaseDataset(train_pth2label, self.train_transform, self.root_dir, self.rtn_pth), BaseDataset(test_pth2label, self.test_transform, self.root_dir, self.rtn_pth)

    def test_dataset(self, domain):
        pth2label = self.__load_list__(osp.join(
            self.config["dataset"]["list_dir"], self.config["dataset"]['domains'][domain] + ".txt"))
        return BaseDataset(pth2label, self.test_transform, self.root_dir, False)

    def __load_list__(self, list_dir):
        return np.loadtxt(list_dir, dtype=str)

class DGBaseData(object):
    def __init__(self,config,train_transform,test_transform,rtn_pth=False):
        self.config = config
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.rtn_pth = rtn_pth
        self.root_dir = config['dataset']['root_dir']

    def split_dset(self,pth2label):
        train_datas = []
        test_datas = []
        for c in range(self.config["dataset"]["class_num"]):
            index = np.where(pth2label[:, 1] == str(c))
            random.shuffle(index[0])
            split = int(len(index[0])*self.config["process"]["scales"][0])
            train_datas.append(pth2label[index[0][:split]])
            test_datas.append(pth2label[index[0][split:]])
        return np.vstack(train_datas), np.vstack(test_datas)
    
    def test_dataset(self, domain):
        pth2label = self.__load_list__(osp.join(self.config["dataset"]["list_dir"], self.config["dataset"]['domains'][domain] + ".txt"))
        return BaseDataset(pth2label, self.test_transform, self.root_dir, False)

    def __load_list__(self, list_dir):
        return np.loadtxt(list_dir, dtype=str)
    
    def train_dataset(self, domain, split=None):
        pass




    

