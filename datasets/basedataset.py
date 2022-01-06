from torch.utils.data import Dataset
from PIL import Image
import torch, random, bisect
import numpy as np
import os.path as osp

def loader(img_pth):
    if img_pth[-1] != "g":
        img_pth += "g"
    return Image.open(img_pth).convert("RGB")

class BaseDataset(Dataset):
    def __init__(self, pth2label, transform, root_dir, rtn_pth=False):
        self.transform = transform
        self.rtn_pth = rtn_pth
        self.pth2label = pth2label
        self.root_dir = root_dir
    
    def __getitem__(self, index):
        item = self.pth2label[index]
        img = loader(osp.join(self.root_dir, item[0]))
        label = int(item[1])
        img = self.transform(img)
        if self.rtn_pth:
            return img, label, item[0]
        else:
            return img, label

    def __len__(self):
        return len(self.pth2label)

class RMnistDataset(Dataset):
    def __init__(self,pth2label,transform,root_dir):
        self.pth2label = pth2label
        self.root_dir = root_dir
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.pth2label[index]
        img = Image.open(osp.join(self.root_dir, item[0]))
        label = int(item[1])
        img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.pth2label)

class ClassDataset(Dataset):
    """Sample data from one class of one domain with batchsize 'class_bs'."""
    def __init__(self,pth2label,transform,root_dir,class_bs,class_num):
        self.pth2label = pth2label
        self.transform = transform
        self.class_bs = class_bs
        self.class_num = class_num
        self.root_dir = root_dir
    
    def __getitem__(self, index):
        inds = random.choice(range(len(self.pth2label[index])), k=self.class_bs)
        pths = np.asarray([self.pth2label[index][ind][0] for ind in inds], dtype = np.str)
        labels = torch.tensor([int(self.pth2label[index][ind][1]) for ind in inds])
        imgs = torch.stack([self.transform(loader(osp.join(self.root_dir, p))) for p in pths], 0)

        return imgs, labels
    
    def __len__(self):
        return self.class_num

    def split_class(self, pth2label):
        result = {}
        for c in range(self.class_num):
            result[c] = min(self.class_bs, len(self.pth2label[c]))
        return result
    
    def reconstruct(self, pth2label):
        result = {}
        keys = np.asarray(list(pth2label.keys()), dtype=str)
        values = np.asarray(list(pth2label.values()), dtype=str)
        for c in range(self.class_num):
            index_c = np.where(values == str(c))
            keys_c = np.asarray(keys[index_c]).reshape(-1, 1)
            values_c = np.asarray(values[index_c]).reshape(-1, 1)
            result[c] = np.concatenate((keys_c,values_c), 1)
        self.pth2label = result

class AlignDataset(Dataset):
    def __init__(self, *datasets, chooses=None):
        self.datasets = datasets
        if chooses:
            assert isinstance(chooses,tuple) or isinstance(chooses,list), "chosses must a Integer list or Ingeger tuple"
            self.all = chooses
        else:
            self.all = list(range(len(self.datasets)))

    def __getitem__(self, index):
        datas, labels = [], []
        for i in self.all:
            data, label = self.datasets[i].__getitem__(index)
            datas.append(data)
            labels.append(label)
        return datas, labels
    
    def __len__(self):
        return self.datasets[0].__len__()
    
    def reconstruct(self, dset_idxs, pth2labels):
        """Use 'dset_idxs' to indicate which datasets need to be reconstructed ('dset_idxs' and 'pth2labels' should have the same shape)."""
        assert len(dset_idxs) == len(pth2labels), "dset_idxs and pth2labels should have the same shape"
        for idx, pth2label in zip(dset_idxs, pth2labels):
            self.datasets[idx].reconstruct(pth2label)

class ConcatDataset(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx
