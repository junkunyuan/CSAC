from datasets.basedataset import BaseDataset
from .basedata import DGBaseData
import os.path as osp

class PACS(DGBaseData):
    def __init__(self, config, train_transform, test_transform, rtn_pth=False):
        super(PACS, self).__init__(config, train_transform, test_transform, rtn_pth)
    
    def train_dataset(self, domain, split=None):
        pth = osp.join(self.config['dataset']['list_dir'], self.config['dataset']['domains'][domain] + '.txt')
        pth2label = self.__load_list__(pth)

        if split is None:
            return BaseDataset(pth2label, self.train_transform, self.root_dir, self.rtn_pth)
        else:
            train_pth2label, test_pth2label = self.split_dset(pth2label)
            return BaseDataset(train_pth2label, self.train_transform, self.root_dir, self.rtn_pth), BaseDataset(test_pth2label, self.test_transform, self.root_dir, self.rtn_pth)

def pacs(config, train_transform, test_transform, rtn_pth=False):
    return PACS(config, train_transform, test_transform, rtn_pth)
