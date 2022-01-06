from datasets.basedataset import BaseDataset
from .basedata import DGBaseData
import os.path as osp

class VLCS(DGBaseData):
    def __init__(self, config, train_transform, test_transform, rtn_pth=False):
        super(VLCS, self).__init__(config, train_transform, test_transform, rtn_pth)
    
    def train_dataset(self, domain, split=None):
        pth_train = osp.join(self.config["dataset"]["list_dir"], self.config["dataset"]['domains'][domain] + "_train.txt")
        pth2label_train = self.__load_list__(pth_train)
        pth_test = osp.join(self.config["dataset"]["list_dir"], self.config["dataset"]['domains'][domain] + "_test.txt")
        pth2label_test = self.__load_list__(pth_test)
        if split is None:
            return BaseDataset(pth2label_train, self.train_transform, self.root_dir, self.rtn_pth)
        else:
            train_dset =  BaseDataset(pth2label_train, self.train_transform, self.root_dir, self.rtn_pth)
            val_dset  = BaseDataset(pth2label_test, self.test_transform, self.root_dir, self.rtn_pth)
            return train_dset, val_dset
    
    def test_dataset(self, domain):
        pth = osp.join(self.config["dataset"]["list_dir"], self.config["dataset"]['domains'][domain] + "_test.txt")
        pth2label = self.__load_list__(pth)
        return BaseDataset(pth2label, self.test_transform, self.root_dir, False)

def vlcs(config, train_transform, test_transform, rtn_pth=False):
    return VLCS(config, train_transform, test_transform, rtn_pth)
