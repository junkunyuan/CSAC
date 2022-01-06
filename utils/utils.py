import json
import os
import os.path as osp

import numpy as np
import yaml
from PIL import Image
import torch


def load_pthList(list_pth):
    content = np.loadtxt(list_pth)
    return content[:, 0], content[:, 1]


def load_img(img_pth):
    if img_pth[-1] != 'g':
        img_pth += 'g'
    return Image.open(img_pth).covert("RGB")


def load_yaml(yaml_pth):
    with open(yaml_pth, encoding="UTF-8") as f:
        content = yaml.full_load(f)
    return content


def save_config(config, pth):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.device):
                return str(obj)
            return super(MyEncoder, self).default(obj)

    with open(pth, 'w', encoding='utf8') as f:
        json.dump(config, f, indent=4, cls=MyEncoder)
