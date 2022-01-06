from torch import mean
from torchvision import transforms

__all__ = ['pacs', 'vlcs']

class DGTransformer(object):
    def __init__(self,process_config):
        """There must be 'size', 'mean' and 'std' keys in the process_config dict."""
        self.size = process_config['size']
        self.mean = process_config['mean']
        self.std = process_config['std']

    def test_transformer(self):
        self.transform_list = [
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        return transforms.Compose(self.transform_list)
    
    def train_transformer(self, scales=[0.8, 1], random_h_f=0.5, jitter=0.4):
        self.transform_list = [transforms.RandomResizedCrop(
            self.size, (scales[0], scales[1]))]
        if random_h_f > 0.0:
            self.transform_list.append(
                transforms.RandomHorizontalFlip(random_h_f))
        if jitter > 0.0:
            self.transform_list.append(transforms.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.5, jitter)))
        self.transform_list.append(transforms.RandomGrayscale(0.1))
        self.transform_list.append(transforms.ToTensor())
        self.transform_list.append(
            transforms.Normalize(mean=self.mean, std=self.std))

        return transforms.Compose(self.transform_list)
    
class RMnist(DGTransformer):
    def __init__(self, process_config):
        super().__init__(process_config)
    
    def test_transformer(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    def train_transformer(self,*args):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

class DigitFive(DGTransformer):
    def __init__(self, process_config):
        super().__init__(process_config)
    
    def test_transformer(self):
        return super().test_transformer()
    
    def train_transformer(self, *args):
        return self.test_transformer()

def pacs(process_config):
    return DGTransformer(process_config)

def vlcs(process_config):
    return DGTransformer(process_config)

def rmnist(process_config):
    return RMnist(process_config)

def digit_five(process_config):
    return DigitFive(process_config)

def office_8(process_config):
    return DGTransformer(process_config)
