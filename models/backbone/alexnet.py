import torch
import torch.nn as nn
from torchvision.models import alexnet as Alexnet


__all__ = ['AlexNet', 'alexnet']

class AlexNet(nn.Module):
    def __init__(self,pretrain = True):
        super(AlexNet,self).__init__()
        anet = Alexnet(pretrained=pretrain)
        self.conv1 = nn.Sequential(
            anet.features[0],
            anet.features[1],
            anet.features[2]
        )

        self.conv2 = nn.Sequential(
            anet.features[3],
            anet.features[4],
            anet.features[5]
        )

        self.conv3 = nn.Sequential(
            anet.features[6],
            anet.features[7]
        )

        self.conv4 = nn.Sequential(
            anet.features[8],
            anet.features[9]
        )

        self.conv5 = nn.Sequential(
            anet.features[10],
            anet.features[11],
            anet.features[12]
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6,6))
        
        self.bottleneck = nn.Sequential(
            anet.classifier[0],
            anet.classifier[1],
            anet.classifier[2],
            anet.classifier[3],
            anet.classifier[4],
            anet.classifier[5]
        )

        self._out_features = 4096

    @property
    def out_features(self):
        return self._out_features

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.avgpool(x5)
        x7 = x6.view(x.size(0), 256 * 6 * 6)
        x7 = self.bottleneck(x7)
        return x7, x3, x4, x5

def alexnet(pretrained=True, **kwargs):
    model = AlexNet(pretrained)
    return model

if __name__ == '__main__':
    datas = torch.rand([64, 3, 222, 222])
    net = alexnet(False)
    out = net(datas)
    
    