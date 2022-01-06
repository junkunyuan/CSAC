import torch
import torch.nn as nn

class BaseFullNet(nn.Module):
    def __init__(self, backbone, class_num,bottleneck=None, bottleneck_dim=-1, classifier=None):
        super(BaseFullNet,self).__init__()
        self.backbone = backbone
        self.class_num = class_num
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._feature_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._feature_dim = bottleneck_dim
        
        if classifier is None:
            self.classifier = nn.Linear(self._feature_dim, class_num)
        else:
            self.classifier = classifier
    
    @property
    def feature_dim(self):
        return self._feature_dim
    
    def forward(self,x):
        f, f2, f3, f4 = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)
        predictions = self.classifier(f)
        return predictions, f, f2, f3, f4
    
    def get_params(self):
        params = [
            {'params':self.backbone.parameters(),'lr0': 0.1},
            {'params':self.bottleneck.parameters(),'lr0': 0.1},
            {'params':self.classifier.parameters(),'lr0': 0.1}
        ]

        return params

class FLDGFullNet(BaseFullNet):
    def __init__(self, backbone, class_num, bottleneck_dim=256):
        classifier = None
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        if backbone.__class__.__name__ == "LeNet":
            bottleneck = None
        
        if backbone.__class__.__name__ == 'AlexNet':
            bottleneck = None
            classifier = nn.Linear(4096, class_num)
            nn.init.xavier_uniform_(classifier.weight, 0.1)
            nn.init.constant_(classifier.bias, 0.0)
        super(FLDGFullNet,self).__init__(backbone, class_num, bottleneck, bottleneck_dim, classifier=classifier)
 