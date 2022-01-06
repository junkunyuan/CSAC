from torch import nn
import torch.nn.functional as F
import torch

class KLLoss(nn.Module):
    def __init__(self,T=4):
        super(KLLoss,self).__init__()
        self.T = T
        self.klloss = nn.KLDivLoss(reduction='mean')
    
    def forward(self, inputs, inputt):
        ps = F.log_softmax(inputs, dim=1)
        pt = F.softmax(inputt, dim=1)

        loss = self.klloss(ps,pt)
        return loss
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


if __name__ == '__main__':
    import torch
    lossf = KLLoss()
    s = torch.rand((32, 7))
    t = torch.rand((32, 7))
    print(lossf(s, t))
