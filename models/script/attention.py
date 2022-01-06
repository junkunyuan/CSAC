import torch
from torch import dtype, nn
import torch.nn.functional as F


class PAM_Module(nn.Module):
    def __init__(self, num, sizes,mode=None):
        super(PAM_Module, self).__init__()
        self.sizes = sizes
        self.mode = mode
        for i in range(num):
            setattr(self, "query" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))
            setattr(self, "value" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))
            setattr(self, "key" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))

    def forward(self, feat_sources, feat_targets):
        """calculate the attention weight and alpha"""
        ret_feats, ret_alphas = [], []
        for i, query in enumerate(feat_targets):
            Bt, Ct, Ht, Wt = query.size()
            pro_query = getattr(self, "query"+str(i)
                                )(query).view(Bt, -1, Ht*Wt).permute(0, 2, 1)
            attentions, means = [], []
            for j, key in enumerate(feat_sources):
                pro_key = getattr(self, "key" + str(j))(key).view(Bt, -1, Ht * Wt)
                energy = torch.bmm(pro_query, pro_key)
                means.append(energy.mean().item())
                attentions.append(torch.softmax(energy, dim=-1))
            
            if self.mode.find('alpha')>=0:
                ret_alphas.append(torch.softmax(torch.tensor(means), dim=0))
            else:
                ret_alphas.append(torch.tensor(means).mean())
            if self.mode in ['all', 'pam', 'cam', 'alpha_cam', 'alpha_cam', 'alpha_all']:
                attention = torch.stack(attentions, dim=0).sum(0)
                value = getattr(self, "value" + str(i))(query).view(Bt, -1, Ht * Wt)

                out = torch.bmm(value, attention.permute(0, 2, 1)).view(Bt, Ct, Ht, Wt)
                ret_feats.append(out)
                
        if self.mode.find('alpha') >= 0:
            ret_alphas = torch.stack(ret_alphas, dim=0)
        else:
            ret_alphas = torch.softmax(torch.tensor(ret_alphas), dim=0)
        return ret_feats, ret_alphas


class CAM_Module(nn.Module):
    def __init__(self, num, sizes, mode=None):
        super(CAM_Module, self).__init__()
        self.sizes = sizes
        self.mode = mode
        for i in range(num):
            setattr(self, "value" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))

    def forward(self, feat_sources, feat_targets):
        ret_feats, ret_alphas = [], []

        for i, query in enumerate(feat_targets):
            Bt, Ct, Ht, Wt = query.size()
            pro_query = query.view(Bt, Ct, -1)
            attentions, means = [], []
            for j, key in enumerate(feat_sources):
                pro_key = key.view(Bt, Ct, -1).permute(0, 2, 1)
                energy = torch.bmm(pro_query, pro_key)
                means.append(energy.mean().item())
                attentions.append(torch.softmax(energy, dim=-1))

            if self.mode.find('alpha') >= 0:
                ret_alphas.append(torch.softmax(torch.tensor(means), dim=0))
            else:
                ret_alphas.append(torch.tensor(means).mean())

            if self.mode in ['all', 'pam', 'cam', 'alpha_cam', 'alpha_cam', 'alpha_all']:
                attention = torch.stack(attentions, dim=0).sum(0)
                value = getattr(self, "value"+str(i))(query).view(Bt, Ct, -1)

                out = torch.bmm(attention, value).view(Bt, Ct, Ht, Wt)
                ret_feats.append(out)
        if self.mode.find('alpha') >= 0:
            ret_alphas = torch.stack(ret_alphas, dim=0)
        else:
            ret_alphas = torch.softmax(torch.tensor(ret_alphas), dim=0)
        return ret_feats, ret_alphas


class ConvReg(nn.Module):
    def __init__(self, s_shape, t_shape, factor=1):
        super(ConvReg, self).__init__()
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(
                s_C, t_C // factor, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(
                s_C, t_C // factor, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented(
                'student size {}, teacher size {}'.format(s_H, t_H))

    def forward(self, x):
        x = self.conv(x)
        return x


class Fit(nn.Module):
    def __init__(self, s_shape, t_shape, factor=1):
        super(Fit, self).__init__()
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        if s_H == 2*t_H:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(
                s_C, t_C//factor, kernel_size=4, stride=2, padding=1)
        elif s_H == t_H:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=1, stride=1, padding=0)
        else:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=(1+s_H-t_H, 1 + s_W-t_W))
        # if channels:
        #     self.conv = nn.Conv2d(s_C,channels,kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        # else:
        #     self.conv = nn.Conv2d(s_C,t_C//factor,kernel_size=(1+s_H-t_H, 1+s_W-t

    def forward(self, x):
        x = self.conv(x)
        return x


# torch.Size([16, 128, 28, 28]) torch.Size([16, 256, 14, 14]) torch.Size([16, 512, 7, 7])

class Project(nn.Module):
    def __init__(self, origin_sizes, new_size=torch.Size([-1, 16, 14, 14]), factor=1):
        super(Project, self).__init__()
        for i, size_o in enumerate(origin_sizes):
            setattr(self, "target"+str(i),
                    Fit(size_o, new_size, factor=factor))
            setattr(self, "source"+str(i),
                    Fit(size_o, new_size, factor=factor))

    def forward(self, feat_sources, feat_targets):
        new_feat_sources, new_feat_targets = [], []
        for i, source in enumerate(feat_sources):
            new_feat_sources.append(getattr(self, "source" + str(i))(source))
        for i, target in enumerate(feat_targets):
            new_feat_targets.append(getattr(self, "target" + str(i))(target))
        return new_feat_sources, new_feat_targets


class DAAttention(nn.Module):
    def __init__(self, origin_sizes, new_size=torch.Size([-1, 32, 7, 7]), factor=1, mode="all"):
        super(DAAttention, self).__init__()
        self.pro = Project(origin_sizes, new_size=new_size, factor=factor)

        self.mode = mode
        self.layer_num = len(origin_sizes)

        if mode in ['all', 'alpha', 'pam', 'alpha_pam', 'alpha_all']:
            self.pam = PAM_Module(self.layer_num, new_size, self.mode)
        if mode in ['all', 'alpha', 'cam', 'alpha_cam', 'alpha_all']:
            self.cam = CAM_Module(self.layer_num, new_size, self.mode)

        self.C = new_size[1]
        self.H = new_size[2]
        self.W = new_size[3]

    def forward(self, feat_sources, feat_targets):
        new_feat_sources, new_feat_targets = self.pro(
            feat_sources, feat_targets)        

        if self.mode in ['pam', 'all', 'alpha', 'alpha_pam', 'alpha_all']:
            feat_pam, alpha_pam = self.pam(new_feat_sources, new_feat_targets)
        if self.mode in ['cam', 'all', 'alpha', 'alpha_cam', 'alpha_all']:
            feat_cam, alpha_cam = self.cam(new_feat_sources, new_feat_targets)
        
        ret_alpha = None
        ret_targets, ret_sources = [], []
        
        for i in range(self.layer_num):
            if self.mode in ['all', 'alpha_all']:
                ret_targets.append(((feat_pam[i] + feat_cam[i]) * 0.5).view(-1, self.C * self.H * self.W))
                ret_alpha = (alpha_cam+alpha_pam) * 0.5
            elif self.mode == 'cam':
                ret_targets.append(feat_cam[i].view(-1, self.C * self.H * self.W))
                ret_alpha = alpha_cam
            elif self.mode == 'pam':
                ret_targets.append(feat_pam[i].view(-1, self.C * self.H * self.W))
                ret_alpha = alpha_pam
            elif self.mode in ['alpha', 'alpha_pam', 'alpha_cam']:
                if self.mode == 'alpha':ret_alpha = (alpha_pam + alpha_cam) * 0.5
                elif self.mode == 'alpha_cam': ret_alpha = alpha_cam
                elif self.mode == 'alpha_pam': ret_alpha = alpha_pam
            elif self.mode[:3] == 'noa':
                ret_targets.append(new_feat_targets[i].view(-1, self.C * self.H * self.W))
            
            ret_sources.append(new_feat_sources[i].view(-1, self.C * self.H * self.W))


        return ret_sources, ret_alpha, ret_targets


if __name__ == '__main__':
    # feat_source1 = torch.rand((16,512,28,28))
    # feat_source2 = torch.rand((16,1024,14,14))
    # feat_source3 = torch.rand((16,2048,7,7))

    # feat_target1 = torch.rand((16, 512, 28, 28))
    # feat_target2 = torch.rand((16, 1024, 14, 14))
    # feat_target3 = torch.rand((16, 2048, 7, 7))

    # att = DAAttention([feat_source1.size(),feat_source2.size(),feat_source3.size()])
    # out,alpha = att([feat_source1,feat_source2,feat_source3],[feat_target1,feat_target2,feat_target3])
    # print(out[0].size(),alpha.size())
    # print(out[1].size(),alpha.size())
    # print(out[2].size(),alpha.size())

    # import sys
    # sys.path.append('../..')
    # sys.path.append('..')
    # from models.fullnet import FLDGFullNet
    # from models.backbone import resnet18
    # backbone = resnet18()
    # net = FLDGFullNet(backbone, 7)
    # data = torch.rand((16, 3, 224, 224))
    # a, b, c, d, e = net(data)
    # print(c.size(), d.size(), e.size())
    # torch.Size([16, 128, 28, 28]) torch.Size([16, 256, 14, 14]) torch.Size([16, 512, 7, 7])

    import torch
    a = torch.rand(3, 3)
    print(a, a[0, 0].item())
