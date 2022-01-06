import random
from torch import optim
import torch
from models.script.attention import DAAttention

class Trainer(object):
    def __init__(self,model,train_loader,epoch,length,optim_config,criterion,device,test_loaders):
        super().__init__()
        self._model = model
        self._loader = train_loader
        self._epoch = epoch
        self._length = length
        self._optimizer = getattr(optim, optim_config['name'])(self.model.get_params(), **optim_config['params'])
        self._criterion = criterion
        self._device = device
        self._test_loaders = test_loaders

    def train(self, logger, accrecorder=None, trace_acc=None, print_step=10, trace_domains=None):
        self._logger = logger
        self._accrecorder = accrecorder

        self._trace_acc = trace_acc
        self._print_step = print_step
        self._trace_domains = trace_domains
        
        for epoch in range(self._epoch):
            self.epoch = epoch
            self._epoch_train()

            self.model.eval()
            acc = self._test(*list(self._test_loaders.keys()))
            logger.append('epoch/Epoch: {:>4d}/{:>4d} acc: {}'.format(epoch, self._epoch, str(acc)))
            self.model.train()
    
    def _epoch_train(self):
        self._model.train()
        for it in range(self._length):
            data,label = next(self._loader)
            data,label = data.to(self._device), label.to(self._device)

            loss = self.cal_loss(data, label)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            """Trace acc."""
            if self._trace_acc and (it+1) % self._print_step == 0:
                self._model.eval()
                self.trace_acc(self.cal_step(it))
                self._model.train()
    
    def _test(self, *domains):
        accs = {}
        for domain in domains:
            total = 0
            with torch.no_grad():
                correct = 0
                for it, (data, label) in enumerate(self._test_loaders[domain]):
                    data,label = data.to(self._device), label.to(self._device)
                    outs = self._model(data)
                    _,cls_pred = outs[0].max(dim=1)
                    correct += torch.sum(cls_pred==label.data)
                    total += data.size(0)
            accs[domain] = round((float(correct) / total) * 100, 4)
        return accs

    def cal_loss(self, x, y):
        out = self._model(x)
        return self._criterion(out[0], y)
    
    def trace_acc(self, step):
        if self._trace_domains == None:
            self._trace_domains = list(self._test_loaders.keys())
        acc = self._test(*list(self._trace_domains))
        self._accrecorder.updata(step, acc)
    
    def cal_step(self,it):
        return self.epoch * self._length+it

    @property
    def model(self):
        return self._model

class ClientTrainer(Trainer):
    def __init__(self, model, train_loader, epoch, length, optim_config, criterion, device, test_loaders):
        super(ClientTrainer,self).__init__(model, train_loader, epoch, length, optim_config, criterion, device, test_loaders)
    
    def train(self, logger, accrecorder=None, trace_acc=None, print_step=None, outer_epoch=0, trace_domains=None):
        self.outer_epoch = outer_epoch
        super(ClientTrainer,self).train(logger, accrecorder, trace_acc, print_step, trace_domains)
    
    def cal_step(self, it):
        return self.outer_epoch * self._epoch * self._length + self.epoch * self._length + it

class AlignPrivateTrainer(Trainer):
    def __init__(self, mkmmd_loss, fix_model, train_model, train_loader, epoch, length, optim_config, criterion, device, test_loader, mode="all", flag=None, net='resnet18'):
        self.mode = mode
        self.flag = flag
        self.mkmmd_loss = mkmmd_loss
        self._fix_model = fix_model
        self._fix_model.eval()
        if self.mode in ['all', 'pam', 'cam', 'noa_mmmd', 'alpha', 'noa_mmd', 'alpha_cam', 'alpha_pam', 'alpha_all']:
            if net == 'resnet18':
                atten_sizes = [torch.Size([-1, 128, 28, 28]),
                        torch.Size([-1, 256, 14, 14]),
                        torch.Size([-1, 512, 7, 7])]
                new_sizes = torch.Size([-1, 32, 7, 7])
            elif net == 'dtnnet':
                atten_sizes = [
                    torch.Size([-1, 64, 14, 14]),
                    torch.Size([-1, 128, 7, 7]),
                    torch.Size([-1, 256, 4, 4])
                ]
                new_sizes = torch.Size([-1, 32, 4, 4])
            elif net == 'alexnet':
                atten_sizes = [
                    torch.Size([-1, 384, 13, 13]),
                    torch.Size([-1, 256, 13, 13]),
                    torch.Size([-1, 256, 6, 6])
                ]
                new_sizes = torch.Size([-1, 32, 4, 4])
                
            self.attention = DAAttention(atten_sizes, mode=mode, new_size=new_sizes).to(device)
            self.att_optimizer =  getattr(optim, optim_config['name'])(self.attention.parameters(), **optim_config['params'])

        super(AlignPrivateTrainer,self).__init__(train_model, train_loader, epoch, length, optim_config, criterion, device, test_loader)
    
    def train(self, logger, accrecorder=None, trace_acc=None, print_step=None, outer_epoch=0, trace_domains=None, lam=1):
        self.outer_epoch = outer_epoch
        self.lam = lam
        super(AlignPrivateTrainer,self).train(logger, accrecorder, trace_acc, print_step, trace_domains)
    
    def cal_step(self,it):
        return self.outer_epoch * self._epoch * self._length + self.epoch * self._length + it
    
    def cal_loss(self, x, y):
        fix_fs = self._fix_model(x)
        train_fs = self._model(x)
        cls_loss = self._criterion(train_fs[0],y)
        transfer_loss = 0
        addition_loss = 0

        if self.flag == 'bone':
            addition_loss += self.mkmmd_loss(fix_fs[0], train_fs[0])
        elif self.flag == 'soft':
            addition_loss += self.mkmmd_loss(fix_fs[1], train_fs[1])
        elif self.flag == 'bone_soft':
            addition_loss += self.mkmmd_loss(fix_fs[0], train_fs[0])
            addition_loss += self.mkmmd_loss(fix_fs[1], train_fs[1])
        else:
            pass

        if self.mode in ["noa_mmmd", 'all', 'pam', 'cam']:
            fix_fs, _, train_fs = self.attention(fix_fs[2:], train_fs[2:])
            for i in range(len(fix_fs)):
                for j in range(len(train_fs)):
                    transfer_loss += self.mkmmd_loss(fix_fs[i], train_fs[j])

        elif self.mode in ['alpha', 'alpha_pam', 'alpha_cam', 'alpha_all']:
            fix_fs,alpha, train_fs = self.attention(fix_fs[2:], train_fs[2:])
            # print(alpha)
            for i in range(len(train_fs)):
                for j in range(len(fix_fs)):
                    transfer_loss += (alpha[i,j].item()*self.mkmmd_loss(train_fs[i], fix_fs[j]))
        elif self.mode in ['noa_mmd']:
            fix_fs, _, train_fs = self.attention(fix_fs[2:], train_fs[2:])
            for i in range(len(fix_fs)):
                transfer_loss += self.mkmmd_loss(fix_fs[i], train_fs[i])


        return cls_loss + self.lam*(transfer_loss + addition_loss)

    def _epoch_train(self):
        self._model.train()
        for it in range(self._length):
            data,label = next(self._loader)
            data,label = data.to(self._device), label.to(self._device)

            loss = self.cal_loss(data,label)
            self._optimizer.zero_grad()
            self.att_optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self.att_optimizer.step()

            """Trace acc."""
            if self._trace_acc and (it + 1)%self._print_step == 0:
                self._model.eval()
                self.trace_acc(self.cal_step(it))
                self._model.train()
