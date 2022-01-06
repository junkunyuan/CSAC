import sys
sys.path.append('.')
from models.script.kernels import GaussianKernel
from models.script.multikernel import MultipleKernelMaximumMeanDiscrepancy
import copy
import torch
import os
import os.path as osp
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import datasets.dg as DGData
import utils.transformers as DGTransformer
import models.backbone as Backbone
from models.fullnet import FLDGFullNet
from utils.utils import load_yaml
from utils.loss import KLLoss
from utils.parse_params import args_parser, merge_config
from utils.script import Trainer, Logger, ShareAccRecorder, ForeverDataIterator, AlignPrivateTrainer


class FLDG(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.pre_model_dir = logger.pre_model_dir
        self.outdir = logger.outdir
        self.device = config['args']['device']
        self.logger.save_config()
        self.weightprint = open(osp.join(self.outdir, 'weightprint.txt'), 'w')

        all_num = 0
        for cli in config['args']['cli_datas']:
            all_num += config['dataset']['nums'][cli]

        self.ratios = {}
        for cli in config['args']['cli_datas']:
            self.ratios[cli] = round(
                config['dataset']['nums'][cli] / all_num, 4)

        print(self.ratios)

        """init model(clients,public)"""
        self.models = {}
        self.accrecorders = {}
        for name in self.config['args']['cli_datas']+['public']:
            backbone = Backbone.__dict__[
                config['args']['net']](pretrained=True)
            self.models[name] = FLDGFullNet(
                backbone, config['dataset']['class_num'], config['model']['bottleneck_dim']).to(self.device)
            self.accrecorders[name] = self.__init_accrecorder__(name)

        """prepare dataset and dataloader"""
        self.prepare_loader(config)

        """init loss function"""
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = KLLoss()
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=3**k) for k in range(-3, 2)],
            linear=False, quadratic_program=False
        )

    def clienttrainer(self, cli):
        if self.config['args']['fix'] == 'private':
            fix_model = self.models[cli]
            train_model = copy.deepcopy(self.models['public'])
        elif self.config['args']['fix'] == 'public':
            fix_model = self.models['public']
            train_model = self.models[cli]
        trainer = AlignPrivateTrainer(self.mkmmd_loss,
                                      fix_model,
                                      train_model,
                                      self.client_train_loaders[cli],
                                      self.config['client']['epoch'],
                                      self.config['client']['length'],
                                      self.config['optim'],
                                      self.criterion_ce,
                                      self.device,
                                      self.test_loaders,
                                      self.config['args']['mode'],
                                      self.config['args']['addition'],
                                      self.config['args']['net'])
        return trainer

    def train(self):
        max_test = 0
        temp_models = {}
        for epoch in range(self.config['args']['round']):
            """train in one epoch"""
            self.epoch = epoch

            """init public model"""
            weights, ratios = [], []
            for cli in self.config['args']['cli_datas']:
                if self.config['args']['fix'] == 'private':
                    if epoch == 0:
                        weights.append(copy.deepcopy(
                            self.models[cli].state_dict()))
                    else:
                        weights.append(copy.deepcopy(
                            temp_models[cli].state_dict()))
                elif self.config['args']['fix'] == 'public':
                    weights.append(copy.deepcopy(
                        self.models[cli].state_dict()))
                ratios.append(self.ratios[cli])

            if self.config['args']['fusion'] == 'fedavg':
                public_weight = self._fedavg_(weights, ratios)
            elif self.config['args']['fusion'] == 'ouravg':
                public_weight = self._ourfedavg_(weights)
            elif self.config['args']['fusion'] in ['l1', 'l2', 'cosion']:
                public_weight = self._weightavg_(weights)

            self.models['public'].load_state_dict(copy.deepcopy(public_weight))

            for cli in self.config['args']['cli_datas']:
                self.logger.append(
                    "{} client:{} {}".format('*' * 50, cli, '*' * 50))
                trainer = self.clienttrainer(cli)
                trainer.train(self.logger,
                              accrecorder=self.accrecorders[cli],
                              outer_epoch=epoch,
                              trace_acc=self.config['args']['trace_acc'],
                              print_step=self.config['args']['print_step'], lam=self.config['args']['lambda'])
                if self.config['args']['fix'] == 'private':
                    temp_models[cli] = trainer.model

            self.logger.append("{} public test {}".format('*'*50, '*'*50))

            self.models['public'].eval()
            acc = self.test(
                self.models['public'], *list(self.config['dataset']['domains'].keys()))
            self.models['public'].train()
            max_test = max(max_test, acc[self.config['args']['test']])

            self.logger.append(
                'epoch/Epoch: {:>4d}/{:>4d} acc:{}'.format(epoch, self.config['args']['round'], str(acc)))
            if self.config['args']['trace_acc']:
                self.accrecorders['public'].updata(self.epoch, acc)

        """save accrecorder"""
        for k, recorder in self.accrecorders.items():
            if recorder:
                recorder.save(osp.join(self.outdir, '{}.json'.format(k)))

        self.logger.append("public max acc:{}".format(max_test))

    def test(self, model, *domains):
        accs = {}
        for domain in domains:
            total = 0
            with torch.no_grad():
                correct = 0
                for it, (data, label) in enumerate(self.test_loaders[domain]):
                    data, label = data.to(self.device), label.to(self.device)
                    class_logit = model(data)
                    _, cls_pred = class_logit[0].max(dim=1)
                    correct += torch.sum(cls_pred == label.data)
                    total += data.size(0)
            accs[domain] = round((float(correct)/total)*100, 4)
        return accs

    def pre_train(self):
        trainers = {}
        for cli in self.config['args']['cli_datas']:
            trainers[cli] = Trainer(self.models[cli],
                                    self.client_train_loaders[cli],
                                    self.config['args']['epoch'],
                                    self.config['args']['length'],
                                    self.config['optim'],
                                    self.criterion_ce, self.device, self.test_loaders)
            self.logger.append(
                "{}  train client {}  {}".format('*' * 50, cli, '*' * 50))
            trainers[cli].train(self.logger,
                                accrecorder=self.accrecorders[cli],
                                trace_acc=self.config['args']['trace_acc'],
                                print_step=self.config['args']['print_step'])

            print(self.models[cli] == trainers[cli].model)
            print(self.models[cli].parameters())
            print(trainers[cli].model.parameters())
            torch.save(self.models[cli].state_dict(), osp.join(
                self.pre_model_dir, '{}.pt'.format(cli)))

        """save accrecorder"""
        for cli in self.config['args']['cli_datas']:
            if self.accrecorders[cli]:
                self.accrecorders[cli].save(
                    osp.join(self.pre_model_dir, '{}.json'.format(cli)))
                self.accrecorders[cli].reset()

    def run(self):
        print("pre-train")
        if self.config['args']['pre_train'] or not osp.exists(osp.join(self.pre_model_dir, '{}.pt'.format(self.config['args']['cli_datas'][0]))):
            self.logger.append("{} pre-train {}".format('*' * 50, '*' * 50))
            self.pre_train()

        self.logger.append("{} train {}".format('*' * 50, '*' * 50))
        for cli in self.config['args']['cli_datas']:
            self.models[cli].load_state_dict(torch.load(
                osp.join(self.pre_model_dir, '{}.pt'.format(cli))))

        self.train()
        torch.save(self.models['public'].state_dict(),
                   osp.join(self.outdir, 'public.pt'))

    def prepare_loader(self, config):
        self.test_loaders = {}

        tf_ge = DGTransformer.__dict__[
            config['args']['dataset']](config['process'])
        test_tf, train_tf = tf_ge.test_transformer(), tf_ge.train_transformer()
        data_ge = DGData.__dict__[config['args']
                                  ['dataset']](config, train_tf, test_tf)

        test_dataset = data_ge.test_dataset(config['args']['test'])
        test_loader = DataLoader(
            test_dataset, batch_size=config['args']['bs'], shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
        self.test_loaders[config['args']['test']] = test_loader

        client_train_loaders = {}
        for cdata in config['args']['cli_datas']:
            train_dset, test_dset = data_ge.train_dataset(cdata, True)
            cli_train_loader = DataLoader(
                train_dset, batch_size=config['client']['bs'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            cli_test_loader = DataLoader(
                test_dset, batch_size=config['client']['bs'], shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

            client_train_loaders[cdata] = ForeverDataIterator(cli_train_loader)
            self.test_loaders[cdata] = cli_test_loader

        self.client_train_loaders = client_train_loaders

    def __init_accrecorder__(self, name):
        if self.config['args']['trace_acc']:
            accrecorder = ShareAccRecorder(
                name, *list(self.config['dataset']['domains'].keys()))
        else:
            accrecorder = None
        return accrecorder

    def _ourfedavg_(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w))
        return w_avg

    def _fedavg_(self, w, ratios):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k].fill_(0)
            for i in range(0, len(w)):
                if not w[i][k].size():
                    w_avg[k] = w[i][k]
                else:
                    w_avg[k] += (w[i][k]*ratios[i])

        return w_avg

    def _weightavg_(self, w):
        w_avg = self._ourfedavg_(w)

        for k in w_avg.keys():
            distance = torch.Tensor([1] * len(w))
            for i in range(len(w)):
                distance[i] = self.L(w_avg[k], w[i][k])
            if torch.sum(distance) == 0:
                coeff = distance
                coeff.fill_(0).long()
            else:
                coeff = distance/sum(distance)
            self.weightprint.write(str(k) + " " + str(coeff) + '\n')
            self.weightprint.flush()
            w_avg[k].fill_(0)
            for i in range(len(w)):
                if coeff[i] == 0:
                    w_avg[k] = w[i][k]
                else:
                    w_avg[k] += (coeff[i]*w[i][k])
        self.weightprint.write('\n')
        self.weightprint.flush()
        return w_avg

    def L(self, w1, w2):
        if self.config['args']['fusion'] == 'l2':
            return torch.sqrt(torch.sum((w1 - w2) ** 2)).data
        elif self.config['args']['fusion'] == 'l1':
            return torch.sum(torch.abs(w1-w2)).data
        elif self.config['args']['fusion'] == 'cosine':
            return torch.cosine_similarity(w1.view(1, -1), w2.view(1, -1))


def office8(config):
    config['args']['test'] = 'R'
    clients = ['a', 'C', 'P', 'w', "A", 'c']

    for i in range(1, len(clients) + 1):
        config['args']['cli_datas'] = clients[:i]
        logger = Logger(config)
        fldg = FLDG(config, logger)
        fldg.run()


def main():
    args = args_parser()
    dataset_config = load_yaml(
        osp.join('configs/datasets', args.exp_type, args.dataset + '.yaml'))
    trainer_config = load_yaml(
        osp.join('configs/trainers', args.exp_type, args.dataset + '.yaml'))

    config = merge_config(args, dataset_config, trainer_config)
    all_domains = list(config['dataset']['domains'].keys())
    config['args']['dataset'] = config['args']['dataset'].replace('-', '_')

    if config['args']['dataset'] == 'office-8':
        office8(config)
        return None

    # config['args']['main_name'] = 'new lab'
    # config['args']['outroot'] = 'log1'
    # config['args']['lambda'] = 0.6
    # config['args']['fusion'] = 'l2'
    # config['args']['addition'] = 'bone'
    # config['args']['mode'] = 'alpha'

    if config['args']['test'] == '*':
        for test in all_domains:
            config['args']['test'] = test
            config['args']['cli_datas'] = [
                item for item in all_domains if item != test]
            logger = Logger(config)

            fldg = FLDG(config, logger)
            fldg.run()
    else:
        test = config['args']['test']
        config['args']['cli_datas'] = [
            item for item in all_domains if item != test]
        logger = Logger(config)

        fldg = FLDG(config, logger)
        fldg.run()


if __name__ == '__main__':
    main()
