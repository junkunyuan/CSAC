import os, time
import os.path as osp
from utils.utils import save_config

class Logger(object):
    def __init__(self,config):
        self.config = config
        self.outdir = self.prepare_outdir()
        self.logfile = open(osp.join(self.outdir,"logs.txt"),'w')
    
    def prepare_outdir(self):
        now_time = time.strftime('%Y-%m-%d_%H%M%S',time.localtime())
        self.pre_model_dir = osp.join(
            self.config['args']['outroot'],
            self.config['args']['dataset'] + '_'+self.config['args']['net'],
            self.config['args']['test']
        )
        outdir = osp.join(
            self.pre_model_dir,
            now_time
        )

        if not osp.exists(outdir):
            os.makedirs(outdir)
        return outdir

    def append(self, log_str):
        self.logfile.write(log_str+'\n')
        self.logfile.flush()
    
    def save_config(self):
        self.config['outdir'] = self.outdir
        save_config(self.config,osp.join(self.outdir,'config.json'))

class AverageMeter(object):
    """
    Computes and stores the average and current vlaue.
    """

    def __init__(self, name, decimal=4):
        self.name = name
        self.decimal = decimal
        self.fmt = ':.{}f'.format(decimal)
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, *vals):
        self.val = vals
        self.sum += sum(vals)
        self.count += len(vals)
        self.avg = self.sum/self.count

    def __str__(self):
        if self.count == 1:
            valstr = ('{'+self.fmt+'}').format(self.val[0])
        else:
            valstr = str([round(v, self.decimal) for v in self.val])
        fmtstr = '{name} '+valstr+' ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)