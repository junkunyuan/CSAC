import json,os
import numpy as np
import os.path as osp
class AccRecorder(object):
    """Save acc on all domains."""

    def __init__(self, name, decimal=4):
        self.decimal = decimal
        self.name = name
        self.reset()

    def reset(self):
        self.accs = {}
        self.iter_steps = {}
        self.count = 0

    def updata(self, kwages):
        self.count+=1
        for k, (step, acc) in kwages.items():
            self.accs.setdefault(k,[])
            self.iter_steps.setdefault(k,[])
            if self.decimal:
                self.accs[k].append(round(float(acc), self.decimal))
            else:
                self.accs[k].append(acc)
            self.iter_steps[k].append(step)
    
    def save(self, pth):
        savefile = {"name": self.name,
                    "count": self.count,
                    "accs": self.accs, 
                    "step": self.iter_steps}
        if not osp.exists(osp.split(pth)[0]):
            os.makedirs(osp.split(pth)[0])
        if pth[-4:] == "json":
            with open(pth, "w", encoding="utf-8") as f:
                json.dump(savefile, f, indent=4)
        elif pth[-4:] == ".npy":
            np.save(pth, savefile)
    
    def load(self,pth):
        with open(pth,"r",encoding='utf-8') as f:
            load_dict = json.load(f)
            self.name = load_dict["name"]
            self.count = load_dict["count"]
            self.accs = load_dict["accs"]
            self.iter_steps = load_dict["step"]
    
    def merge(self,pth,name=None):
        with open(pth, 'r') as f:
            load_dict = json.load(f)
            print(load_dict.keys(), self.accs.keys())
            if name:
                self.name = name
            else:
                self.name = self.name+"_" + load_dict['name']
            self.count += load_dict['count']
            for key in self.accs.keys():
                self.accs[key] += load_dict['accs'][key]
            new_steps = [item + self.iter_steps[-1] for item in load_dict['step']]
            # print(new_steps)
            self.iter_steps += new_steps

    def merge_r(self,recorder):
        self.count += recorder.count
        for key in self.accs.keys():
            self.accs[key] += recorder.accs[key]
        new_steps = [item + self.iter_steps[-1] for item in recorder.iter_steps]
        self.iter_steps += new_steps
    
    def __str__(self):
        fmtstr = 'name: {name}\ncount: {count}\naccs:{accs}\nsteps:{iter_steps}\n'
        return fmtstr.format(**self.__dict__)

class ShareAccRecorder(AccRecorder):
    def __init__(self, name, *domains, decimal=None):
        self.domains = list(domains)
        super().__init__(name, decimal)
    
    def reset(self):
        super().reset()
        for domain in self.domains:
            self.accs[domain] = []
        self.iter_steps = []

    def updata(self, iter_step, kwages):
        self.count+=1

        self.iter_steps.append(iter_step)
        for k, v in kwages.items():
            if self.decimal:
                self.accs[k].append(round(float(v), self.decimal))
            else:
                self.accs[k].append(v)
            assert len(self.accs[k]) == self.count
    
    def pure(self):
        accs = {}
        for k,v in self.accs.items():
            accs[k] = []
        steps = []
        for i,step in enumerate(self.iter_steps):
            if step < 0:
                continue
            steps.append(step)
            for k,v in self.accs.items():
                accs[k].append(self.accs[k][i])
        self.accs = accs
        self.iter_steps = steps
    
    def addkey(self,key):
        self.domains.append(key)
        self.accs[key] = []

        