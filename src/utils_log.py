import time
import os
from collections import defaultdict
import torch
from src.save_function import checkpoint_save
from torch.utils.tensorboard import SummaryWriter

def saveCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, lr_scheduler):

    checkpoint_save({"state_dict": model.state_dict(),
                     "optimizer": opt.state_dict(),
                     "epoch": epoch+1}, ckpt_dir, ckpt_name)

    if lr_scheduler:
        checkpoint_save({"lr_scheduler": lr_scheduler.state_dict()},
                        ckpt_dir, ckpt_name)
    print("SAVED CHECKPOINT")

class metaLogger(object):
    def __init__(self, log_path, flush_sec=5):
        self.log_path = log_path
        self.log_dict = self.load_log(self.log_path)
        self.writer = SummaryWriter(log_dir=self.log_path, flush_secs=flush_sec)

    def load_log(self, log_path):
        try:
            log_dict = torch.load(log_path + "/log.pth.tar")
        except FileNotFoundError:
            log_dict = defaultdict(lambda: list())
        return log_dict

    def log_value(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        self.log_dict[name] += [(time.time(), int(step), float(val))]

    def add_scalar(self, name, val, step):
        self.log_value(name, val, step)

    def log_fig(self, name, val, step):
        self.writer.add_figure(name, val, step)

    def save_log(self, filename="log.pth.tar"):
        try:
            os.makedirs(self.log_path)
        except os.error:
            pass
        torch.save(dict(self.log_dict), self.log_path+'/'+filename)

    # def log_obj(self, name, val):
        # self.logobj[name] = val

    # def log_objs(self, name, val, step=None):
        # self.logobj[name] += [(time.time(), step, val)]

    # def log_vector(self, name, val, step=None):
        # name += '_v'
        # if step is None:
            # step = len(self.logobj[name])
        # self.logobj[name] += [(time.time(), step, list(val.flatten()))]

    def close(self):
        self.writer.close()
