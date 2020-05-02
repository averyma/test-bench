import time
import os
from collections import defaultdict
import torch
from src.save_function import checkpoint_save
from torch.utils.tensorboard import SummaryWriter

def rotateCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, lr_scheduler):
    ckpt_curr = os.path.join(ckpt_dir, ckpt_name+"_curr.pth")
    ckpt_prev = os.path.join(ckpt_dir, ckpt_name+"_prev.pth")

    # no existing ckpt
    if not (os.path.exists(ckpt_curr) or os.path.exists(ckpt_prev)):
        saveCheckpoint(ckpt_dir, 
                       ckpt_name+"_curr.pth",
                       model,
                       opt,
                       epoch,
                       lr_scheduler)

    elif os.path.exists(ckpt_curr):
        # overwrite ckpt_prev with ckpt_curr
        cmd = "cp -r {} {}".format(ckpt_curr, ckpt_prev)
        os.system(cmd)
        saveCheckpoint(ckpt_dir,
                       ckpt_name+"_curr.pth",
                       model,
                       opt,
                       epoch,
                       lr_scheduler)

def saveCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, lr_scheduler):
    if lr_scheduler:
        checkpoint_save({"state_dict": model.state_dict(),
                         "optimizer": opt.state_dict(),
                         "epoch": epoch+1,
                         "lr_scheduler":lr_scheduler.state_dict()},
                        ckpt_dir, ckpt_name)
    else:
        checkpoint_save({"state_dict": model.state_dict(),
                         "optimizer": opt.state_dict(),
                         "epoch": epoch+1}, ckpt_dir, ckpt_name)

    print("SAVED CHECKPOINT")

class metaLogger(object):
    def __init__(self, log_path, flush_sec=5):
        self.log_path = log_path+"/log/"
        self.tb_path = log_path+"/tb/"
        self.ckpt_status = "curr"
        self.log_dict = self.load_log(self.log_path)
        self.writer = SummaryWriter(log_dir=self.tb_path, flush_secs=flush_sec)

    def load_log(self, log_path):
        try:
            log_dict = torch.load(log_path + "/log_curr.pth")
        except FileNotFoundError:
            log_dict = defaultdict(lambda: list())
        except TypeError:
            log_dict = torch.load(log_path + "/log_prev.pth")
            self.ckpt_status = "prev"
        return log_dict

    def add_scalar(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        self.log_dict[name] += [(time.time(), int(step), float(val))]

    def add_scalars(self, name, val_dict, step):
        self.writer.add_scalars(name, val_dict, step)
        for key, val in val_dict.items():
            self.log_dict[name+key] += [(time.time(), int(step), float(val))]

    def add_figure(self, name, val, step):
        self.writer.add_figure(name, val, step)
        val.savefig(self.log_path + "/" + name + ".png")

    def save_log(self):
        try:
            os.makedirs(self.log_path)
        except os.error:
            pass

        log_curr = os.path.join(self.log_path, "log_curr.pth")
        log_prev = os.path.join(self.log_path, "log_prev.pth")

        # no existing logs
        if not (os.path.exists(log_curr) or os.path.exists(log_prev)):
            torch.save(dict(self.log_dict), log_curr)

        elif os.path.exists(log_curr):
            # overwrite log_prev with log_curr
            cmd = "cp -r {} {}".format(log_curr, log_prev)
            os.system(cmd)
            torch.save(dict(self.log_dict), log_curr)

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
