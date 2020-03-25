import yaml
import argparse
import os
from src.utils_general import DictWrapper
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method",
                        default=argparse.SUPPRESS)
    parser.add_argument("--dataset",
                        default=argparse.SUPPRESS)
    parser.add_argument("--arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--log_dir",
    			default=argparse.SUPPRESS)
    parser.add_argument("--pretrain",
    			default=argparse.SUPPRESS)
    
    # hyper-param for optimization
    parser.add_argument("--optim",
    			default=argparse.SUPPRESS)
    parser.add_argument("--lr",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lr_update",
    			default=argparse.SUPPRESS, nargs="+", type=float)
    parser.add_argument("--momentum",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--weight_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--batch_size",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--seed",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--epoch",
    			default=argparse.SUPPRESS, type=int)

    # hyper-param for job_id, and ckpt
    parser.add_argument("--job_id",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--ckpt_dir",
    			default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_freq",
    			default=argparse.SUPPRESS, type=int)
    
    # for adversarial training, we just need to specify pgd steps
    parser.add_argument("--pgd_steps",
    			default=argparse.SUPPRESS, type=int)

    # hyper-param for SOAR
    parser.add_argument("--init",
    			default=argparse.SUPPRESS)
    parser.add_argument("--step_size",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--epsilon",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lambbda",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--grad_clip",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--norm_clip",
    			default=argparse.SUPPRESS, nargs="+", type=float)

    args = parser.parse_args()

    return args

def make_dir(args):
    _dir = str(args["log_dir"]) + str(args["job_id"])
    try:
        os.makedirs(_dir)
    except os.error:
        pass

    if not os.path.exists(_dir + "/hyper_params.yaml"):
        f = open(_dir + "/hyper_params.yaml" ,"w+")
        f.close()
    with open(_dir + "/hyper_params.yaml", "a") as f:
        # f.write(str(args).replace(", ", "\n") + "\n\n")
        f.write(yaml.dump(args))

def get_default(yaml_path):
    default = {}
    with open(yaml_path, 'r') as handle:
        default = yaml.load(handle, Loader=yaml.FullLoader)
    return default 

def get_args():
    args = parse_args()
    default = get_default('options/default.yaml')
    # opt_s = yaml_opt(os.path.join('options/{}/{}'.format(args.dataset,
                                                         # args.path_opt)))
    # opt.update(opt_s)
    default.update(vars(args).items())
    make_dir(default)
    args_dict = DictWrapper(default)

    # opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    # if opt.g_batch_size == -1:
        # opt.g_batch_size = opt.batch_size
    # opt.adam_betas = make_tuple(opt.adam_betas)
    return args_dict

