import yaml
import argparse
import os
from src.utils_general import DictWrapper
import distutils.util
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method",
                        default=argparse.SUPPRESS)
    parser.add_argument("--dataset",
                        default=argparse.SUPPRESS)
    parser.add_argument("--arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--pretrain",
    			default=argparse.SUPPRESS)
    
    # hyper-param for optimization
    parser.add_argument("--lr",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lr_scheduler_type",
    			default=argparse.SUPPRESS)
    parser.add_argument("--momentum",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--weight_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--nesterov",
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)
    parser.add_argument("--warmup",
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)
    parser.add_argument("--warmup_multiplier",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--warmup_epoch",
    			default=argparse.SUPPRESS, type=int)

    parser.add_argument("--batch_size",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--seed",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--epoch",
    			default=argparse.SUPPRESS, type=int)

    # hyper-param for job_id, and ckpt
    parser.add_argument("--j_dir", required=True)
    parser.add_argument("--j_id",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--ckpt_freq",
    			default=argparse.SUPPRESS, type=int)

    # setup wandb logging
    parser.add_argument("--wandb_project",
    			default=argparse.SUPPRESS)
    parser.add_argument('--enable_wandb',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)

    parser.add_argument('--input_normalization',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)
    parser.add_argument('--enable_batchnorm',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)

    # imagenet training
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
    parser.add_argument('--optimize_cluster_param',
                        default=False, type=distutils.util.strtobool)
    parser.add_argument("--print_freq",
                        default=argparse.SUPPRESS, type=int)

    parser.add_argument('--fancy_eval',
                        default=False, type=distutils.util.strtobool)

    args = parser.parse_args()

    return args

def make_dir(args):
    _dir = str(args["j_dir"]+"/config/")
    try:
        os.makedirs(_dir)
    except os.error:
        pass

    if not os.path.exists(_dir + "/config.yaml"):
        f = open(_dir + "/config.yaml" ,"w+")
        f.write(yaml.dump(args))
        f.close()

def get_default(yaml_path):
    default = {}
    with open(yaml_path, 'r') as handle:
        default = yaml.load(handle, Loader=yaml.FullLoader)
    return default 

def get_args():
    args = parse_args()
    if args.dataset.startswith('cifar') or args.dataset == 'svhn':
        default = get_default('options/default_cifar.yaml')
    elif args.dataset == 'imagenet':
        default = get_default('options/default_imagenet.yaml')

    default.update(vars(args).items())
    
    make_dir(default)

    if args.dataset.startswith('cifar'):
        args_dict = DictWrapper(default)
        return args_dict

    return argparse.Namespace(**default)

def print_args(args):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
