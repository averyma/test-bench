import os
import sys
import logging

import torch
import numpy as np

from src.attacks import pgd_rand
from src.train import train_standard, train_adv, train_soar
from src.evaluation import test_clean, test_adv
from src.args import get_args
from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, saveCheckpoint
from src.utils_general import seed_everything, get_model, get_optim
from src.utils_plot import plot_standard_adv, plot_soar

def train(args, epoch, logger, loader, model, opt, device):
    """perform one epoch of training."""
    if args.method == "standard":
        train_log = train_standard(logger, epoch, loader, model, opt, device)
        logger.add_scalar("train_acc_ep", train_log[0], epoch+1)
        logger.add_scalar("train_loss_ep", train_log[1], epoch+1)
        logging.info(
            "Epoch: [{0}]\t"
            "Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                epoch,
                loss=train_log[1],
                acc=train_log[0]))

    elif args.method == "adv":
        train_log = train_adv(logger, epoch, loader, args.pgd_steps, model, opt, device)
        logger.add_scalar("train_acc_ep", train_log[0], epoch+1)
        logger.add_scalar("train_loss_ep", train_log[1], epoch+1)
        logging.info(
            "Epoch: [{0}]\t"
            "Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                epoch,
                loss=train_log[1],
                acc=train_log[0]))

    elif args.method == "soar":
        soar_param = {
            "init": args.init,
            "grad_clip": args.grad_clip,
            "norm_clip": args.norm_clip,
            "eps": args.epsilon,
            "lambbda": args.lambbda,
            "step_size": args.step_size}

        train_log = train_soar(logger, epoch, loader, model, soar_param, opt, device)
        logger.add_scalar("train_acc_ep", train_log[0], epoch+1)
        logger.add_scalar("train_loss_ep", train_log[1], epoch+1)
        logger.add_scalar("train_reg_ep", train_log[2], epoch+1)
        logging.info(
            "Epoch: [{0}]\t"
            "Loss: {loss:.6f}\t"
            "Regularizer: {reg: 6f}\t"
            "Accuracy: {acc:.2f}".format(
                epoch,
                loss=train_log[1],
                acc=train_log[0],
                reg=train_log[2]))
    else:
        raise  NotImplementedError("Training method not implemented!")
    return train_log

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    attack_param = {"ord":np.inf, "epsilon": 8./255., "alpha":2./255., "num_iter": 20, "restart": 1}

    args = get_args()
    log_path = args.log_dir + "/" + str(args.job_id)
    logger = metaLogger(log_path)
    logging.basicConfig(
        filename=log_path + "/log.txt",
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    seed_everything(args.seed)
    train_loader, test_loader = load_dataset(args.batch_size)

    model = get_model(args, device)
    opt, lr_scheduler = get_optim(model, args)
    ckpt_epoch = 0

    ckpt_location = os.path.join(args.ckpt_dir, "custome_ckpt.pth")
    if os.path.exists(ckpt_location):
        ckpt = torch.load(ckpt_location)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        ckpt_epoch = ckpt["epoch"]
        if lr_scheduler:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        print("LOADED CHECKPOINT")

    for _epoch in range(ckpt_epoch, args.epoch):
        train_log = train(args, _epoch, logger, train_loader, model, opt, device)

        test_log = test_clean(test_loader, model, device)
        adv_log = test_adv(test_loader, model, pgd_rand, attack_param, device)

        logger.add_scalar("test_acc", test_log[0], _epoch+1)
        logger.add_scalar("test_loss", test_log[1], _epoch+1)
        logger.add_scalar("pgd20_acc", adv_log[0], _epoch+1)
        logger.add_scalar("pgd20_loss", adv_log[1], _epoch+1)
        logging.info(
            "Test set: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=test_log[1],
                acc=test_log[0]))
        logging.info(
            "PGD20: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=adv_log[1],
                acc=adv_log[0]))

        if lr_scheduler:
            lr_scheduler.step()

        if (_epoch+1) % args.ckpt_freq == 0:
            saveCheckpoint(args.ckpt_dir, "custome_ckpt.pth", model, opt, _epoch, lr_scheduler)
        if args.method == "soar":
            fig = plot_soar(logger.log_dict)
        else:
            fig = plot_standard_adv(logger.log_dict)

        logger.log_fig("fig", fig, _epoch+1)
        fig.savefig("./exp/"+ str(args.job_id) +"/main.png")
        logger.save_log()

if __name__ == "__main__":
    main()
