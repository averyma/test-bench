import os
import sys
import logging

import torch
import numpy as np

from src.train import train_standard
from src.evaluation import test_clean
from src.args import get_args, print_args
from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim
import ipdb


best_acc1 = 0

def train(args, epoch, logger, loader, model, opt, lr_scheduler, device):
    """perform one epoch of training."""
    if args.method == "standard":
        train_log = train_standard(loader, model, opt, device, epoch, lr_scheduler)

    else:
        raise  NotImplementedError("Training method not implemented!")

    logger.add_scalar("train/acc_ep", train_log[0], epoch)
    logger.add_scalar("train/loss_ep", train_log[1], epoch)
    logging.info(
        "Epoch: [{0}]\t"
        "Loss: {loss:.6f}\t"
        "Accuracy: {acc:.2f}".format(
            epoch,
            loss=train_log[1],
            acc=train_log[0]))

    return train_log

def main():

    global best_acc1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()
    print_args(args)

    logger = metaLogger(args)
    logging.basicConfig(
        filename=args.j_dir+ "/log/log.txt",
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    seed_everything(args.seed)
    train_loader, test_loader, _, _ = load_dataset(
                                    args.dataset,
                                    args.batch_size,
                                    args.op_name,
                                    args.op_prob,
                                    args.op_magnitude)

    model = get_model(args, device)
    opt, lr_scheduler = get_optim(model, args)
    ckpt_epoch = 1

    ckpt_dir = args.j_dir+"/"+str(args.j_id)+"/"
    if logger.ckpt_status in ['curr', 'prev']:
        ckpt_location = os.path.join(ckpt_dir, "custom_ckpt_"+logger.ckpt_status+".pth")
        if os.path.exists(ckpt_location):
            ckpt = torch.load(ckpt_location)
            model.load_state_dict(ckpt["state_dict"])
            opt.load_state_dict(ckpt["optimizer"])
            ckpt_epoch = ckpt["epoch"]
            best_acc1 = ckpt["best_acc1"]
            if lr_scheduler is not None:
                for _dummy in range(ckpt_epoch-1):
                    lr_scheduler.step()
                # lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            print("LOADED CHECKPOINT")
    else:
        print("CHECKPOINT STATUS: {}".format(logger.ckpt_status))

    actual_trained_epoch = args.epoch

    for _epoch in range(ckpt_epoch, args.epoch+1):
        train_log = train(args, _epoch, logger, train_loader, model, opt, lr_scheduler, device)

        # evaluation on testset
        test_log = test_clean(test_loader, model, device)

        is_best = test_log[0] > best_acc1
        best_acc1 = max(test_log[0], best_acc1)

        logger.add_scalar("lr", opt.param_groups[0]['lr'], _epoch)
        logger.add_scalar("test/top1_acc", test_log[0], _epoch)
        logger.add_scalar("test/top5_acc", test_log[2], _epoch)
        logger.add_scalar("test/loss", test_log[1], _epoch)
        logger.add_scalar("test/best_top1_acc", best_acc1, _epoch)
        logging.info(
            "Test set: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=test_log[1],
                acc=test_log[0]))

        # checkpointing for preemption
        if _epoch % args.ckpt_freq == 0:
            # since preemption would happen in the next epoch, so we want to start from {_epoch+1}
            rotateCheckpoint(ckpt_dir, "custom_ckpt", model, opt, _epoch+1, best_acc1)
            logger.save_log()

        # save best model (after 75% way through the training)
        if is_best and _epoch > int(args.epoch*3/4):
            saveModel(args.j_dir+"/model/", "best_model", model.state_dict())

        # Early terminate training when half way thru training and test accuracy still below 20%
        if np.isnan(train_log[1]) or (_epoch > int(args.epoch/2) and best_acc1 < 20):
            actual_trained_epoch = _epoch
            saveModel(args.j_dir+"/model/", "final_model", model.state_dict())
            break # break the training for-loop

    if actual_trained_epoch == args.epoch:
        # at the end of training, save after test set evaluation, since Autoattack sometimes fail
        saveModel(args.j_dir+"/model/", "final_model", model.state_dict())
        if args.fancy_eval:
            # evaluate using the best checkpoint
            try:
                ckpt_best_model = torch.load(args.j_dir+"/model/best_model.pt")
                model.load_state_dict(ckpt_best_model)
            except:
                print("Problem loading best_model ckpt at {}/model/best_model.pt!".format(args.j_dir))
                print("Evaluating using the model from the last epoch!")

            if best_acc1 > eval_threshold:
                # place some evaluation here
                pass

    # upload runs to wandb:
    if args.enable_wandb:
        save_wandb_retry = 0
        save_wandb_successful = False
        while not save_wandb_successful and save_wandb_retry < 5:
            print('Uploading runs to wandb...')
            try:
                wandb_logger = wandbLogger(args)
                wandb_logger.upload(logger, actual_trained_epoch)
            except:
                save_wandb_retry += 1
                print('Retry {} times'.format(save_wandb_retry))
            else:
                save_wandb_successful = True

        if not save_wandb_successful:
            print('Failed at uploading runs to wandb.')

    logger.save_log(is_final_result=True)

    # delete slurm checkpoints
    delCheckpoint(args.j_dir, args.j_id)

if __name__ == "__main__":
    main()
