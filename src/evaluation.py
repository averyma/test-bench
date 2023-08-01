import torch
import torch.nn as nn
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
import numpy as np

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_clean(loader, model, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            y_hat = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            batch_acc = accuracy(y_hat, y, topk=(1,5))
            batch_correct = batch_acc[0].sum().item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].sum().item()*x.shape[0]/100
        # print(accuracy(y_hat, y, topk=(1,5)), batch_correct/128*100)
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]
    # ipdb.set_trace()
    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss, test_acc_5

def test_gaussian(loader, model, var, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
        
        noise = (var**0.5)*torch.randn_like(x, device = x.device)

        with torch.no_grad():
            y_hat = model(x+noise)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            batch_acc = accuracy(y_hat, y, topk=(1,5))
#             ipdb.set_trace()
            batch_correct = batch_acc[0].item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].item()*x.shape[0]/100
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss, test_acc_5

