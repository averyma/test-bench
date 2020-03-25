import torch
import torch.nn as nn
from torch.autograd import grad

from tqdm import trange
import numpy as np

from src.attacks import pgd_rand
from src.soar import soar
from src.context import ctx_noparamgrad_and_eval, ctx_eval
from src.utils_general import ep2itr

def data_init(init, X, y, model):
    if init == "rand":
        delta = torch.empty_like(X.detach(), requires_grad=False).uniform_(-8./255.,8./255.)
        delta.data = (X.detach() + delta.detach()).clamp(min = 0, max = 1.0) - X.detach()
    elif init == "fgsm":
        with ctx_noparamgrad_and_eval(model):
            param = {"ord":np.inf, "epsilon": 2./255.}
            delta = fgsm(**param).generate(model,X,y)
    elif init == "pgd1":
        with ctx_noparamgrad_and_eval(model):
            param = {"ord":np.inf, "epsilon": 8./255., "alpha":2./255., "num_iter": 1, "restart": 1}
            delta = pgd_rand(**param).generate(model,X,y)
    elif init == "none":
        delta = torch.zeros_like(X.detach(), requires_grad=False)

    return delta

def train_standard(logger, epoch, loader, model, opt, device):
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = (yp.argmax(dim=1) == y).sum().item()
            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            logger.add_scalar("train_acc_itr", batch_acc, curr_itr)
            logger.add_scalar("train_loss_itr", loss, curr_itr)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)

    return acc, total_loss

def train_adv(logger, epoch, loader, pgd_steps, model, opt, device):
    total_loss_adv = 0.
    total_correct_adv = 0.

    attack = pgd_rand
    curr_itr = ep2itr(epoch, loader)
    param = {'ord': np.inf,
             'epsilon': 8./255.,
             'alpha': 2./255.,
             'num_iter': pgd_steps,
             'restarts': 1}

    with trange(len(loader)) as t:
        for X,y in loader:
            model.train()
            X,y = X.to(device), y.to(device)

            with ctx_noparamgrad_and_eval(model):
                delta = attack(**param).generate(model, X, y)
            
            yp_adv = model(X+delta)
            loss_adv = nn.CrossEntropyLoss()(yp_adv, y)
                
            opt.zero_grad()
            loss_adv.backward()
            opt.step()
    
            batch_correct_adv = (yp_adv.argmax(dim = 1) == y).sum().item()
            total_correct_adv += batch_correct_adv

            batch_acc_adv = batch_correct_adv / X.shape[0]
            total_loss_adv += loss_adv.item() * X.shape[0]

            t.set_postfix(loss_adv = loss_adv.item(),
                          acc_adv = '{0:.2f}%'.format(batch_acc_adv*100))
            t.update()
            curr_itr += 1
            logger.add_scalar("train_acc_itr", batch_acc_adv, curr_itr)
            logger.add_scalar("train_loss_itr", loss_adv, curr_itr)

    acc_adv = total_correct_adv / len(loader.dataset) * 100
    total_loss_adv = total_loss_adv / len(loader.dataset)

    return acc_adv, total_loss_adv

def train_soar(logger, epoch, loader, model, soar_param, opt, device):
    init = soar_param["init"]
    eps = soar_param["eps"]
    norm_clip = soar_param["norm_clip"]
    grad_clip = soar_param["grad_clip"]
    lambbda = soar_param["lambbda"]
    step_size = soar_param["step_size"]

    len_data = len(loader.dataset)
    total_loss, total_correct, total_reg= 0.,0.,0.
    
    curr_itr = ep2itr(epoch, loader)
    with trange(len(loader)) as t:
        for X,y in loader:
            model.train()
            X,y = X.to(device), y.to(device)
            
            delta = data_init(init, X, y, model)
            X_delta = X.detach() + delta.detach()
            X_delta.requires_grad = True

            yp_delta = model(X_delta)
            loss_delta = nn.CrossEntropyLoss()(yp_delta, y)
            _dim = X.shape[1]*X.shape[2]*X.shape[3]
            
            reg = soar(X_delta, y, loss_delta, norm_clip, step_size, model, device)
            reg_delta = lambbda * 0.5 * (eps**2 *  _dim + 1) * reg
            
            opt.zero_grad()
            reg_delta.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            loss_delta.backward()
            opt.step()

            batch_correct = (yp_delta.argmax(dim = 1) == y).sum().item()
            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss_delta.item() * X.shape[0]
            total_reg += reg.detach().item() * X.shape[0]
            
            t.set_postfix(loss = loss_delta.item(),
                          reg = reg.item(),
                          acc = "{0:.2f}%".format(batch_acc*100))
            t.update()
            
            curr_itr += 1
            logger.add_scalar("train_acc_itr", batch_acc, curr_itr)
            logger.add_scalar("train_loss_itr", loss_delta.item(), curr_itr)
            logger.add_scalar("train_reg_itr", reg.item(), curr_itr)

        acc = total_correct / len_data * 100
        total_loss = total_loss / len_data
        total_reg = total_reg / len_data
        
    return acc, total_loss, total_reg

def train_exp(logger, epoch, loader, pgd_steps, model, opt, device):
    total_loss_adv = 0.
    total_correct_adv = 0.

    attack = pgd_rand
    curr_itr = ep2itr(epoch, loader)
    param = {'ord': np.inf,
             'epsilon': 8./255.,
             'alpha': 2./255.,
             'num_iter': pgd_steps,
             'restarts': 1}

    with trange(len(loader)) as t:
        for X,y in loader:
            model.train()
            X,y = X.to(device), y.to(device)

            with ctx_noparamgrad_and_eval(model):
                delta = attack(**param).generate(model, X, y)
            
            yp_adv = model(X+delta)
            loss_adv = nn.CrossEntropyLoss()(yp_adv, y)
                
            opt.zero_grad()
            loss_adv.backward()
            opt.step()
    
            batch_correct_adv = (yp_adv.argmax(dim = 1) == y).sum().item()
            total_correct_adv += batch_correct_adv

            batch_acc_adv = batch_correct_adv / X.shape[0]
            total_loss_adv += loss_adv.item() * X.shape[0]

            t.set_postfix(loss_adv = loss_adv.item(),
                          acc_adv = '{0:.2f}%'.format(batch_acc_adv*100))
            t.update()
            curr_itr += 1
            logger.add_scalar("train_acc_itr", batch_acc_adv, curr_itr)
            logger.add_scalar("train_loss_itr", loss_adv, curr_itr)

    acc_adv = total_correct_adv / len(loader.dataset) * 100
    total_loss_adv = total_loss_adv / len(loader.dataset)

    return acc_adv, total_loss_adv

