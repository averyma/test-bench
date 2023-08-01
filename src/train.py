import torch.nn as nn

from tqdm import trange

AVOID_ZERO_DIV = 1e-6

def train_standard(loader, model, opt, device, epoch=1, lr_scheduler=None):
    total_loss, total_correct = 0., 0.
    # curr_itr = ep2itr(epoch, loader)
    total_steps = len(loader)
    steps = 0
    with trange(len(loader)) as t:
        for X, y in loader:
            steps+=1
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            batch_correct = (yp.argmax(dim=1) == y).sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100),
                          lr=opt.param_groups[0]['lr'])
            t.update()

            if lr_scheduler is not None:
                lr_scheduler.step(epoch-1+float(steps)/total_steps)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss
