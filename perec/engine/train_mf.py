import torch

from tqdm import tqdm
from time import time


def train_mf(model, data_loader, optimizer, cur_epoch):
    loss, bpr_loss, reg_loss = 0, 0, 0

    tbar = tqdm(data_loader, ascii=True)
    for data_batch in tbar:
        tbar.set_description("Epoch {}".format(cur_epoch))

        if torch.cuda.is_available():
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        optimizer.zero_grad()

        user = data_batch["user"]
        pos = data_batch["pos"]
        neg = data_batch["neg"]

        batch_bpr_loss, batch_reg_loss = model(user, pos, neg)
        batch_loss = batch_bpr_loss + batch_reg_loss
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss
        bpr_loss += batch_bpr_loss
        reg_loss += batch_reg_loss

    loss_str = "\n Train loss: [{0:.5f} = {1:.5f}(bpr) + {2:.5f}(reg)]".format(
        loss, bpr_loss, reg_loss
    )
    return loss_str
