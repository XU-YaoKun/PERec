import random
import numpy as np

import torch


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bpr_loss(u_e, pos_e, neg_e):
    pos_score = torch.sum(u_e * pos_e, dim=1)
    neg_score = torch.sum(u_e * neg_e, dim=1)

    maxi = torch.log(torch.sigmoid(pos_score - neg_score))
    loss = torch.neg(torch.mean(maxi))

    return loss


def l2_loss(*args):
    result = 0
    for x in args:
        result += torch.sum(x ** 2) / 2

    return result


def euclidean_distance(u, v):
    return torch.sum((u - v) ** 2, dim=-1)


def get_row_index(e):
    num_row = e.size(0)
    row_index = torch.arange(
        num_row, device=e.device, dtype=torch.long
    ).unsqueeze(dim=1)

    return row_index
