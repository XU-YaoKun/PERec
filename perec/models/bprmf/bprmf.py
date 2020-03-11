import torch
import torch.nn as nn

from perec.utils.torch_utils import l2_loss, bpr_loss


class BPRMF(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_size,
                 regs):
        super(BPRMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.regs = regs

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_users, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_items, embed_size))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def forward(self, user, pos, neg):
        u_e = self.user_embedding[user]
        pos_e = self.item_embedding[pos]
        neg_e = self.item_embedding[neg]

        loss = bpr_loss(u_e, pos_e, neg_e)

        regularizer = l2_loss(u_e, pos_e, neg_e)
        reg_loss = self.regs * regularizer

        return loss, reg_loss
