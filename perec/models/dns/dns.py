import torch
import torch.nn as nn

from perec.utils.torch_utils import l2_loss, bpr_loss


class DNS(nn.Module):
    def __init__(self, n_users, n_items, embed_size, regs):
        super(DNS, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.regs = regs

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_users, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_items, embed_size))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def forward(self, user, pos, negs):
        u_e = self.user_embedding[user]
        pos_e = self.item_embedding[pos]
        negs_e = self.item_embedding[negs]

        with torch.no_grad():
            ranking = self.rank(u_e, negs_e)

        indices = torch.argmax(ranking, dim=1).unsqueeze(dim=1)

        batch_size = negs.size(0)
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(dim=1)

        good_neg = negs[row_id, indices].squeeze()
        neg_e = self.item_embedding[good_neg]

        loss = bpr_loss(u_e, pos_e, neg_e)

        regularizer = l2_loss(u_e, pos_e, neg_e)
        reg_loss = self.regs * regularizer

        return loss, reg_loss

    @staticmethod
    def rank(u_e, negs_e):
        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * negs_e, dim=2)
        ranking = ranking.squeeze()

        return ranking
