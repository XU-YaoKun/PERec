import torch
import torch.nn as nn
import torch.functional as F

from perec.utils.torch_utils import l2_loss, euclidean_distance


class Generator(nn.Module):
    def __init__(self, n_item, n_user, embed_size, regs):
        super(Generator, self).__init__()
        self.n_item = n_item
        self.n_user = n_user
        self.regs = regs

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_user, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_item, embed_size))

        self.umlp = nn.Linear(embed_size, embed_size)
        self.imlp = nn.Linear(embed_size, embed_size)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def forward(self, user, items, ids, reward):
        u_e = self.umlp(self.user_embedding[user])
        i_e = self.imlp(self.item_embedding[items])

        u_e = u_e.unsqueeze(dim=1)
        distance = euclidean_distance(u_e, i_e) + 1e-6
        prob = F.softmax(distance, dim=-1)

        reg_loss = self.regs * l2_loss(u_e, i_e)

        batch_size = user.size(0)
        row_ids = torch.arange(
            batch_size, device=user.device, dtype=torch.long
        ).unsqueeze(dim=1)
        good_prob = prob[row_ids, ids].squeeze()

        gan_loss = -torch.mean(torch.log(good_prob) * reward)

        return gan_loss, reg_loss

    def throw(self, user, items):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]

        u_e = self.umlp(u_e)
        i_e = self.imlp(i_e)

        u_e = u_e.unsqueeze(dim=1)
        distance = euclidean_distance(u_e, i_e) + 1e-6
        prob = F.softmax(distance, dim=-1)
        sampled_id = torch.multinomial(prob, num_samples=1)

        batch_size = user.size(0)
        row_idx = torch.arange(
            batch_size, device=user.device, dtype=torch.long
        ).unsqueeze(dim=1)

        good_neg = items[row_idx, sampled_id].squeeze()

        return good_neg, sampled_id


class Discriminator(nn.Module):
    def __init__(self, n_item, n_user, embed_size, regs, margin):
        super(Discriminator, self).__init__()
        self.n_item = n_item
        self.n_user = n_user
        self.margin = margin
        self.regs = regs

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_user, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_item, embed_size))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def forward(self, user, pos, neg, **kwargs):
        negs = kwargs["negs"]

        u_e = self.user_embedding[user]
        pos_e = self.item_embedding[pos]
        neg_e = self.item_embedding[neg]
        negs_e = self.item_embedding[negs]

        reg_loss = self.regs * l2_loss(u_e, pos_e, neg_e, negs_e)

        pos_d = euclidean_distance(u_e, pos_e)
        neg_d = euclidean_distance(u_e, neg_e)
        negs_d = euclidean_distance(u_e.unsqueeze(dim=1) - negs_e)

        impostor = pos_d.unsqueeze(dim=1) - negs_d + self.margin
        rank = torch.mean(impostor, dim=1) * self.n_user

        hinge_loss = torch.sum(
            torch.log(rank + 1) * torch.clamp(self.m + pos_d - neg_d, min=0)
        )

        return hinge_loss, reg_loss

    def step(self, user, item):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[item]

        reward = (-1) * euclidean_distance(u_e, i_e)

        return reward


class NMRN:
    def __init__(self, n_users, n_items, embed_size, regs, margin):
        super(NMRN, self).__init__()

        self.name = "NMRN"
        self.n_users = n_users
        self.n_items = n_items

        self.netG = Generator(
            n_user=n_users, n_item=n_items, embed_size=embed_size, regs=regs
        )
        self.netD = Discriminator(
            n_item=n_items,
            n_user=n_users,
            embed_size=embed_size,
            regs=regs,
            margin=margin,
        )

        self.user_embedding = self.netD.user_embedding
        self.item_embedding = self.netD.item_embedding

    def cuda(self):
        self.netD = self.netD.cuda()
        self.netG = self.netG.cuda()

        return self
