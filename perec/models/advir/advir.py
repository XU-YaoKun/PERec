import torch
import torch.nn as nn

from perec.utils.torch_utils import l2_loss


class Generator(nn.Module):
    def __init__(self, n_users, n_items, embed_size, regs):
        super(Generator, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.regs = regs

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_users, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_items, embed_size))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def forward(self, user, items, ids, reward):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]

        u_e = u_e.unsqueeze(dim=1)
        logits = torch.sum(u_e * i_e, dim=2)
        probs = torch.softmax(logits, dim=1)

        reg_loss = self.regs * l2_loss(u_e, i_e)

        batch_size = user.size(0)
        row_ids = torch.arange(
            batch_size, device=user.device, dtype=torch.long
        ).unsqueeze(dim=1)
        good_prob = probs[row_ids, ids].squeeze()

        gan_loss = -torch.mean(torch.log(good_prob) * reward)

        return gan_loss, reg_loss

    def throw(self, user, items):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]

        u_e = u_e.unsqueeze(dim=1)
        scores = torch.sum(u_e * i_e, dim=2)
        probs = torch.softmax(scores, dim=-1)

        sampled_id = torch.multinomial(probs, num_samples=1)

        batch_size = u_e.size(0)
        row_idx = torch.arange(
            batch_size, device=u_e.device, dtype=torch.long
        ).unsqueeze(dim=1)

        sampled_neg = items[row_idx, sampled_id].squeeze()

        return sampled_neg, sampled_id


class Discriminator(nn.Module):
    def __init__(self, n_users, n_items, embed_size, regs):
        super(Discriminator, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.regs = regs

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_users, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_items, embed_size))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def forward(self, user, pos, neg, **kwargs):
        u_e = self.user_embedding[user]
        pos_e = self.item_embedding[pos]
        neg_e = self.item_embedding[neg]

        reg_loss = self.regs * l2_loss(u_e, pos_e, neg_e)

        pos_score = torch.sum(pos_e * u_e, dim=1)
        neg_score = torch.sum(neg_e * u_e, dim=1)

        maxi = torch.log(torch.sigmoid(pos_score - neg_score))
        bpr_loss = torch.neg(torch.mean(maxi))

        return bpr_loss, reg_loss

    def step(self, user, item):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[item]

        reward = torch.sum(u_e * i_e, dim=1)

        return reward


class AdvIR:
    def __init__(self, n_users, n_items, embed_size, regsD, regsG):
        super(AdvIR, self).__init__()

        self.name = "AdvIR"
        self.n_users = n_users
        self.n_items = n_items

        self.netG = Generator(
            n_users=n_users, n_items=n_items, embed_size=embed_size, regs=regsG
        )
        self.netD = Discriminator(
            n_users=n_users, n_items=n_items, embed_size=embed_size, regs=regsD
        )

        self.user_embedding = self.netD.user_embedding
        self.item_embedding = self.netD.item_embedding

    def cuda(self):
        self.netD = self.netD.cuda()
        self.netG = self.netG.cuda()

        return self
