import torch
import torch.nn as nn
import torch.nn.functional as F

from perec.utils.torch_utils import l2_loss, get_row_index


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

    def forward(self, user, items, reward):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]

        u_e = u_e.unsqueeze(dim=1)
        logits = torch.sum(u_e * i_e, dim=2)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        sampled_id = torch.multinomial(probs, num_samples=1)
        row_idx = get_row_index(u_e)

        sampled_probs = log_probs[row_idx, sampled_id]
        sampled_reward = reward[row_idx, sampled_id]
        gan_loss = -torch.mean(sampled_probs * sampled_reward)

        reg_loss = self.regs * l2_loss(u_e, i_e)

        return gan_loss, reg_loss

    def throw(self, user, items):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]

        u_e = u_e.unsqueeze(dim=1)
        scores = torch.sum(u_e * i_e, dim=2)
        probs = torch.softmax(scores, dim=-1)

        sampled_id = torch.multinomial(probs, num_samples=1)
        row_idx = get_row_index(u_e)

        sampled_neg = items[row_idx, sampled_id].squeeze()

        return sampled_neg


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

    def step(self, user, items):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]

        u_e = u_e.unsqueeze(dim=1)
        reward = torch.sum(u_e * i_e, dim=2)

        return reward


class AdvIR:
    def __init__(self, n_users, n_items, embed_size, regsD, regsG):
        super(AdvIR, self).__init__()

        self.name = "AdvIR"
        self.n_users = n_users
        self.n_items = n_items

        self.netG = Generator(
            n_users=n_users, n_items=n_items, embed_size=embed_size, regs=regsG,
        )
        self.netD = Discriminator(
            n_users=n_users, n_items=n_items, embed_size=embed_size, regs=regsD,
        )

        self.user_embedding = self.netD.user_embedding
        self.item_embedding = self.netD.item_embedding

    def cuda(self):
        self.netD = self.netD.cuda()
        self.netG = self.netG.cuda()

        return self
