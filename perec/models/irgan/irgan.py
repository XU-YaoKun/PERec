import torch
import torch.nn as nn
import torch.nn.functional as F

from perec.utils.torch_utils import l2_loss


class Generator(nn.Module):
    def __init__(self, n_users, n_items, embed_size, regs):
        super(Generator, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.regs = regs
        self.delta = 0.1

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_users, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_items, embed_size))
        self.bias = nn.Parameter(torch.FloatTensor(n_items))

        self._init_weight()

    def _init_weight(self):
        delta = self.delta
        nn.init.uniform_(self.user_embedding, a=-delta, b=delta)
        nn.init.uniform_(self.item_embedding, a=-delta, b=delta)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, user, items, reward):
        u_e, i_e, b = self._get_embedding(user, items)
        u_e = u_e.unsqueeze(dim=1)

        logits = torch.sum(u_e * i_e, dim=2) + b
        log_probs = F.log_softmax(logits, dim=1)

        batch_size = user.size(0)
        row_id = torch.arange(batch_size, device=user.device).unsqueeze(dim=1)
        max_idx = torch.argmax(log_probs, dim=1)

        max_log_probs = log_probs[row_id, max_idx]
        max_reward = reward[row_id, max_idx]
        gan_loss = -torch.mean(max_log_probs * max_reward)

        regularizer = l2_loss(u_e, i_e, b)
        reg_loss = self.regs * regularizer

        return gan_loss, reg_loss

    def throw(self, user, items):
        u_e, i_e, b = self._get_embedding(user, items)
        u_e = u_e.unsqueeze(dim=1)

        ranking = torch.sum(u_e * i_e, dim=2) + b
        p = F.softmax(ranking, dim=1)
        indices = torch.multinomial(p, num_samples=1)

        batch_size = user.size(0)
        row_id = torch.arange(batch_size, device=user.device).unsqueeze(dim=1)

        good_neg = items[row_id, indices].squeeze()

        return good_neg

    def _get_embedding(self, user, items):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]
        b = self.bias[items]

        return u_e, i_e, b


class Discriminator(nn.Module):
    def __init__(self, n_users, n_items, embed_size, regs):
        super(Discriminator, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.regs = regs
        self.delta = 0.1

        self.user_embedding = nn.Parameter(torch.FloatTensor(n_users, embed_size))
        self.item_embedding = nn.Parameter(torch.FloatTensor(n_items, embed_size))
        self.bias = nn.Parameter(torch.FloatTensor(n_items))

        self._init_weight()

    def _init_weight(self):
        delta = self.delta

        nn.init.uniform_(self.user_embedding, a=-delta, b=delta)
        nn.init.uniform_(self.item_embedding, a=-delta, b=delta)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, user, pos, neg, **kwargs):
        batch_size = user.size(0)
        pos_label = torch.ones(batch_size, device=user.device)
        neg_label = torch.zeros(batch_size, device=user.device)

        pos_loss, pos_reg = self.get_cls_loss(user, pos, pos_label)
        neg_loss, neg_reg = self.get_cls_loss(user, neg, neg_label)

        cls_loss = pos_loss + neg_loss
        reg_loss = pos_reg + neg_reg

        return cls_loss, reg_loss

    def get_cls_loss(self, user, item, label):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[item]
        b = self.bias[item]

        logits = torch.sum(u_e * i_e, dim=1) + b
        cls_loss = F.binary_cross_entropy_with_logits(logits, label)

        reg_loss = self.regs * l2_loss(u_e, i_e)

        return cls_loss, reg_loss

    def step(self, user, items):
        u_e = self.user_embedding[user]
        i_e = self.item_embedding[items]
        b = self.bias[items]

        u_e = u_e.unsqueeze(dim=1)
        logits = torch.sum(u_e * i_e, dim=2) + b
        reward = 2 * (torch.sigmoid(logits) - 0.5)

        return reward


class IRGAN:
    def __init__(self, n_users, n_items, embed_size, regsD, regsG):
        super(IRGAN, self).__init__()

        self.name = "IRGAN"

        self.netG = Generator(
            n_users=n_users, n_items=n_items, embed_size=embed_size, regs=regsG
        )
        self.netD = Discriminator(
            n_users=n_users, n_items=n_items, embed_size=embed_size, regs=regsD
        )

        self.user_embedding = self.netD.user_embedding
        self.item_embedding = self.netD.item_embedding
        self.bias = self.netD.bias
        self.n_users = self.netD.n_users
        self.n_items = self.netD.n_items

    def cuda(self):
        self.netD = self.netD.cuda()
        self.netG = self.netG.cuda()

        return self
