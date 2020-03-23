import random
import numpy as np
from time import time

import torch
from torch.utils.data import Dataset

from perec.data.datasets.utils import load_rating
from perec.utils.misc import sample_one, key_value


class BaseDataset(Dataset):
    """Base Dataset

    Attributes:
        data_path (str): the root directory of data.
        k_negs (int): number of negative items for one positive items.
        train_dict (dict): {user_id: positive_items} in training set.
        test_dict (dict): {user_id: positive_items} in testing set.
        
    """

    def __init__(self, cfg_data):
        self.data_path = cfg_data.ROOT_DIR
        self.k_negs = cfg_data.K_NEGS

        self.n_train, self.train_dict = load_rating(self.data_path + "train.dat")
        self.n_test, self.test_dict = load_rating(self.data_path + "test.dat")

        self.train_users, self.train_items = key_value(self.train_dict)
        self.test_users, self.test_items = key_value(self.test_dict)

        self._statistic()
        self._print()

    @staticmethod
    def max_list(l):
        return max(map(lambda x: max(x), l))

    def _statistic(self):
        self.n_users = max(max(self.train_users), max(self.test_users)) + 1
        self.n_items = (
            max(self.max_list(self.train_items), self.max_list(self.test_items),) + 1
        )

    def _print(self):
        print("-" * 50)
        print("- num_train - {}".format(self.n_train))
        print("- num_test  - {}".format(self.n_test))
        print("- num_users - {}".format(self.n_users))
        print("- num_items - {}".format(self.n_items))
        print("-" * 50)

    def __len__(self):
        return self.n_train

    def __getitem__(self, item):
        out_dict = {}

        u = sample_one(self.train_users)
        out_dict["user"] = u

        pos_items = self.train_dict[u]
        pos_n = len(pos_items)
        pos_id = np.random.randint(low=0, high=pos_n, size=1)[0]
        pos = pos_items[pos_id]
        out_dict["pos"] = pos

        neg_list = []
        for _ in range(self.k_negs):
            neg_id = self._get_one_neg(u, pos, neg_list)
            neg_list.append(neg_id)
        out_dict["neg"] = torch.tensor(neg_list).squeeze()

        return out_dict

    def _get_one_neg(self, u, pos, neg_list):
        raise NotImplementedError("subclass must override _get_one_neg()!")
