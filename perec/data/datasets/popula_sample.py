import numpy as np
import random

from perec.data.datasets.base import BaseDataset
from perec.data.datasets.utils import cal_popularity
from perec.utils.misc import (
    sample_one,
    sample_one_v2,
    list_range,
)


class PopulaSample(BaseDataset):
    """Sample based on popularity

    Sample negative items based on frequency in dataset

    """

    def __init__(self, cfg_data):
        super(PopulaSample, self).__init__(cfg_data)

        item_list = [item for sublist in self.train_items for item in sublist]
        self.popula = cal_popularity(self.n_items, item_list)

        self.all_id = list_range(self.n_items)

    def _get_one_neg(self, u, pos, neg_list):
        while True:
            neg = np.random.choice(self.all_id, p=self.popula)
            if neg not in self.train_dict[u] and neg not in neg_list:
                break

        return neg
