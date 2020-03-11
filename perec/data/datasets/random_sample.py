import numpy as np

from perec.data.datasets.base import BaseDataset
from perec.utils.misc import sample_one, list_range


class RandomSample(BaseDataset):
    """Random Sample

    Sample negative item randomly

    """

    def __init__(self, cfg_data):
        super(RandomSample, self).__init__(cfg_data)

    def _get_one_neg(self, u, neg_list):
        while True:
            if self.k_negs == 1:
                all_id = list_range(self.n_items)
                neg = sample_one(all_id)
            else:
                all_id = np.arange(self.n_items)
                neg = np.random.choice(all_id)

            if neg not in self.train_dict[u] and neg not in neg_list:
                break

        return neg
