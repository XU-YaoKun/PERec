from collections import defaultdict
import numpy as np

from perec.data.datasets.base import BaseDataset


class RandomWalkSample(BaseDataset):
    """Random Walk Sample

    Sample negative item using knowledge graph.
    To choose negative item, start from positive item,
    use random walk, and end in negative item
    """

    def __init__(self, cfg_data):
        super(RandomWalkSample, self).__init__(cfg_data)

        self.num_step = cfg_data.NUM_STEP

        kg_path = cfg_data.ROOT_DIR + "kg_final.txt"
        self.path_dict = self._load_kg(kg_path)

    def _load_kg(self, file_name):
        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        path_dict = defaultdict(list)
        for head, relation, tail in kg_np:
            path_dict[head].append(tail)
            path_dict[tail].append(head)

        return path_dict

    def step(self, u, pos, neg_list):
        neighbor = self.path_dict[pos]
        train_set = self.train_dict[u]

        while True:
            if len(neighbor) == 0:
                all_id = np.arange(self.n_items)
                neg = np.random.choice(all_id)
            else:
                neg = np.random.choice(neighbor)

            if neg not in train_set + neg_list:
                break

        return neg

    def _get_one_neg(self, u, pos, neg_list):
        neg = pos
        for _ in range(self.num_step):
            neg = self.step(u, neg, neg_list)

        return neg
