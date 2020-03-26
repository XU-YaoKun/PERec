from tqdm import tqdm
import numpy as np

import torch

from perec.utils.torch_utils import euclidean_distance


def cal_ndcg(topk, test_set, num_pos, k):
    n = min(num_pos, k)
    nrange = np.arange(n) + 2
    idcg = np.sum(1 / np.log2(nrange))

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:
            dcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg

    return ndcg


def get_score_v2(model, train_user_dict, s, t):
    u_e = model.user_embedding[s:t, :]
    i_e = model.item_embedding

    if model.name == "NMRN":
        u_e = u_e.unsqueeze(dim=1)
        score_matrix = (-1) * euclidean_distance(u_e, i_e)
    else:
        score_matrix = torch.matmul(u_e, i_e.t())
        if hasattr(model, "bias"):
            score_matrix = score_matrix + model.bias

    for u in range(s, t):
        pos = train_user_dict[u]
        score_matrix[u - s][pos] = -1e5

    return score_matrix


def test_model_v2(model, ks, user_dict, n_batchs=32):
    train_user_dict, test_user_dict = (
        user_dict.train_user_dict,
        user_dict.test_user_dict,
    )
    n_test_user = len(test_user_dict)

    n_k = len(ks)
    result = {
        "PRECISION": np.zeros(n_k),
        "RECALL": np.zeros(n_k),
        "NDCG": np.zeros(n_k),
        "HIT_RATIO": np.zeros(n_k),
    }

    n_users = model.n_users
    batch_size = n_users // n_batchs
    for batch_id in tqdm(range(n_batchs), ascii=True, desc="Evaluate"):
        s = batch_size * batch_id
        t = batch_size * (batch_id + 1)
        if t > n_users:
            t = n_users
        if s == t:
            break

        score_matrix = get_score_v2(model, train_user_dict, s, t)

        for i, k in enumerate(ks):
            precision, recall, ndcg, hr = 0, 0, 0, 0
            _, topk_index = torch.topk(score_matrix, k)
            topk_index = topk_index.cpu().numpy()

            for u in range(s, t):
                gt_pos = test_user_dict[u]
                topk = topk_index[u - s]
                num_pos = len(gt_pos)

                topk_set = set(topk)
                test_set = set(gt_pos)
                num_hit = len(topk_set & test_set)

                precision += num_hit / k
                recall += num_hit / num_pos
                hr += 1 if num_hit > 0 else 0

                ndcg += cal_ndcg(topk, test_set, num_pos, k)

            result["PRECISION"][i] += precision / n_test_user
            result["RECALL"][i] += recall / n_test_user
            result["NDCG"][i] += ndcg / n_test_user
            result["HIT_RATIO"][i] += hr / n_test_user

    return result
