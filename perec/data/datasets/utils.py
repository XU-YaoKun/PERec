import numpy as np


def cal_popularity(n_items, item_list):
    popularity = np.zeros(n_items)

    for item in item_list:
        popularity[item] += 1

    popularity /= np.sum(popularity)
    return popularity


def load_rating(file_name):
    ui_dicts = dict()
    n = 0

    lines = open(file_name, "r").readlines()
    for l in lines:
        v = [int(i) for i in l.strip().split(" ")]

        u_id, pos_ids = v[0], v[1:]
        pos_ids = list(set(pos_ids))

        n = n + len(pos_ids)

        if len(pos_ids) > 0:
            ui_dicts[u_id] = pos_ids

    return n, ui_dicts
