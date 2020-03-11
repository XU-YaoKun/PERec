import random
import numpy as np


def sample_one(l):
    """random sample element from l, using random

    this function is about twice slower than sample_one_v2
    but it's thread-safe
    """
    ele = random.choice(l)

    return ele


def sample_one_v2(l):
    """random sample element from l, using np.random

    when using multi-threa, do not use this function
    it might not generate "random" element very well
    """
    sample_id = np.random.randint(low=0, high=len(l), size=0)[0]

    return l[sample_id]


def key_value(dic):
    """covert dict to two list, keys and values

    in some cases, list is much more efficient than generator
    """
    keys = list(dic.keys())
    values = list(dic.values())

    return keys, values


def list_range(n):
    """list of range

    example: 4
    output: [0, 1, 2, 3]
    """
    return list(range(n))


def print_dict(dic):
    """print dictionary using specified format

    example: {"a": 1, "b": 2}
    output:
            "a": 1
            "b": 2
    """
    print("\n".join("{:10s}: {}".format(key, values) for key, values in dic.items()))
