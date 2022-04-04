import os
import importlib
import datetime


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_downstream(config, data_feature, vectors, logger):
    return getattr(importlib.import_module('downstream'), config["downstream"]) \
        (config, data_feature, vectors, logger)


def get_model(config, dataset, logger):
    return getattr(importlib.import_module('model'), config["method"]) \
        (config, dataset, logger)


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size + 1
        vectors[int(vec[0])] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur
