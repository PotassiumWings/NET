import os
import importlib


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_downstream(config, data_feature, vectors):
    try:
        return getattr(importlib.import_module('downstream'), config.get("downstream"))\
            (config, data_feature, vectors)
    except AttributeError:
        raise AttributeError("method is not found")


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

