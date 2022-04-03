import random
import time
from argparse import ArgumentParser

import numpy
from sklearn.linear_model import LogisticRegression

from Graph import Graph
from downstream import classify
from model import node2vec
from config import Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output',
                        help='Output representation file')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--method', required=True, choices=[
        'node2vec',
        'deepWalk'
    ], help='The learning method')
    args = parser.parse_args()
    return Config(vars(args))


def main(config):
    start_time = time.time()
    g = Graph()
    g.read_edgelist(config.get("input"))

    m = node2vec.node2vec(g, config)

    finish_time = time.time()
    print(finish_time - start_time)
    print("Saving embeddings...")
    m.save_embeddings(config.get("output"))

    vectors = m.vectors
    X, Y = classify.read_node_label(config.get("label_file"))
    print("Training classifier using {:.2f}% nodes...".format(
        0.5 * 100))
    clf = classify.classify(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, 0.5, seed=0)


if __name__ == '__main__':
    random.seed(32)
    numpy.random.seed(32)
    main(parse_args())

