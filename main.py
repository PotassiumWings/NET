import logging
import random
from argparse import ArgumentParser

import numpy

from data import data
from downstream import classify
from model import node2vec
from config import Config
from logging import getLogger
from utils import get_downstream, load_embeddings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset file',
                        choices=[
                            'wiki', 'blogCatalog', 'bj_roadmap_edge'
                        ])
    parser.add_argument('--config_file', required=False, help='Config file', default=None)
    parser.add_argument('--method', required=True, help='The learning method',
                        choices=[
                            'node2vec',
                            'deepWalk'
                        ])
    parser.add_argument('--downstream', required=True, help='The downstream task',
                        choices=[
                            'classify',
                            'rnn'
                        ])
    parser.add_argument('--exp_id', required=False, help='Experiment ID', default=None)
    parser.add_argument('--cached_embedding', required=False, help='Cached Embedding', default=None)
    args = parser.parse_args()
    return Config(vars(args))


def run_model(config):
    logging.basicConfig(level=logging.INFO)
    logger = getLogger()
    logger.info("Begin pipeline, method={}, dataset={}, downstream={}, exp_id={}"
                .format(config.get("method"), config.get("dataset"),
                        config.get("downstream"), config.get("exp_id")))
    logger.info(config.config)
    logger.info("Start reading dataset...")
    dataset = data(config)
    logger.info("Dataset read.")

    cached_embedding = config.get("cached_embedding", None)
    if cached_embedding is not None:
        vectors = load_embeddings(cached_embedding)
    else:
        logger.info("Start building method...")
        m = node2vec.node2vec(dataset.G, config)
        logger.info("Model built.")

        logger.info("Saving embeddings...")
        m.save_embeddings()
        logger.info("Embeddings saved.")
        vectors = m.vectors

    logger.info("Getting data...")
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    logger.info("Finished getting data.")

    downstream = get_downstream(config, data_feature, vectors)

    logger.info("Start training downstream task...")
    downstream.train(train_data, valid_data)
    logger.info("Start Evaluating...")
    downstream.evaluate(test_data)


if __name__ == '__main__':
    random.seed(32)
    numpy.random.seed(32)
    run_model(parse_args())
