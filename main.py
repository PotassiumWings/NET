import logging
import random
from argparse import ArgumentParser
import sys

import numpy

import os
from data import data
from downstream import classify
from model import node2vec
from config import Config
from logging import getLogger
from utils import get_downstream, load_embeddings, ensure_dir, get_local_time, get_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset file',
                        choices=[
                            'wiki', 'blogCatalog', 'bj_roadmap_edge'
                        ])
    parser.add_argument('--config_file', required=False, help='Config file', default=None)
    parser.add_argument('--output_dim', required=False, help='Output dim', default=None)
    parser.add_argument('--method', required=True, help='The learning method',
                        choices=[
                            'node2vec',
                            'deepWalk',
                            'HRNR'
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
    logger = getLogger()
    logger.setLevel(logging.INFO)

    log_dir = './output/log'
    ensure_dir(log_dir)
    log_filename = '{}-{}-{}-{}.log'.format(config['exp_id'],
                                            config['model'], config['dataset'], get_local_time())
    log_file_name = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Begin pipeline, method={}, dataset={}, downstream={}, exp_id={}"
                .format(config["method"], config["dataset"],
                        config["downstream"], config["exp_id"]))
    logger.info(config.config)
    logger.info("Start reading dataset...")
    dataset = data(config, logger)
    logger.info("Dataset read.")

    logger.info("Getting data...")
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    logger.info("Finished getting data.")

    cached_embedding = config["cached_embedding"]
    if cached_embedding is not None:
        vectors = load_embeddings(cached_embedding)
    else:
        logger.info("Start building method...")
        m = get_model(config, dataset, logger)
        logger.info("Model built.")

        logger.info("Saving embeddings...")
        m.save_embeddings()
        logger.info("Embeddings saved.")
        vectors = m.vectors

    downstream = get_downstream(config, data_feature, vectors, logger)

    logger.info("Start training downstream task...")
    downstream.train(train_data, valid_data)
    logger.info("Start Evaluating...")
    downstream.evaluate(test_data)


if __name__ == '__main__':
    random.seed(32)
    numpy.random.seed(32)
    run_model(parse_args())
