import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import os
from utils import ensure_dir


class data(object):
    def __init__(self, config, logger):
        self.G = None
        self.node_size = 0
        self.config = config

        # param
        self.dataset = config.get("dataset")
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)

        # cache
        self.data_path = './dataset/' + self.dataset + '/'
        self.cache_file_folder = './output/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.parameters_str = str(self.dataset) + '_' + str(self.train_rate) + '_' + str(self.eval_rate)
        self.cache_file_name = os.path.join('./output/dataset_cache/',
                                            'road_rep_{}.npz'.format(self.parameters_str))

        # files
        self.logger = logger
        self.geo_file = self.config.get('info').get('geo_file')
        self.rel_file = self.config.get('info').get('rel_file')
        self.feature_dim = 0
        self._load_geo()
        self._load_rel()
        self._load_data_feature()
        self._split_train_val_test()

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        self.logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))
        self.road_info = geofile

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的矩阵，默认.rel存在的边表示为1，不存在的边表示为0

        Returns:
            scipy.sparse.coo.coo_matrix: self.adj_mx, N*N的稀疏矩阵
            nx.DiGraph: self.G, networkx 形式的图
        """
        map_info = pd.read_csv(self.data_path + self.rel_file + '.rel')
        # 使用稀疏矩阵构建邻接矩阵
        adj_row = []
        adj_col = []
        adj_data = []
        adj_set = set()
        cnt = 0
        for i in range(map_info.shape[0]):
            if map_info['origin_id'][i] in self.geo_to_ind and map_info['destination_id'][i] in self.geo_to_ind:
                f_id = self.geo_to_ind[map_info['origin_id'][i]]
                t_id = self.geo_to_ind[map_info['destination_id'][i]]
                if (f_id, t_id) not in adj_set:
                    adj_set.add((f_id, t_id))
                    adj_row.append(f_id)
                    adj_col.append(t_id)
                    adj_data.append(1.0)
                    cnt = cnt + 1
        self.adj_mx = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(self.num_nodes, self.num_nodes))
        self.G = nx.from_scipy_sparse_matrix(self.adj_mx, create_using=nx.DiGraph())
        save_path = self.cache_file_folder + "{}_adj_mx.npz".format(self.dataset)
        sp.save_npz(save_path, self.adj_mx)
        self.logger.info('Total link between geo = {}'.format(cnt))
        self.logger.info('Adj_mx is saved at {}'.format(save_path))

    def _load_data_feature(self):
        """
        获取路网原子文件中的节点属性信息，并返回
        Returns:
            node_features: N \times F 特征矩阵
            node_labels: N 的标签信息，路段分类
            lane_features: N 的车道特征
            length_features: N 的道长特征
            id_features: N 的下标
        """
        # node_features = self.road_info[['highway', 'length', 'lanes', 'tunnel', 'bridge',
        #                                 'maxspeed', 'width', 'service', 'junction', 'key']].values
        # 'tunnel', 'bridge', 'service', 'junction', 'key'是01 1+1+1+1+1
        # 'lanes', 'highway'是类别 47+6
        # 'length', 'maxspeed', 'width'是浮点 1+1+1 共61
        node_features = self.road_info[self.road_info.columns[3:]]
        node_labels = None

        # 对部分列进行归一化
        norm_dict = {'length': 1, 'maxspeed': 5, 'width': 6}

        for k, v in norm_dict.items():
            d = node_features[k]
            min_ = d.min()
            max_ = d.max()
            dnew = (d - min_) / (max_ - min_)
            node_features = node_features.drop(k, 1)
            node_features.insert(v, k, dnew)

        self.lane_features = torch.LongTensor(node_features["lanes"])
        self.length_features = torch.LongTensor(node_features["length"])
        self.id_features = torch.LongTensor(self.road_info["geo_id"])

        # 对部分列进行独热编码
        onehot_list = ['lanes', 'highway']
        for col in onehot_list:
            if col == "highway":
                self.node_labels = torch.LongTensor(node_features[col])
            dum_col = pd.get_dummies(node_features[col], col)
            node_features = node_features.drop(col, axis=1)
            node_features = pd.concat([node_features, dum_col], axis=1)

        self.node_features = node_features.values
        np.save(self.cache_file_folder + '{}_node_features.npy'.format(self.dataset), self.node_features)

    def _split_train_val_test(self):
        """
        分离训练、验证、测试数据集
        :return:
            train_mask, valid_mask, test_mask: 下标的 list，分别表示选中训练、验证、测试的节点下标
        """
        # mask 索引
        sindex = list(range(self.num_nodes))
        np.random.seed(1234)
        np.random.shuffle(sindex)

        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(self.num_nodes * test_rate)
        num_train = round(self.num_nodes * self.train_rate)
        num_val = self.num_nodes - num_test - num_train

        self.train_mask = np.array(sorted(sindex[0: num_train]))
        self.valid_mask = np.array(sorted(sindex[num_train: num_train + num_val]))
        self.test_mask = np.array(sorted(sindex[-num_test:]))

        self.logger.info("len train feature\t" + str(len(self.train_mask)))
        self.logger.info("len eval feature\t" + str(len(self.valid_mask)))
        self.logger.info("len test feature\t" + str(len(self.test_mask)))

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            batch_data: dict
        """
        # 数据归一化
        self.feature_dim = self.node_features.shape[-1]
        train_dataloader = {'mask': self.train_mask}
        eval_dataloader = {'mask': self.valid_mask}
        test_dataloader = {'mask': self.test_mask}
        return train_dataloader, eval_dataloader, test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                'node_features': self.node_features, 'node_labels': self.node_labels,
                'id_features': self.id_features, 'lane_features': self.lane_features,
                'length_features': self.length_features,
                "feature_dim": self.feature_dim}
