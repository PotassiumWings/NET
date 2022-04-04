import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.cluster import SpectralClustering
from torch.nn import Module
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score
import os
from model.AbstractModel import AbstractModel
from utils import ensure_dir


class HRNR(AbstractModel, nn.Module):
    def __init__(self, config, dataset, logger, kwargs=None):
        super().__init__(config, dataset, logger, kwargs)
        nn.Module.__init__(self)
        self.device = config.get("device", torch.device("cpu"))
        self.special_spmm = SpecialSpmm()

        # get transfer matrix
        self.preprocess = HRNRPreprocess(config, dataset, logger)

        self.struct_assign = self.preprocess.struct_assign
        self.fnc_assign = self.preprocess.fnc_assign
        self.adj = get_sparse_adj(self.preprocess.adj_matrix.toarray(), self.device)

        edge = self.adj.indices()
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.device)
        struct_inter = self.special_spmm(edge, edge_e, torch.Size([self.adj.shape[0], self.adj.shape[1]]),
                                         self.struct_assign)  # N*N   N*C
        struct_adj = torch.mm(self.struct_assign.t(), struct_inter)  # get struct_adj

        hparams = dict_to_object(config.config)
        self.graph_enc = GraphEncoderTL(hparams, self.struct_assign, self.fnc_assign, struct_adj, self.device)

        self.linear = torch.nn.Linear(hparams.hidden_dims * 2, hparams.label_num)

        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 100)
        self.node_emb, self.init_emb = None, None

        # start training
        self.train_embedding()
        self.vectors = {}

        data_feature = self.dataset.get_data_feature()
        embeddings = self.forward(data_feature).detach()
        for i, emb in enumerate(embeddings):
            self.vectors[i] = emb

    def train_embedding(self):
        self.logger.info("Starting training...")
        hparams = dict_to_object(self.config.config)
        ce_criterion = torch.nn.CrossEntropyLoss()
        max_f1 = 0
        count = 0
        model_optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lp_learning_rate)
        train_dataloader, eval_dataloader, test_dataloader = self.dataset.get_data()
        data_feature = self.dataset.get_data_feature()

        eval_mask = eval_dataloader['mask']
        eval_iter = 0

        for i in range(hparams.label_epoch):
            self.logger.info("epoch " + str(i) + ", processed " + str(count))
            mask = train_dataloader['mask']
            for step in range(0, len(train_dataloader['mask']) - 128, 128):
                model_optimizer.zero_grad()
                train_set = mask[step: step + 128]
                train_label = torch.LongTensor(data_feature['node_labels'][train_set]).to(self.device)
                pred = self.predict(data_feature, train_set)
                loss = ce_criterion(pred, train_label)
                loss = loss.requires_grad_()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.lp_clip)
                model_optimizer.step()
                if count % 20 == 0:
                    eval_data = []
                    for j in range(0, 128):
                        eval_data.append(eval_mask[(eval_iter + j) % len(eval_mask)])
                    eval_iter += 128

                    precision, recall, f1 = self.test_label_pred(data_feature, eval_data, self.device)
                    if f1 > max_f1:
                        max_f1 = f1
                    self.logger.info("max_f1: " + str(max_f1))
                    self.logger.info("step " + str(count))
                    self.logger.info(loss.item())
                count += 1

    def test_label_pred(self, data_feature, mask, device):
        # TODO 这里的 prf 都是 0，不知道是什么情况
        right = 0
        sum_num = 0
        pred = self.predict(data_feature, mask).detach()

        test_label = torch.LongTensor(data_feature['node_labels'][mask]).to(self.device)

        pred_loc = torch.argmax(pred, 1).tolist()
        right_pos = 0
        right_neg = 0
        wrong_pos = 0
        wrong_neg = 0
        for item1, item2 in zip(pred_loc, test_label):
            if item1 == item2:
                right += 1
                if item2 == 1:
                    right_pos += 1
                else:
                    right_neg += 1
            else:
                if item2 == 1:
                    wrong_pos += 1
                else:
                    wrong_neg += 1
            sum_num += 1
        recall_sum = right_pos + wrong_pos
        precision_sum = wrong_neg + right_pos
        if recall_sum == 0:
            recall_sum += 1
        if precision_sum == 0:
            precision_sum += 1
        recall = float(right_pos) / recall_sum
        precision = float(right_pos) / precision_sum
        if recall == 0 or precision == 0:
            self.logger.info("p/r/f:0/0/0")
            return 0.0, 0.0, 0.0
        f1 = 2 * recall * precision / (precision + recall)
        self.logger.info("label prediction @acc @p/r/f: " + str(float(right) / sum_num) + " " + str(precision) +
                          " " + str(recall) + " " + str(f1))
        return precision, recall, f1

    def forward(self, data_feature):
        node_feature = torch.LongTensor(data_feature['id_features']).to(self.device)
        type_feature = torch.LongTensor(data_feature['node_labels']).to(self.device)
        length_feature = torch.LongTensor(data_feature['length_features']).to(self.device)
        lane_feature = torch.LongTensor(data_feature['lane_features']).to(self.device)
        return self.graph_enc(node_feature, type_feature, length_feature, lane_feature, self.adj)

    def predict(self, data_feature, mask):
        return self.forward(data_feature)[mask]


class HRNRPreprocess(object):
    def __init__(self, config, dataset, logger):
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.device = config.get("device", torch.device("cpu"))
        self._transfer_files()
        self._calc_transfer_matrix()
        self.logger.info("transfer matrix calculated.")

    def _transfer_files(self):
        """
        加载.geo .rel，生成HRNR所需的部分文件
        .geo
            [geo_id, type, coordinates, lane, type, length, bridge]
            from
            [geo_id, type, coordinates, highway, length, lanes, tunnel, bridge, maxspeed, width, alley, roundabout]
        .rel [rel_id, type, origin_id, destination_id]
        """
        # node_features [[lane, type, length, id]]
        data_feature = self.dataset.get_data_feature()

        self.lane_feature = torch.LongTensor(data_feature['lane_features'])
        self.type_feature = torch.LongTensor(data_feature['node_labels'])
        self.length_feature = torch.LongTensor(data_feature['length_features'])
        self.node_feature = torch.LongTensor(data_feature['id_features'])
        self.num_nodes = len(self.node_feature)
        self.adj_matrix = data_feature['adj_mx']

    def _calc_transfer_matrix(self):
        # calculate T^SR T^RZ with 2 loss functions
        self.logger.info("calculating transfer matrix...")
        self.cache_file_folder = "./output/dataset_cache/{}".format(self.config['dataset'])
        ensure_dir(self.cache_file_folder)
        # import pdb
        # pdb.set_trace()
        self.tsr = os.path.join(self.cache_file_folder, self.config.get("struct_assign", "struct_assign"))
        self.trz = os.path.join(self.cache_file_folder, self.config.get("fnc_assign", "fnc_assign"))
        if os.path.exists(self.tsr) and os.path.exists(self.trz):
            self.struct_assign = pickle.load(open(self.tsr, 'rb'))
            self.fnc_assign = pickle.load(open(self.trz, 'rb'))
            return

        self.node_emb_layer = nn.Embedding(self.config.get("node_num"), self.config.get("node_dims")).to(self.device)
        self.type_emb_layer = nn.Embedding(self.config.get("type_num"), self.config.get("type_dims")).to(self.device)
        self.length_emb_layer = nn.Embedding(self.config.get("length_num"), self.config.get("length_dims")).to(
            self.device)
        self.lane_emb_layer = nn.Embedding(self.config.get("lane_num"), self.config.get("lane_dims")).to(self.device)

        node_emb = self.node_emb_layer(self.node_feature)
        type_emb = self.type_emb_layer(self.type_feature)
        length_emb = self.length_emb_layer(self.length_feature)
        lane_emb = self.lane_emb_layer(self.lane_feature)

        # Segment, Region, Zone dimensions
        self.k1 = self.num_nodes
        self.k2 = self.config.get("struct_cmt_num")
        self.k3 = self.config.get("fnc_cmt_num")
        self.logger.info("k1: " + str(self.k1) + ", k2: " + str(self.k2) + ", k3: " + str(self.k3))

        NS = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        AS = torch.tensor(self.adj_matrix + np.array(np.eye(self.num_nodes)), dtype=torch.float)

        self.hidden_dims = self.config.get("hidden_dims")
        self.dropout = self.config.get("dropout")
        self.alpha = self.config.get("alpha")

        # 从计算图中分离出来
        if os.path.exists(self.tsr):
            TSR = pickle.load(open(self.tsr, 'rb'))
        else:
            TSR = self.calc_tsr(NS, AS).detach()
            pickle.dump(TSR, open(self.tsr, "wb"))

        AR = TSR.t().mm(AS).mm(TSR).detach()
        NR = TSR.t().mm(NS).detach()
        TRZ = self.calc_trz(NR, AR, TSR)

        self.struct_assign = TSR.clone().detach()
        self.fnc_assign = TRZ.clone().detach()
        pickle.dump(self.fnc_assign, open(self.trz, "wb"))

    def calc_tsr(self, NS, AS):
        TSR = None
        self.logger.info("calculating TSR...")

        # 谱聚类 求出M1
        sc = SpectralClustering(self.k2, affinity="precomputed",
                                n_init=1, assign_labels="discretize")
        sc.fit(self.adj_matrix)
        labels = sc.labels_
        M1 = [[0 for i in range(self.k2)] for j in range(self.k1)]
        for i in range(self.k1):
            M1[i][labels[i]] = 1
        M1 = torch.tensor(M1, dtype=torch.long, device=self.device)

        sparse_AS = get_sparse_adj(AS, self.device)
        SR_GAT = SPGAT(in_features=self.hidden_dims, out_features=self.k2,
                     alpha=self.alpha, dropout=self.dropout).to(self.device)
        self.logger.info("SR_GAT: " + str((self.k1, self.hidden_dims))
                          + " -> " + str((self.k1, self.k2)))
        loss1 = torch.nn.BCELoss()
        optimizer1 = torch.optim.Adam(SR_GAT.parameters(), lr=5e-2)  # TODO: lr
        optimizer1.zero_grad()
        for i in range(1):  # TODO: 迭代次数
            self.logger.info("epoch " + str(i))
            W1 = SR_GAT(NS, sparse_AS)
            TSR = W1 * M1
            # TSR = W1
            TSR = torch.softmax(TSR, dim=0)

            NR = TSR.t().mm(NS)
            _NS = TSR.mm(NR)
            _AS = torch.sigmoid(_NS.mm(_NS.t()))
            loss = loss1(_AS.reshape(self.k1 * self.k1), AS.reshape(self.k1 * self.k1))
            self.logger.info(" loss: " + str(loss))
            loss.backward(retain_graph=True)
            optimizer1.step()
            optimizer1.zero_grad()
        return TSR

    def calc_trz(self, NR, AR, TSR):
        TRZ = None
        RZ_GCN = GraphConvolution(in_features=self.hidden_dims, out_features=self.k3,
                     device=self.device).to(self.device)
        self.logger.info("RZ_GCN: " + str((self.k2, self.hidden_dims))
                          + " -> " + str((self.k2, self.k3)))
        self.logger.info("getting reachable matrix...")
        loss2 = torch.nn.MSELoss()
        optimizer2 = torch.optim.Adam(RZ_GCN.parameters(), lr=5e-2)  # TODO: lr
        optimizer2.zero_grad()
        C = torch.tensor(Utils(self.num_nodes, self.adj_matrix).get_reachable_matrix(), dtype=torch.float)
        self.logger.info("calculating TRZ...")
        for i in range(1):  # TODO: 迭代次数
            self.logger.info("epoch " + str(i))
            TRZ = RZ_GCN(NR.unsqueeze(0), AR.unsqueeze(0)).squeeze()
            TRZ = torch.softmax(TRZ, dim=0)

            NZ = TRZ.t().mm(NR)
            _NS = TSR.mm(TRZ).mm(NZ)
            _C = _NS.mm(_NS.t())
            loss = loss2(C.reshape(self.k1 * self.k1), _C.reshape(self.k1 * self.k1))
            self.logger.info(" loss: " + str(loss))
            loss.backward(retain_graph=True)
            optimizer2.step()
            optimizer2.zero_grad()
        return TRZ


class Utils:
    def __init__(self, n, adj):
        self.n = n
        # adj: coo_matrix
        self.adj = adj.toarray()
        self.v = [[] for i in range(self.n)]
        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.adj[i][j] != 0:
                    self.v[i].append(j)
        self.visited = [False for i in range(self.n)]
        self.t = [[0 for i in range(self.n)] for j in range(self.n)]
        self.temp = []

    def get_reachable_matrix(self):
        # TODO: 使用 osm extract 北京路网，然后用真实的北京下一跳数据
        """
        计算 4.3.2 eq.17 的可达矩阵，只使用邻接矩阵

        Returns:
            列表形式的矩阵
        """
        lam = 5  # lambda in eq.17
        for i in range(0, self.n):
            self.temp = []
            self.dfs(i, i, lam)
            for x in self.temp:
                self.visited[x] = False
        return self.t

    def dfs(self, start, cur, step):
        if step == 0 or self.visited[cur]:
            return
        self.visited[cur] = True
        self.t[start][cur] += 1
        self.temp.append(cur)
        for i in range(len(self.v[cur])):
            self.dfs(start, self.v[cur][i], step - 1)


class GraphEncoderTL(Module):
    def __init__(self, hparams, struct_assign, fnc_assign, struct_adj, device):
        super(GraphEncoderTL, self).__init__()
        self.hparams = hparams
        self.device = device
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.struct_adj = struct_adj

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(self.device)
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(self.device)
        self.length_emb_layer = nn.Embedding(hparams.length_num, hparams.length_dims).to(self.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(self.device)

        self.tl_layer_1 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign, self.device)
        self.tl_layer_2 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign, self.device)
        self.tl_layer_3 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign, self.device)

        self.init_feat = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.init_feat = raw_feat

        raw_feat = self.tl_layer_1(self.struct_adj, raw_feat, adj)
        raw_feat = self.tl_layer_2(self.struct_adj, raw_feat, adj)
        return raw_feat


def get_sparse_adj(adj, device):
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)

    adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1),
                               dtype=torch.long, device=device).t()
    adj_values = torch.tensor(adj.data, dtype=torch.float, device=device)
    adj_shape = adj.shape
    adj = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape).to(device)
    return adj.coalesce()


class GraphEncoderTLCore(Module):
    def __init__(self, hparams, struct_assign, fnc_assign, device):
        super(GraphEncoderTLCore, self).__init__()
        self.device = device
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign

        self.fnc_gcn = GraphConvolution(
            in_features=hparams.hidden_dims,
            out_features=hparams.hidden_dims,
            device=device).to(self.device)

        self.struct_gcn = GraphConvolution(
            in_features=hparams.hidden_dims,
            out_features=hparams.hidden_dims,
            device=self.device).to(self.device)

        self.node_gat = SPGAT(
            in_features=hparams.hidden_dims,
            out_features=hparams.hidden_dims,
            alpha=hparams.alpha, dropout=hparams.dropout).to(self.device)

        self.l_c = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.device)

        self.l_s = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, struct_adj, raw_feat, raw_adj):
        # forward
        self.raw_struct_assign = self.struct_assign
        self.raw_fnc_assign = self.fnc_assign

        self.struct_assign = self.struct_assign / (F.relu(torch.sum(self.struct_assign, 0) - 1.0) + 1.0)
        self.fnc_assign = self.fnc_assign / (F.relu(torch.sum(self.fnc_assign, 0) - 1.0) + 1.0)

        self.struct_emb = torch.mm(self.struct_assign.t(), raw_feat)
        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)

        # backward
        ## F2F
        self.fnc_adj = torch.sigmoid(torch.mm(self.fnc_emb, self.fnc_emb.t()))  # n_f * n_f
        self.fnc_adj = self.fnc_adj + torch.eye(self.fnc_adj.shape[0]).to(self.device) * 1.0
        self.fnc_emb = self.fnc_gcn(self.fnc_emb.unsqueeze(0), self.fnc_adj.unsqueeze(0)).squeeze()

        ## F2C
        fnc_message = torch.div(torch.mm(self.raw_fnc_assign, self.fnc_emb),
                                (F.relu(torch.sum(self.fnc_assign, 1) - 1.0) + 1.0).unsqueeze(1))

        self.r_f = self.sigmoid(self.l_c(torch.cat((self.struct_emb, fnc_message), 1)))
        self.struct_emb = self.struct_emb + 0.15 * fnc_message  # magic number: 0.15

        ## C2C
        struct_adj = F.relu(struct_adj - torch.eye(struct_adj.shape[1]).to(self.device) * 10000.0) + torch.eye(
            struct_adj.shape[1]).to(self.device) * 1.0
        self.struct_emb = self.struct_gcn(self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)).squeeze()

        ## C2N
        struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
        self.r_s = self.sigmoid(self.l_s(torch.cat((raw_feat, struct_message), 1)))
        raw_feat = raw_feat + 0.5 * struct_message

        ## N2N
        raw_feat = self.node_gat(raw_feat, raw_adj)
        return raw_feat


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape,
                b):  # indices, value and shape define a sparse tensor, it will do mm() operation with b
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SPGAT(Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SPGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, inputs, adj):
        inputs = inputs.squeeze()
        dv = 'cuda' if inputs.is_cuda else 'cpu'
        N = inputs.size()[0]
        edge_index = adj.indices()

        h = torch.mm(inputs, self.W)
        # h: N x out
        edge_h = torch.cat((h[edge_index[0, :], :], h[edge_index[1, :], :]), dim=1).t()  # 2*D x E
        values = self.a.mm(edge_h).squeeze()
        edge_value_a = self.leakyrelu(values)

        # softmax
        edge_value = torch.exp(edge_value_a - torch.max(edge_value_a))  # E
        e_rowsum = self.special_spmm(edge_index, edge_value, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1
        edge_value = self.dropout(edge_value)
        # edge_value: E
        h_prime = self.special_spmm(edge_index, edge_value, torch.Size([N, N]), h)
        # h_prime: N x out
        epsilon = 1e-15
        h_prime = h_prime.div(e_rowsum + torch.tensor([epsilon], device=dv))
        # h_prime: N x out
        if self.concat:  # if this layer is not last layer,
            return F.elu(h_prime)
        else:  # if this layer is last layer
            return h_prime


class GraphConvolution(Module):
    """
      Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        node_num = adj.shape[-1]
        # add remaining self-loops
        self_loop = torch.eye(node_num, dtype=torch.float).to(self.device)
        self_loop = self_loop.reshape((1, node_num, node_num))
        self_loop = self_loop.repeat(adj.shape[0], 1, 1)
        adj_post = adj + self_loop
        # signed adjacent matrix
        deg_abs = torch.sum(torch.abs(adj_post), dim=-1)
        deg_abs_sqrt = deg_abs.pow(-0.5)
        diag_deg = torch.diag_embed(deg_abs_sqrt, dim1=-2, dim2=-1)

        norm_adj = torch.matmul(torch.matmul(diag_deg, adj_post), diag_deg)
        return norm_adj

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        adj_norm = self.norm(adj)
        output = torch.matmul(support.transpose(1, 2), adj_norm.transpose(1, 2))
        output = output.transpose(1, 2)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst
