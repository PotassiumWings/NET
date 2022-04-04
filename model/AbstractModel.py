from utils import ensure_dir


class AbstractModel(object):
    def __init__(self, config, dataset, logger, kwargs=None):
        self.config = config
        self.g = dataset.G
        self.dataset = dataset
        self.logger = logger
        self.vectors = None
        self.output_dim = 0

    def save_embeddings(self):
        ensure_dir("./output/{}".format(self.config["exp_id"]))
        filename = "./output/{}/embeddings.emb".format(self.config["exp_id"])
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.output_dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
