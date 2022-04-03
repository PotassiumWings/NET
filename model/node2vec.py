from . import walker
from gensim.models import Word2Vec


class node2vec(object):
    def __init__(self, g, config):
        self.config = config
        self.walker = walker.Walker(g, p=config.get("p", 1.0), q=config.get("q", 1.0))
        self.walker.preprocess_transition_probs()

        self.size = config.get("size", 128)

        sentences = self.walker.simulate_walks(
            num_walks=config.get("num_paths", 10),
            walk_length=config.get("walk_length", 80)
        )
        kwargs = {}
        kwargs["sentences"] = sentences
        kwargs["min_count"] = 0
        kwargs["sg"] = 1
        kwargs["workers"] = 1

        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in g.G.nodes():
            self.vectors[word] = word2vec.wv[word]

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
