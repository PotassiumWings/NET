from . import walker
from gensim.models import Word2Vec
from model.AbstractModel import AbstractModel


class node2vec(AbstractModel):
    def __init__(self, g, config, kwargs=None):
        super().__init__(config)
        if kwargs is None:
            kwargs = {}
        self.config = config
        self.walker = walker.Walker(g, p=config.get("p", 1.0), q=config.get("q", 1.0))
        self.walker.preprocess_transition_probs()

        self.size = config.get("size", 128)

        sentences = self.walker.simulate_walks(
            num_walks=config.get("num_paths", 10),
            walk_length=config.get("walk_length", 80)
        )
        kwargs["sentences"] = sentences
        kwargs["min_count"] = config.get("min_count", 0)
        kwargs["sg"] = config.get("sg", 1)
        kwargs["workers"] = config.get("workers", 1)
        kwargs["vector_size"] = self.size

        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in g.nodes():
            self.vectors[word] = word2vec.wv[word]
