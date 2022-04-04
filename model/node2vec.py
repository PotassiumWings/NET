from . import walker
from gensim.models import Word2Vec
from model.AbstractModel import AbstractModel


class node2vec(AbstractModel):
    def __init__(self, config, dataset, logger, kwargs=None):
        super().__init__(config, dataset, logger, kwargs)
        if kwargs is None:
            kwargs = {}
        self.walker = walker.Walker(self.g, p=config.get("p", 1.0), q=config.get("q", 1.0), logger=self.logger)
        self.walker.preprocess_transition_probs()

        self.output_dim = config.get("output_dim", 128)

        sentences = self.walker.simulate_walks(
            num_walks=config["num_walks"],
            walk_length=config["walk_length"]
        )
        kwargs["sentences"] = sentences
        kwargs["min_count"] = config.get["min_count"]
        kwargs["sg"] = config["sg"]
        kwargs["workers"] = config["workers"]
        kwargs["vector_size"] = self.output_dim

        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in self.g.nodes():
            self.vectors[word] = word2vec.wv[word]
