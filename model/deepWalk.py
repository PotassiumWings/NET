from model.AbstractModel import AbstractModel
from . import walker
from gensim.models import Word2Vec


class deepWalk(AbstractModel):
    def __init__(self, config, g, logger, kwargs=None):
        super().__init__(config)
        if kwargs is None:
            kwargs = {}
        self.config = config
        self.logger = logger
        self.walker = walker.BasicWalker(g, logger=self.logger)

        self.output_dim = config.get("output_dim", 128)

        sentences = self.walker.simulate_walks(
            num_walks=config["num_walks"],
            walk_length=config["walk_length"]
        )
        kwargs["sentences"] = sentences
        kwargs["min_count"] = config["min_count"]
        kwargs["sg"] = config["sg"]
        kwargs["workers"] = config["workers"]
        kwargs["vector_size"] = self.output_dim
        kwargs["hs"] = 1
        kwargs["window"] = config["window_size"]

        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in g.nodes():
            self.vectors[word] = word2vec.wv[word]
