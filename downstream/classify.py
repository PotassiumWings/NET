import numpy
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression


# multi label
class classify(object):
    def __init__(self, config, data_feature, vectors, logger):
        self.config = config
        self.logger = logger
        self.data_feature = data_feature
        self.embeddings = vectors
        self.clf = TopKRanker(LogisticRegression())
        self.binarizer = LabelBinarizer(sparse_output=True)

    def train(self, train_dataloader, eval_dataloader):
        X = train_dataloader["mask"]
        Y_all = train_dataloader['node_labels']
        Y = Y_all[X]
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, test_dataloader):
        X = test_dataloader["mask"]
        Y = test_dataloader['node_labels'][X]
        top_k_list = [1 for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        self.logger.info(results)
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)

