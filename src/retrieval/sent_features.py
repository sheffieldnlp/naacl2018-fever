from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import hstack

import numpy as np

from rte.riedel.fever_features import TermFrequencyFeatureFunction


class SentenceTermFrequencyFeatureFunction(TermFrequencyFeatureFunction):

    def __init__(self,doc_db,lim_unigram=5000,naming=None):
        super().__init__(doc_db,lim_unigram,naming=naming)
        self.ename = "sentences"

    def bodies(self,data):
        return set([datum[self.ename] for datum in data])

    def texts(self,data):
        return set([datum[self.ename] for datum in data])

    def body_id(self,data):
        return [datum[self.ename] for datum in data]


