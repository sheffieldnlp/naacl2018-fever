from shared.features.vocab import Vocab
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from shared.features.vocab import Vocab


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def char_ngrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


class Features():
    def __init__(self,features=list(),preprocessing=None):
        self.preprocessing = preprocessing
        self.feature_functions = features
        self.vocabs = dict()

    def load(self,dataset):
        fs = []
        preprocessed = self.preprocess_all(dataset.data)
        for feature_function in self.feature_functions:
            print("Load {0}".format(feature_function))
            fs.append(feature_function.lookup(preprocessed))
        return np.hstack(fs),self.labels(dataset.data)

    def labels(self,data):
        return [datum["label"] for datum in data]

    def preprocess_all(self,data):
        return list(
            map(
                lambda datum: self.preprocessing(datum["data"]) if self.preprocessing is not None else datum["data"],
                data))

    def generate_vocab(self,dataset):
        preprocessed = self.preprocess_all(dataset.data)
        print(len(preprocessed))
        for feature_function in self.feature_functions:
            print("Inform {0}".format(feature_function))
            feature_function.inform(preprocessed)

class FeatureFunction():

    def __init__(self):
        pass

    def inform(self,data):
        return self.process(data)

    def lookup(self,data):
        return self.process(data)

    def process(self,data):
        pass




