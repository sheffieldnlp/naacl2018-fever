from scipy.sparse import hstack
from features.vocab import Vocab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
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


class LexFeatureFunction(FeatureFunction):

    def __init__(self):
        super().__init__()
        self.vocab = Vocab()

    def inform(self,data):
        generated = self.process(data)
        self.vocab.add(generated)
        self.vocab.generate_dict()

    def lookup(self,data):
        size = len(data)
        processed = self.process(data)

        lu = self.vocab.lookup_sparse(processed,size)
        return lu



class BleuOverlapFeatureFuntion(FeatureFunction):
    def process(self,data):
        smth = SmoothingFunction().method3
        return np.array(list(map(lambda instance: [sentence_bleu([instance["s2_words"]], instance["s1_words"],smoothing_function=smth)],data)))

class UnigramFeatureFunction(LexFeatureFunction):
    def process(self,data):
        return map(lambda text: [w for w in " ".join(text.split()).split()], data)

class BigramFeatureFunction(LexFeatureFunction):
    def process(self, data):
        return  map(lambda text: ["_".join(ng) for ng in ngrams(" ".join(text.split()), 2)], data)

class CharNGramFeatureFunction(LexFeatureFunction):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def process(self,data):
        return map(lambda text: ["".join(ng) for ng in char_ngrams(" ".join(text.split()), self.size)], data)




