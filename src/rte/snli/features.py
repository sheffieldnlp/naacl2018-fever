from common.features.feature_function import FeatureFunction


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


