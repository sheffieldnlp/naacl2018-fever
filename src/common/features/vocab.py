from collections import defaultdict
from scipy.sparse import dok_matrix
from tqdm import tqdm
class Vocab():
    def __init__(self):
        self.vocab = set()
        self.vocab.add("UNKNOWN")

    def add(self,all_items):
        for item in all_items:
            for f in item:
                self.vocab.add(f)

    def generate_dict(self):
        vocab = dict()
        for i,word in enumerate(self.vocab):
            vocab[word] = i
        self.vocab = vocab

    def lookup(self,instances):

        rr = []
        for instance in instances:
            ret = defaultdict(int)
            for feature in instance:
                if feature in self.vocab:
                    ret[self.vocab[feature]] += 1
                else:
                    ret[self.vocab["UNKNOWN"]] += 1

            rr.append(ret)

        return rr

    def lookup_sparse(self, data, data_size):
        vocab_size = len(self.vocab)

        dok = dok_matrix((data_size,vocab_size))


        for idx,instance in tqdm(enumerate(data)):
            for feature in instance:
                dim = self.vocab[feature] if feature in self.vocab else self.vocab["UNKNOWN"]
                dok[idx, dim] +=1

        return dok