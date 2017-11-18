from collections import defaultdict

import pickle
from tqdm import tqdm


class ReverseIndex:
    def __init__(self,docs):
        self.lookup = defaultdict(set)

        if docs is not None:
            for title,words in tqdm(docs):
                self.add(title,words)

    def add(self,title,words):
        for word in words:
            self.lookup[word].add(title)

    def docs(self,phrase):
        ret = []
        for word in phrase:
            ret.extend(self.lookup[word])
        return ret

    def save(self,file):
        pickle.dump(self.lookup,file)

    def load(self,file):
        self.lookup = pickle.load(file)