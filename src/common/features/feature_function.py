import numpy as np


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

    def inform(self,dataset):
        preprocessed = self.preprocess_all(dataset.data)
        print(len(preprocessed))
        for feature_function in self.feature_functions:
            print("Inform {0}".format(feature_function))
            feature_function.inform(preprocessed)

class FeatureFunction():

    def __init__(self):
        pass

    def inform(self,train,dev,test):
        raise NotImplementedError("Not Implemented Here")

    def lookup(self,data):
        return self.process(data)

    def process(self,data):
        pass
