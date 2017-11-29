import numpy as np


class Features():
    def __init__(self,features=list(),label_name="label"):
        self.feature_functions = features
        self.vocabs = dict()
        self.label_name = label_name

    def load(self,dataset):
        fs = []
        for feature_function in self.feature_functions:
            print("Load {0}".format(feature_function))
            fs.append(feature_function.lookup(dataset))
        return np.hstack(fs) if len(fs)>1 else fs,self.labels(dataset)

    def labels(self,data):
        return [datum[self.label_name] for datum in data]

    def inform(self,train,dev=None,test=None):
        for feature_function in self.feature_functions:
            print("Inform {0} with {1} data".format(feature_function,len(train.data)))

            feature_function.inform(train.data,
                                    dev.data if dev is not None else None,
                                    test.data if test is not None else None)

class FeatureFunction():

    def __init__(self):
        pass

    def inform(self,train,dev,test):
        raise NotImplementedError("Not Implemented Here")

    def lookup(self,data):
        return self.process(data)

    def process(self,data):
        pass
