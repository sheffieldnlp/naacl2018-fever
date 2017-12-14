import numpy as np
import os
import pickle
class Features():
    def __init__(self,features=list(),label_name="label",base_path="features"):
        self.feature_functions = features
        self.vocabs = dict()
        self.label_name = label_name
        self.base_path = base_path


    def load(self,train,dev=None,test=None):
        train_fs = []
        dev_fs = []
        test_fs = []

        for ff in self.feature_functions:
            ffpath = os.path.join(self.base_path, ff.get_name())

            if not os.path.exists(ffpath):
                os.makedirs(ffpath)

            #If we need train/dev/test data and these don't exist, we have to recreate the features
            if (not os.path.exists(os.path.join(ffpath,"train"))) \
                or (dev is not None and not os.path.exists(os.path.join(ffpath,"dev"))) \
                or (test is not None and not os.path.exists(os.path.join(ffpath, "test"))) or \
                os.getenv("DEBUG").lower() in ["y", "1", "t", "yes"]:

                self.inform(train,dev,test)

            train_fs.append(self.generate_or_load(ff, train, "train"))
            dev_fs.append(self.generate_or_load(ff, dev, "dev"))
            test_fs.append(self.generate_or_load(ff, test, "test"))


        return self.out(train_fs,train), self.out(dev_fs,dev), self.out(test_fs,test)

    def out(self,features,ds):
        if ds is not None:
            return np.hstack(features) if len(features) > 1 else features[0], self.labels(ds.data)
        return [[]],[]

    def generate_or_load(self,feature,dataset,name):
        ffpath = os.path.join(self.base_path, feature.get_name())

        if dataset is not None:
            if os.path.exists(os.path.join(ffpath,name)) and os.getenv("DEBUG").lower() not in ["y","1","t","yes"]:
                print("Loading Features for {0}.{1}".format(feature, name))
                with open(os.path.join(ffpath, name), "rb") as f:
                    features = pickle.load(f)

            else:
                print("Generating Features for {0}.{1}".format(feature,name))
                features = feature.lookup(dataset.data)

                with open(os.path.join(ffpath, name), "wb+") as f:
                    pickle.dump(features, f)

            return features

        return None

    def _load(self,dataset):
        fs = []
        for feature_function in self.feature_functions:
            print("Load {0}".format(feature_function))
            fs.append(feature_function(dataset))
        return np.hstack(fs) if len(fs) > 1 else fs, self.labels(dataset)

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
