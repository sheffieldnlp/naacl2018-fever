import os
import pickle

from common.util.log_helper import LogHelper


class Block(object):
    def __init__(self,block,name,path):
        self.logger = LogHelper.get_logger(Block.__name__)
        self.volume = block
        self.path = path
        self.name = name
        self.data = None

    def save(self,name,data):
        self.data[name] = data

    def write(self):
        self.logger.info("Write block {0}".format(self.volume))
        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p")), "wb+") as f:
            pickle.dump(self.data, f)

        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p.idx")), "wb+") as f:
            pickle.dump(set(self.data.keys()), f)

        self.data = dict()

    def close(self):
        self.write()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self

    def __getitem__(self, item):
        return self.data[item]

    def list(self):
        return self.data.keys()

    def load(self):
        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p")), "rb") as f:
            self.data = pickle.load(f)
            self.logger.info("Loaded {0} articles".format(len(self.data)))

    def __iter__(self):
        if self.data is None:
            self.logger.info("Load block {0}".format(self.volume))
            self.load()
        return iter(self.data)

