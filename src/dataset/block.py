import pickle

import os


class Block(object):
    def __init__(self,block,name,path):
        self.volume = block
        self.path = path
        self.name = name
        self.data = dict()

    def save(self,name,data):
        self.data[name] = data

    def write(self):
        print("Write block " + str(self.volume))
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

    def list(self):
        return self.data.keys()

    def load(self):
        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p")), "rb") as f:
            self.data = pickle.load(self.data, f)