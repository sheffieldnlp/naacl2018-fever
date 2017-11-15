import pickle

import os


class Writer(object):
    def __init__(self,name,path,limit=250000):
        self.volume = 0
        self.name = name
        self.path = path
        self.data = dict()
        self.size = 0
        self.limit = limit

    def save(self,name,data):

        self.data[name] = data
        self.size += 1

        if self.size % 10000 == 0:
            print(self.size)

        if self.size >= self.limit:
            self.write()

    def write(self):
        print("Write block " + str(self.volume))
        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p")), "wb+") as f:
            pickle.dump(self.data, f)

        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p.idx")), "wb+") as f:
            pickle.dump(set(self.data.keys()), f)

        self.data = dict()
        self.volume += 1
        self.size = 0

    def close(self):
        self.write()