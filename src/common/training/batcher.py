import torch
import numpy as np

from scipy.sparse import coo_matrix
from torch.autograd import Variable

from common.training.options import gpu


class Batcher():

    def __init__(self,data,size):
        self.data = data
        self.size = size
        self.pointer = 0

        if isinstance(self.data,coo_matrix):
            self.data = self.data.tocsr()

    def __next__(self):
        if self.pointer == splen(self.data):
            self.pointer = 0
            raise StopIteration

        next = min(splen(self.data),self.pointer+self.size)
        to_return = self.data[self.pointer : next]

        start,end = self.pointer,next

        self.pointer = next


        return to_return, splen(to_return), start, end

    def __iter__(self):
        return self

def splen(data):
    try:
        return data.shape[0]
    except:
        return len(data)



def prepare_with_labels(data,labels):
    data = data.todense()
    v = torch.FloatTensor(np.array(data))
    if gpu():
        return Variable(v.cuda()), Variable(torch.LongTensor(labels).cuda())
    return Variable(v), Variable(torch.LongTensor(labels))


def prepare(data):
    data = data.todense()
    v = torch.FloatTensor(np.array(data))
    if gpu():
        return Variable(v.cuda())
    return Variable(v)