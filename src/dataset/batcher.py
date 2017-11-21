
class Batcher():

    def __init__(self,data,size):
        self.data = data
        self.size = size
        self.pointer = 0

    def __next__(self):
        if self.pointer == len(self.data):
            raise StopIteration

        next = min(len(self.data),self.pointer+self.size)
        to_return = self.data[self.pointer : next]
        self.pointer = next


        return to_return, len(to_return)

    def __iter__(self):
        return self