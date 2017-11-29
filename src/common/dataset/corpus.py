from common.dataset.block import Block
from common.util.log_helper import LogHelper


class Corpus:

    def __init__(self,name,path,blocks,preprocessing=None):
        self.logger = LogHelper.get_logger(Corpus.__name__)
        self.name = name
        self.path = path
        self.blocks = blocks
        self.active_block_iter = None
        self.active_block = None
        self.active_block_number = None
        self.preprocessing = preprocessing

    def __iter__(self):
        self.active_block_iter = None
        self.active_block = None
        self.active_block_number = None
        return self


    def next_block(self):
        if self.active_block_number is None:
            self.active_block_number = 0
        else:
            self.active_block_number += 1

        self.logger.info("Trying to load block {0}".format(self.active_block_number))
        if self.active_block_number >= self.blocks:
            self.logger.info("No more blocks")
            raise StopIteration

        self.active_block = Block(self.active_block_number, self.name,self.path)
        self.active_block_iter = iter(self.active_block)

    def __next__(self):
        # Check if we have started with a block
        if self.active_block_iter is None:
            self.next_block()

        # Get the next item from this block
        try:
            n = next(self.active_block_iter)

        except StopIteration:
            # If the block is exhausted, try and get next from the next block
            try:
                self.next_block()
                n = next(self.active_block_iter)
            except StopIteration as e:
                # If we're out of blocks, reset and stop iteration
                self.active_block_iter = None
                self.active_block = None
                self.active_block_number = None
                raise e

        return n, self.preprocessing(self.active_block[n])

    def __getitem__(self, item):
        return self.active_block[item]
