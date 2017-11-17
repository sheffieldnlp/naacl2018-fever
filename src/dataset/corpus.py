from dataset.block import Block


class Corpus:

    def __init__(self,name,path,blocks):
        self.name = name
        self.path = path
        self.blocks = blocks
        self.active_block = None
        self.active_block_number = None

    def __iter__(self):
        return self


    def next_block(self):
        if self.active_block_number is None:
            self.active_block_number = 0
        else:
            self.active_block_number += 1

        if self.active_block >= self.blocks:
            raise StopIteration

        self.active_block = Block(self.active_block_number, self.name,self.path)

    def __next__(self):
        # Check if we have started with a block
        if self.active_block is None:
            self.next_block()

        # Get the next item from this block
        try:
            n = next(self.active_block)

        except StopIteration:
            # If the block is exhausted, try and get next from the next block
            try:
                self.next_block()
                n = next(self.active_block)
            except StopIteration as e:
                # If we're out of blocks, reset and stop iteration
                self.active_block = None
                self.active_block_number = None
                raise e

        return n

