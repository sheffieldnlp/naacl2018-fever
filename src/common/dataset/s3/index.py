import pickle

from common.util.log_helper import LogHelper


class Indexer:
    def __init__(self,file):
        self.pages = []
        self.file = file
        self.logger = LogHelper.get_logger(__name__)
        self.logger.info("Indexing Pages")

    def index_page(self,key):
        self.logger.debug("Index Page: {0}".format(key))
        self.pages.append(key)

    def load(self):
        self.pages.extend(pickle.load(self.file))

    def get_block(self,block,num_blocks=50):
        return self.pages[block*len(self.pages)//num_blocks:(block+1)*len(self.pages)//num_blocks]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Saving index")
        pickle.dump(self.pages,self.file)
