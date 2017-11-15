from util.log_helper import LogHelper
import pickle

class Indexer:
    def __init__(self,file):
        self.pages = []
        self.file = file
        self.logger = LogHelper.get_logger(__name__)
        self.logger.info("Indexing Pages")

    def index_page(self,key):
        self.logger.debug("Index Page: {0}".format(key))
        self.pages.append(key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Saving index")
        pickle.dump(self.pages,self.file)
