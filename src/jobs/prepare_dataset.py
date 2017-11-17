import os

from dataset.block import Block
from util.log_helper import LogHelper
from gensim.models.tfidfmodel import *
if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("Prepare dataset")

    blocks = 50

    for blk in range(blocks):
        block = Block(blk, "page", os.path.join("data", "fever"))
        block.load()


    tfidf = TfidfModel(corpus)

