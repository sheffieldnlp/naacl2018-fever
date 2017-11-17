import os

from dataset.block import Block
from dataset.corpus import Corpus
from util.log_helper import LogHelper
from gensim.models.tfidfmodel import *


def read_lines(wikifile):
    return [line for line in wikifile.split("\n")]

def read_text(wikifile):
    return [line.split('\t')[1] if len(line.split('\t'))>1 else "" for line in read_lines(wikifile) ]

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_words(wikifile):
    return flatten([line.split(" ") for line in read_text(wikifile)])

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("Prepare dataset")

    blocks = 1

    corpus = Corpus("page",os.path.join("data","fever"),blocks, read_words)
    for doc in corpus:
        d1 = doc

    #tfidf = TfidfModel(corpus)

