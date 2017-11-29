import os

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import *

from common.dataset import Corpus
from common.util import LogHelper


def read_lines(wikifile):
    return [line for line in wikifile.split("\n")]

def read_text(wikifile):
    return [line.split('\t')[1] if len(line.split('\t'))>1 else "" for line in read_lines(wikifile) ]

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_words(wikifile):
    return flatten([line.split(" ") for line in read_text(wikifile)])

def read_dic(dic,pp):
    return lambda doc: dic.doc2bow(pp(doc))

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("Prepare dataset")

    blocks = 50

    corpus = Corpus("page",os.path.join("data","fever"),blocks, read_words)
    dic = Dictionary(corpus)
    dic.save(os.path.join("data","fever","dict"))

    corpus = Corpus("page",os.path.join("data","fever"),blocks, read_dic(dic,read_words))
    tfidf = TfidfModel(corpus)
    tfidf.save(os.path.join("data","fever","tfidf"))

