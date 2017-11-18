import os
from gensim.corpora import Dictionary
from gensim.models import TfidfModel



import os

from gensim.corpora import Dictionary
from gensim.similarities import Similarity

from dataset.corpus import Corpus
from dataset.reverse_index import ReverseIndex
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

def read_dic(dic,pp):
    return lambda doc: dic.doc2bow(pp(doc))

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("Prepare dataset")

    blocks = 1

    dic = Dictionary.load(os.path.join("data" ,"fever","dict"))

    corpus = Corpus("page",os.path.join("data","fever"),blocks, read_dic(dic,read_words))
    ri = ReverseIndex(corpus)
    print(ri.docs(dic.doc2bow("Leonardo went to the sea".split())))


#    corpus = Corpus("page",os.path.join("data","fever"),blocks, read_dic(dic,read_words))
#    tfidf = TfidfModel(corpus,dictionary=dic)
#    sim = Similarity(os.path.join("data" ,"fever","sim"),tfidf,num_features=1182440)


#st = "Leonardo da Vinci was an inventor .".split(" ")
#¢dict = Dictionary.load(os.path.join("data" ,"fever","dict"))
#tfidf = TfidfModel.load(os.path.join("data" ,"fever","tfidf"))
#print(tfidf[dict.doc2bow(st)])


