import os
import unicodedata

from gensim.models.tfidfmodel import *

from common.dataset import Corpus
from common.dataset import Page
from common.dataset import get_engine
from common.dataset import get_session
from common.util import LogHelper


def read_lines(wikifile):
    return [line for line in wikifile.split("\n")]

def read_text(wikifile):
    return [normalize(line.split('\t')[1]) if len(line.split('\t'))>1 else "" for line in read_lines(wikifile) ]

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_words(wikifile):
    return flatten([line.split(" ") for line in read_text(wikifile)])


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def read_dic(dic,pp):
    return lambda doc: dic.doc2bow(pp(doc))

# Stop words and preprocessing https://github.com/facebookresearch/DrQA/blob/master/drqa/retriever/utils.py
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


def stopWord(w):
    return w in STOPWORDS

def preprocess(words):
    return list(filter(lambda w: not stopWord(w), map(lambda w:w.lower(),words)))

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("FEVER IR")

    blocks = 50
    pp = preprocess
    corpus = Corpus("page",os.path.join("data","fever"),blocks,read_words)

    engine = get_engine("pages")
    session = get_session(engine)


    logger.info("Query")
    res = session.query(Page).filter(Page.doc.like("%leonardo%"))

    for r in res:
        logger.info("Result: {0}".format(r.name))


    #if not os.path.exists(os.path.join("data","fever","reverse_index_unigram.p")):
    #    logger.warn("Reverse index missing - reconstructing")
    #    ri = ReverseIndex(corpus, pp)
    #    ri.save(os.path.join("data","fever","reverse_index_unigram.p"))
    #else:
    #    logger.info("Loading Reverse Index")
    #    ri = ReverseIndex(None,pp)
    #    ri.load(os.path.join("data","fever","reverse_index_unigram.p"))

    #logger.info("Done")
    #print(ri.docs("Leonardo went to the sea".split()))

    #dic = Dictionary.load(os.path.join("data", "fever", "dict"))
    #corpus = Corpus("page",os.path.join("data","fever"),blocks, read_dic(dic,read_words))
    #tfidf = TfidfModel.load(os.path.join("data" ,"fever","tfidf"))



#    tfidf = TfidfModel(corpus,dictionary=dic)
#    sim = Similarity(os.path.join("data" ,"fever","sim"),tfidf,num_features=1182440)


#st = "Leonardo da Vinci was an inventor .".split(" ")
#¢dict = Dictionary.load(os.path.join("data" ,"fever","dict"))
#tfidf = TfidfModel.load(os.path.join("data" ,"fever","tfidf"))
#print(tfidf[dict.doc2bow(st)])


