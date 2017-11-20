import sys

import os

import json

from dataset.corpus import Corpus
from dataset.persistence.engine import get_engine
from dataset.persistence.page import Page

from tqdm import tqdm

from dataset.persistence.session import get_session
from util.log_helper import LogHelper



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
    blocks = int(sys.argv[1])
    LogHelper.setup()
    logger = LogHelper.get_logger("convert")

    blk = Corpus("page",os.path.join("data","fever"),blocks,read_words)


    with open(os.path.join("data","fever","wiki","all-wiki.jsonl"),"w+") as f:
        for idx,data in tqdm(enumerate(blk)):

            page, body = data
            f.write(json.dumps({"id":page , "text": " ".join(body)})+"\n")

        #    if idx % 10000 == 9999:
        #        logger.info("Commit")
        #        session.commit()

