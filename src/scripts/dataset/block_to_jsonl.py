import json
import os
import sys

from tqdm import tqdm

from common.dataset.corpus import Corpus
from common.util.log_helper import LogHelper


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


class BlockWriter():
    def __init__(self,path,max):
        self.added = 0
        self.block = 0
        self.max = max
        self.path = path
        self.file = None

    def write(self,line):
        self.file.write(line+"\n")

        if self.added % self.max == self.max - 1:
            self.nextblock()

        self.added +=1


    def __enter__(self):
        self.nextblock()
        return self


    def nextblock(self):
        self.block = (self.block + 1) if self.block is not None else 0
        if self.file is not None:
            self.file.close()
        self.file = open(os.path.join("data", "fever", "wiki", "wiki-{0}.jsonl".format(str.zfill(str(self.block),3))), "w+")


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()



if __name__ == "__main__":
    blocks = int(sys.argv[1])
    LogHelper.setup()
    logger = LogHelper.get_logger("convert")

    blk = Corpus("page",os.path.join("data","fever"),blocks,lambda x:(x,read_words(x)))


    with BlockWriter(os.path.join("data","fever","wiki"),50000) as f:
        for page, body in tqdm(blk):
            f.write(json.dumps({"id":page , "text": " ".join(body[1]),"lines":body[0]}))
