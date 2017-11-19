import sys

import os

from dataset.corpus import Corpus
from dataset.persistence.engine import get_engine
from dataset.persistence.page import Page

from tqdm import tqdm

from dataset.persistence.session import get_session
from util.log_helper import LogHelper

if __name__ == "__main__":
    blocks = int(sys.argv[1])
    LogHelper.setup()
    logger = LogHelper.get_logger("convert")



    blk = Corpus("page",os.path.join("data","fever"),blocks,lambda x:x)
    engine = get_engine("pages")
    session = get_session(engine)


    for page,body in tqdm(blk):
        p = Page(name=page, doc=body)
        session.add(p)
    session.commit()