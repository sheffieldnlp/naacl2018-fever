import os
import sys

from tqdm import tqdm

from common.dataset import Corpus
from common.dataset import Page
from common.dataset import get_engine
from common.dataset import get_session
from common.util import LogHelper

if __name__ == "__main__":
    blocks = int(sys.argv[1])
    LogHelper.setup()
    logger = LogHelper.get_logger("convert")



    blk = Corpus("page",os.path.join("data","fever"),blocks,lambda x:x)
    engine = get_engine("pages")
    session = get_session(engine)


    for idx,data in tqdm(enumerate(blk)):
        page, body = data
        p = Page(name=page, doc=body)
        session.add(p)

        if idx%10000 == 9999:
            logger.info("Commit")
            session.commit()

    session.commit()