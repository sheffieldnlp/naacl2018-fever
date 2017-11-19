import sys

import os

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dataset.corpus import Corpus

Base = declarative_base()

from tqdm import tqdm

from util.log_helper import LogHelper


engine = create_engine('sqlite:///data/fever/pages.db')


blocks = int(sys.argv[1])

LogHelper.setup()
logger = LogHelper.get_logger("convert")


class Page(Base):
    __tablename__ = "page"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    doc = Column(Text)



Session = sessionmaker(bind=engine)
if not engine.dialect.has_table(engine, Page.__tablename__):
    Base.metadata.create_all(engine)

session = Session()
blk = Corpus("page",os.path.join("data","fever"),blocks,lambda x:x)



for page,body in tqdm(blk):
    p = Page(name=page, doc=body)
    session.add(p)
session.commit()