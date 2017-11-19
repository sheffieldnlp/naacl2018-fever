from sqlalchemy import Table
from sqlalchemy.orm import sessionmaker

from dataset.persistence.page import Page
from sqlalchemy.ext.declarative import declarative_base

def get_session(engine):
    Base = declarative_base()
    Session = sessionmaker(bind=engine)

    session =  Session()
    if not engine.dialect.has_table(engine, Page.__tablename__):
        Page.__table__.create(bind=engine,checkfirst=True)
    return session