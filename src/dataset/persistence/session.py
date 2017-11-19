from sqlalchemy.orm import sessionmaker

from dataset.persistence import engine
from dataset.persistence.page import Page
from sqlalchemy.ext.declarative import declarative_base


def get_session(engine):
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    if not engine.dialect.has_table(engine, Page.__tablename__):
        Base.metadata.create_all(engine)

    return Session()