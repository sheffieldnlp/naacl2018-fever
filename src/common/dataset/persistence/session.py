from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from common.dataset.persistence.page import Page

def get_session(engine):
    Base = declarative_base()
    Session = sessionmaker(bind=engine)

    session =  Session()
    if not engine.dialect.has_table(engine, Page.__tablename__):
        Page.__table__.create(bind=engine,checkfirst=True)
    return session