from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import  Column, Integer, String, Text

Base = declarative_base()


class Page(Base):
    __tablename__ = "page"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    doc = Column(Text)
    raw = Column(Text)
