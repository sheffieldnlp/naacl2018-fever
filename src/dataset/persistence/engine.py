from sqlalchemy import create_engine


def get_engine(file):
    return create_engine('sqlite:///data/fever/{0}.db'.format(file), echo=False)
