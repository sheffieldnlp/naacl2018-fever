import re

def uninformative(title):
    return  '-LRB-disambiguation-RRB-' in title.lower() \
            or '-LRB-disambiguation_page-RRB-' in title.lower() \
            or title.lower().startswith('list_of_') \
            or title.lower().startswith('index_of_.') \
            or title.lower().startswith('outline_of_')

def preprocess(doc):
    return doc if not uninformative(doc['id']) else None