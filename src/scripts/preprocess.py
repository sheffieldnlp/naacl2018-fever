import re
def useless(title):
    return  '-LRB-disambiguation-RRB-' in title.lower() or '-LRB-disambiguation_page-RRB-' in title.lower() or re.match(r'(List_of_.+)|(Index_of_.+)|(Outline_of_.+)',  title)
def preprocess(doc):
    return doc if not  useless(doc['id']) else None