def uninformative(title):
    return title.lower().startswith('list_of_') \
        or title.lower().startswith("lists_of_") \
        or title.lower().startswith('index_of_.') \
        or title.lower().startswith('outline_of_')

def preprocess(doc):
    return None if uninformative(doc['id']) else doc