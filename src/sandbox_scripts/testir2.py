import argparse

from drqa import retriever
from drqa.retriever import DocDB


def process(db, ranker, query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)

    for page in doc_names:
        print(ranker.text2spvec(db.get_doc_text(page)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--db', type=str, default=None)
    args = parser.parse_args()

    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
    db = DocDB(args.db)
    pages = process(db, ranker,"banana")



