import argparse
import json
from tqdm import tqdm
from drqa import retriever
from drqa.retriever import DocDB


def process(db, ranker, query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)

    return doc_names



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--db', type=str, default=None)
    args = parser.parse_args()

    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
    db = DocDB(args.db)

    with open("data/fever/fever.dev.jsonl","r") as f:
        with open("data/fever/example.jsonl","w+") as f2:
            for line in tqdm(f.readlines()):

                js = json.loads(line)

                print(js['claim'])

                pages = process(db, ranker,js['claim'],k=5)

                for page in pages:
                    f2.write(json.dumps({"question":js['claim'],"doc":page})+"\n")
                print(pages)
                print("")






