import argparse
import json

from common.util.log_helper import LogHelper
from tqdm import tqdm
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def tf_idf_sim(claim, lines):
    test = []
    for line in lines:
        test.append({"claim": claim, "text": line})

    return tf.lookup(test).reshape(-1).tolist()

class XTermFrequencyFeatureFunction(TermFrequencyFeatureFunction):
    def texts(self, data):
        return [item["text"] for item in data]

    def process(self, data):
        claim_bow = self.bow_vectorizer.transform(self.claims(data))
        claim_tfs = self.tfreq_vectorizer.transform(claim_bow)
        claim_tfidf = self.tfidf_vectorizer.transform(self.claims(data))

        body_texts = self.texts(data)
        body_bow = self.bow_vectorizer.transform(body_texts)
        body_tfs = self.tfreq_vectorizer.transform(body_bow)
        body_tfidf = self.tfidf_vectorizer.transform(body_texts)

        cosines = np.array([cosine_similarity(c, b)[0] for c, b in zip(claim_tfidf, body_tfidf)])

        return cosines

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('out_file', type=str, help='/path/to/saved/db.db')

    parser.add_argument('max_page',type=int)
    parser.add_argument('max_sent',type=int)

    args = parser.parse_args()

    db = FeverDocDB("data/fever/fever.db")
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    train_ds = DataSet(file="data/fever/train.ns.pages.p{0}.jsonl".format(1), reader=jlr, formatter=formatter)
    dev_ds = DataSet(file="data/fever/dev.pages.p{0}.jsonl".format(1), reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()

    tf = XTermFrequencyFeatureFunction(db)
    tf.inform(train_ds.data, dev_ds.data)


    jlr = JSONLineReader()
    with open(args.in_file,"r") as f:
        lines = jlr.process(f)


    files = []
    for i in range(1,args.max_sent):
        files.append(open("{0}.{1}.final.jsonl".format(args.out_file,i),"w+"))


    for line in tqdm(lines):
        if 'predicted_pages' in line:
            sorted_p = list(sorted(line['predicted_pages'],reverse=True, key=lambda elem:elem[1]))

            pages = [p[0] for p in sorted_p[:args.max_page]]
            p_lines = []
            for page in pages:
                lines = db.get_doc_lines(page)
                lines = [line.split("\t")[1] if len(line.split("\t")[1])>1 else "" for line in lines.split("\n")]

                p_lines.extend(zip(lines,[page]*len(lines),range(len(lines))))

            scores = tf_idf_sim(line["claim"],[pl[0] for pl in p_lines])
            scores = list(zip(scores,[pl[1] for pl in p_lines],[pl[2] for pl in p_lines],[pl[0] for pl in p_lines]))
            scores = list(filter(lambda score:len(score[3].strip()),scores))
            sentences_l = list(sorted(scores,reverse=True,key=lambda elem:elem[0]))

            for i in range(1,args.max_sent):

                line["predicted_sentences"] = [(s[1],s[2]) for s in sentences_l[:i]]
                files[i - 1].write(json.dumps(line)+"\n")


    for file in files:
        file.close()