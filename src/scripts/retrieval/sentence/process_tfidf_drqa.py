import argparse
import json
from multiprocessing.pool import ThreadPool

from drqa.retriever import utils

from common.util.log_helper import LogHelper
from tqdm import tqdm
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
import numpy as np

from drqascripts.retriever.build_tfidf_lines import OnlineTfidfDocRanker


def tf_idf_sim(claim, lines,freqs=None):
    tfidf = OnlineTfidfDocRanker(args,[line["sentence"] for line in lines],freqs)
    line_ids,scores = tfidf.closest_docs(claim,args.max_sent)
    ret_lines = []
    for idx,line in enumerate(line_ids):
        ret_lines.append(lines[line])
        ret_lines[-1]["score"] = scores[idx]
    return ret_lines



def tf_idf_claim(line):
    if 'predicted_pages' in line:
        sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))

        pages = [p[0] for p in sorted_p[:args.max_page]]
        p_lines = []
        for page in pages:
            lines = db.get_doc_lines(page)
            lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in
                     lines.split("\n")]

            p_lines.extend(zip(lines, [page] * len(lines), range(len(lines))))

        lines = []
        for p_line in p_lines:
            lines.append({
                "sentence": p_line[0],
                "page": p_line[1],
                "line_on_page": p_line[2]
            })

        scores = tf_idf_sim(line["claim"], lines, doc_freqs)

        line["predicted_sentences"] = [(s["page"], s["line_on_page"]) for s in scores]
    return line


def tf_idf_claims_batch(lines):
    with ThreadPool(args.num_workers) as threads:
        results = threads.map(tf_idf_claim, lines)
    return results

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()


    parser.add_argument('--db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--model', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--in_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--max_page',type=int)
    parser.add_argument('--max_sent',type=int)
    parser.add_argument('--use_precomputed', type=str2bool, default=True)
    parser.add_argument('--split', type=str)
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(np.math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))

    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()
    doc_freqs=None
    if args.use_precomputed:
        _, metadata = utils.load_sparse_csr(args.model)
        doc_freqs = metadata['doc_freqs'].squeeze()

    db = FeverDocDB("data/fever/fever.db")
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    jlr = JSONLineReader()

    with open(args.in_file,"r") as f, open("data/fever/{0}.sentences.{3}.p{1}.s{2}.jsonl".format(args.split, args.max_page, args.max_sent,"precomputed" if args.use_precomputed else "not_precomputed"), "w+") as out_file:
        lines = jlr.process(f)
        #lines = tf_idf_claims_batch(lines)

        for line in tqdm(lines):
            line = tf_idf_claim(line)
            out_file.write(json.dumps(line) + "\n")
