import argparse
import json
import spacy
import wmd

from common.dataset.reader import JSONLineReader
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB

from tqdm import tqdm

nlp = spacy.load("en", create_pipeline=wmd.WMD.create_spacy_pipeline)


def wmd_sim(claim,lines):
    cl = nlp(claim)
    scores = []
    for line in lines:
        scores.append(cl.similarity(nlp(line)))
    return scores


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

    jlr = JSONLineReader()
    with open(args.in_file,"r") as f:
        lines = jlr.process(f)

    db = FeverDocDB(args.db)

    with open(args.out_file,"w+") as f:
        for line in tqdm(lines):
            if 'predicted_pages' in line:
                sorted_p = list(sorted(line['predicted_pages'],reverse=True, key=lambda elem:elem[1]))

                pages = [p[0] for p in sorted_p[:args.max_page]]
                p_lines = []
                for page in pages:
                    lines = db.get_doc_lines(page)
                    lines = [line.split("\t")[1] if len(line.split("\t")[1])>1 else "" for line in lines.split("\n")]

                    p_lines.extend(zip(lines,[page]*len(lines),range(len(lines))))

                scores = wmd_sim(line["claim"],[pl[0] for pl in p_lines])
                scores = list(zip(scores,[pl[1] for pl in p_lines],[pl[2] for pl in p_lines],[pl[0] for pl in p_lines]))
                scores = list(filter(lambda score:len(score[3].strip()),scores))
                sentences_l = list(sorted(scores,reverse=True,key=lambda elem:elem[0]))

                line["predicted_sentences"] = [(s[1],s[2]) for s in sentences_l[:args.max_sent]]
                f.write(json.dumps(line)+"\n")