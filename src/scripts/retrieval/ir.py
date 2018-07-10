import argparse
import json
from multiprocessing.pool import ThreadPool

import os,sys

from common.util.log_helper import LogHelper
from tqdm import tqdm

from retrieval.top_n import TopNDocsTopNSents
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema


def process_line(method,line):
    sents = method.get_sentences_for_claim(line["claim"])
    pages = list(set(map(lambda sent:sent[0],sents)))
    line["predicted_pages"] = pages
    line["predicted_sentences"] = sents
    return line


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_map_function(parallel):
    return p.imap_unordered if parallel else map

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help='drqa doc db file')
    parser.add_argument('--model', type=str, help='drqa index file')
    parser.add_argument('--in-file', type=str, help='input dataset')
    parser.add_argument('--out-file', type=str, help='path to save output dataset')
    parser.add_argument('--max-page',type=int)
    parser.add_argument('--max-sent',type=int)
    parser.add_argument('--parallel',type=str2bool,default=True)
    args = parser.parse_args()

    db = FeverDocDB(args.db)
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    method = TopNDocsTopNSents(db, args.max_page, args.max_sent, args.model)


    processed = dict()

    with open(args.in_file,"r") as f, open(args.out_file, "w+") as out_file:
        lines = jlr.process(f)
        #lines now contains all list of claims
        logger.info("first line is:")
        logger.info(lines[0])

        for claim in lines:
            evidences=claim["evidence"]
            evidences_all_sent=[]
            for evidence in evidences[0]:
                t=evidence[2]
                l=evidence[3]
                sent=method.get_sentences_given_claim(t,logger,l)
                evidences_all_sent.append(sent)

            logger.info("evidence sentences for the first claim are:")
            logger.info(evidences_all_sent)
            sys.exit(1)


        sys.exit(1)
        counter=0

        with ThreadPool() as p:
            for line in tqdm(get_map_function(args.parallel)(lambda line: process_line(method,line),lines), total=len(lines)):
                #out_file.write(json.dumps(line) + "\n")
                processed[line["id"]] = line
                logger.info("processed line is:"+str(line))
                counter=counter+1
                if(counter==10):
                    sys.exit(1)

        logger.info("Done, writing to disk")

        for line in lines:
            out_file.write(json.dumps(processed[line["id"]]) + "\n")