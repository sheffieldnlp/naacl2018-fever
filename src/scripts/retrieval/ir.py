import argparse
import json
from multiprocessing.pool import ThreadPool
import tqdm
import os,sys

from common.util.log_helper import LogHelper
from tqdm import tqdm

from retrieval.top_n import TopNDocsTopNSents
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from rte.mithun.ds import indiv_headline_body

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
        all_claims = jlr.process(f)
        #lines now contains all list of claims
        logger.info("first line is:")
        logger.info(all_claims[0])

        obj_all_heads_bodies=[]
        for claim in tqdm(all_claims,total=len(all_claims),desc="get_claim_ev:"):
            x = indiv_headline_body()
            evidences=claim["evidence"]
            ev_claim=[]
            for evidence in evidences[0]:
                t=evidence[2]
                l=evidence[3]
                sent=method.get_sentences_given_claim(t,logger,l)
                ev_claim.append(sent)
            str_ev_claim=' '.join(ev_claim)
            x.headline=claim
            x.body=str_ev_claim
            obj_all_heads_bodies.append(x)
        logger.info("length of claims is:" + str(len(all_claims)))
        logger.info("length of obj_all_heads_bodies is:" + str(len(obj_all_heads_bodies)))
        sys.exit(1)
        counter=0

        with ThreadPool() as p:
            for line in tqdm(get_map_function(args.parallel)(lambda line: process_line(method,line), all_claims), total=len(all_claims)):
                #out_file.write(json.dumps(line) + "\n")
                processed[line["id"]] = line
                logger.info("processed line is:"+str(line))
                counter=counter+1
                if(counter==10):
                    sys.exit(1)

        logger.info("Done, writing to disk")

        for line in all_claims:
            out_file.write(json.dumps(processed[line["id"]]) + "\n")