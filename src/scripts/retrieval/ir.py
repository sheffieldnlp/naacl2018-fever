import argparse
import json
from multiprocessing.pool import ThreadPool
import tqdm
import os,sys
import logging
from common.util.log_helper import LogHelper
from tqdm import tqdm

from retrieval.top_n import TopNDocsTopNSents
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from retrieval.read_claims import uofa_training,uofa_testing,uofa_dev
from rte.mithun.log import setup_custom_logger




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
    #setup_custom_logger
    # LogHelper.setup()
    # logger = LogHelper.get_logger(__name__)
    logger = setup_custom_logger('root',args)
    logger.debug('main message')

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help='drqa doc db file')
    parser.add_argument('--model', type=str, help='drqa index file')
    parser.add_argument('--in-file', type=str, help='input dataset')
    parser.add_argument('--out-file', type=str, help='path to save output dataset')
    parser.add_argument('--max-page',type=int)
    parser.add_argument('--max-sent',type=int)
    parser.add_argument('--parallel',type=str2bool,default=True)
    parser.add_argument('--mode', type=str, help='do training or testing' )
    parser.add_argument('--load_feat_vec', type=str2bool,default=False)
    parser.add_argument('--pred_file', type=str, help='path to save predictions',default="predictions.jsonl")
    parser.add_argument('--dynamic_cv',type=str2bool,default=False)
    parser.add_argument('--lmode', type=str,default="DEBUG")


    args = parser.parse_args()

    db = FeverDocDB(args.db)
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    method = TopNDocsTopNSents(db, args.max_page, args.max_sent, args.model)


    processed = dict()

    if(args.mode=="train" or args.mode=="small"):
        uofa_training(args,jlr,method,logger)
    else:
        if(args.mode=="dev"):
            uofa_dev(args,jlr,method,logger)
            logger.info("Done, testing ")

        else:
            if(args.mode=="test" ):
                uofa_testing(args,jlr,method,logger)
                logger.info("Done, testing ")




    with ThreadPool() as p:
        for line in tqdm(get_map_function(args.parallel)(lambda line: process_line(method,line), all_claims), total=len(all_claims)):
            processed[line["id"]] = line
            logger.info("processed line is:"+str(line))
            counter=counter+1
            if(counter==10):
                sys.exit(1)

    logger.info("Done, writing to disk")

    for line in all_claims:
        out_file.write(json.dumps(processed[line["id"]]) + "\n")
