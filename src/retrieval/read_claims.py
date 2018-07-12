from common.util.log_helper import LogHelper
from rte.mithun.ds import indiv_headline_body
from processors import ProcessorsBaseAPI
from tqdm import tqdm
from processors import Document
import logging
from rte.mithun.trainer import read_json_create_feat_vec,do_training,do_testing,load_model
import numpy as np
import os,sys

ann_head_tr = "ann_head_tr.json"
ann_body_tr = "ann_body_tr.json"
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
logger=None
load_ann_corpus=True
load_combined_vector=False
load_model=True

def read_claims_annotate(args,jlr,logger,method):
    try:
        os.remove(ann_head_tr)
        os.remove(ann_body_tr)

    except OSError:
        logger.error("not able to find file")

    logger.debug("inside read_claims_annotate")
    with open(args.in_file,"r") as f, open(args.out_file, "w+") as out_file:
        all_claims = jlr.process(f)
        obj_all_heads_bodies=[]
        ver_count=0
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_claim_ev:"):
            logger.debug("entire claim_full is:")
            logger.debug(claim_full)
            claim=claim_full["claim"]
            logger.debug("just claim alone is:")
            logger.debug(claim)
            x = indiv_headline_body()
            evidences=claim_full["evidence"]
            label=claim_full["label"]
            if not (label=="NOT ENOUGH INFO"):
                ver_count=ver_count+1
                logger.debug("length of evidences for this claim_full  is:" + str(len(evidences)))
                logger.debug("length of evidences for this claim_full  is:" + str(len(evidences[0])))
                ev_claim=[]
                for evidence in evidences[0]:
                    t=evidence[2]
                    l=evidence[3]
                    logger.debug(t)
                    logger.debug(l)
                    sent=method.get_sentences_given_claim(t,logger,l)
                    ev_claim.append(sent)
                all_evidences=' '.join(ev_claim)
                annotate_and_save_doc(claim, all_evidences,index, API, ann_head_tr, ann_body_tr, logger)

        return obj_all_heads_bodies

def uofa_training(args,jlr,method,logger):
    logger.debug("got inside uofa_training")

    #this code annotates the given file using pyprocessors. Run it only once in its lifetime.
    # tr_data=read_claims_annotate(args,jlr,logger,method)
    # logger.info(
    #     "Finished writing json to disk . going to quit. names of the files are:" + ann_head_tr + ";" + ann_body_tr)

    gold_labels_tr = get_gold_labels(args, jlr)
    logging.info("number of rows in label list is is:" + str(len(gold_labels_tr)))
    combined_vector = read_json_create_feat_vec(load_ann_corpus, gold_labels_tr, logger, load_combined_vector)
    do_training(combined_vector, gold_labels_tr)

    sys.exit(1)

def uofa_testing(args,jlr,method,logger):
    logger.debug("got inside uofa_testing")
    gold_labels_tr = get_gold_labels(args, jlr)
    logging.info("number of rows in label list is is:" + str(len(gold_labels_tr)))
    combined_vector= read_json_create_feat_vec(load_ann_corpus, logger, load_combined_vector)
    trained_model=load_model()
    do_testing(combined_vector,gold_labels_tr,trained_model)
    logging.info("done testing. going to quit")
    sys.exit(1)

def annotate_save_quit(test_data,logger):


    for i, d in tqdm(enumerate(test_data), total=len(test_data),desc="annotate_json:"):
        annotate_and_save_doc(d, i, API, ann_head_tr, ann_body_tr,logger)




    sys.exit(1)



def annotate_and_save_doc(headline,body, index, API, json_file_tr_annotated_headline,json_file_tr_annotated_body,
                          logger):
    logger.debug("got inside annotate_and_save_doc")
    logger.debug("headline:"+headline)
    logger.debug("body:" + body)
    doc1 = API.fastnlp.annotate(headline)
    doc1.id=index
    with open(json_file_tr_annotated_headline, "a") as out:
      out.write(doc1.to_JSON())
      out.write("\n")


    doc2 = API.fastnlp.annotate(body)
    logger.debug(doc2)
    doc2.id = index

    with open(json_file_tr_annotated_body, "a") as out:
          out.write(doc2.to_JSON())
          out.write("\n")

    return

def get_gold_labels(args,jlr):
    labels = np.array([[]])

    with open(args.in_file,"r") as f, open(args.out_file, "w+") as out_file:
        all_claims = jlr.process(f)
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels:"):
            label=claim_full["label"]
            if (label == "SUPPORTS"):
                labels = np.append(labels, 0)
            else:
                if (label == "REFUTES"):
                    labels = np.append(labels, 1)
                # else:
                #     if (label=="NOT ENOUGH INFO"):
                #         labels = np.append(labels, 2)

    return labels
