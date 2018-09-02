from common.util.log_helper import LogHelper
from rte.mithun.ds import indiv_headline_body
from processors import ProcessorsBaseAPI
from tqdm import tqdm
from processors import Document
import logging
from rte.mithun.trainer import read_json_create_feat_vec,do_training,do_testing,load_model,print_missed
import numpy as np
import os,sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ann_head_tr = "ann_head_tr.json"
ann_body_tr = "ann_body_tr.json"
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
logger=None
load_ann_corpus=True
#load_combined_vector=True
from scorer.src.fever.scorer import fever_score
import json
from sklearn.externals import joblib

predicted_results="predicted_results.pkl"

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


def print_cv(combined_vector,gold_labels_tr):
    logging.debug(gold_labels_tr.shape)
    logging.debug(combined_vector.shape)
    x= np.column_stack([gold_labels_tr,combined_vector])
    np.savetxt("cv.csv", x, delimiter=",")
    sys.exit(1)


def uofa_training(args,jlr,method,logger):
    logger.warning("got inside uofatraining")

    #this code annotates the given file using pyprocessors. Run it only once in its lifetime.
    #tr_data=read_claims_annotate(args,jlr,logger,method)
    # logger.info(
    #     "Finished writing json to disk . going to quit. names of the files are:" + ann_head_tr + ";" + ann_body_tr)

    gold_labels_tr =None
    if(args.mode =="small"):
        gold_labels_tr =get_gold_labels_small(args, jlr)
    else:
        gold_labels_tr = get_gold_labels(args, jlr)

    logging.info("number of rows in label list is is:" + str(len(gold_labels_tr)))
    combined_vector = read_json_create_feat_vec(load_ann_corpus,args)

    logging.warning("done with generating feature vectors. Model training next")
    logging.info("gold_labels_tr is:" + str(len(gold_labels_tr)))
    logging.info("shape of cv:" + str(combined_vector.shape))
    logging.info("above two must match")

    do_training(combined_vector, gold_labels_tr)

    logging.warning("done with training. going to exit")
    sys.exit(1)



def uofa_testing(args,jlr,method,logger):


    logger.warning("got inside uofa_testing")
    gold_labels = get_gold_labels(args, jlr)
    label_ev=get_gold_labels_evidence(args, jlr)




    combined_vector= read_json_create_feat_vec(load_ann_corpus,args)
    #print_cv(combined_vector, gold_labels)
    logging.info("done with generating feature vectors. Model loading and predicting next")
    logging.info("shape of cv:"+str(combined_vector.shape))
    logging.info("number of rows in label list is is:" + str(len(gold_labels)))
    logging.info("above two must match")
    assert(combined_vector.shape[0]==len(gold_labels))
    trained_model=load_model()

    pred=do_testing(combined_vector,trained_model)



    logging.debug(str(pred))
    logging.debug("and golden labels are:")
    logging.debug(str(gold_labels))
    logging.warning("done testing. and the accuracy is:")
    acc=accuracy_score(gold_labels, pred)*100
    logging.warning(str(acc)+"%")
    logging.info(classification_report(gold_labels, pred))
    logging.info(confusion_matrix(gold_labels, pred))

    

    # get number of support vectors for each class
    #logging.debug(trained_model.n_support_)
    logging.info("done with testing. going to exit")
    final_predictions=write_pred_str_disk(args,jlr,pred)
    fever_score(final_predictions,label_ev)
    sys.exit(1)

def annotate_save_quit(test_data,logger):

    for i, d in tqdm(enumerate(test_data), total=len(test_data),desc="annotate_json:"):
        annotate_and_save_doc(d, i, API, ann_head_tr, ann_body_tr,logger)


    sys.exit(1)


#load predictions, convert it based on label and write it as string.
def write_pred_str_disk(args,jlr,pred):
    logging.debug("here1"+str(args.out_file))
    final_predictions=[]
    #pred=joblib.load(predicted_results)
    with open(args.in_file,"r") as f:
        ir = jlr.process(f)
        logging.debug("here2"+str(len(ir)))

        for index,(p,q) in enumerate(zip(pred,ir)):
            line=dict()
            label="not enough info"
            if(p==0):
                label="supports"
            else:
                if(p==1):
                    label="refutes"

            line["id"]=q["id"]
            line["predicted_label"]=label
            line["predicted_evidence"]=q["predicted_sentences"]
            logging.debug(q["id"])
            logging.debug(label)
            logging.debug(q["predicted_sentences"])
            logging.debug(index)

            final_predictions.append(line)

    logging.info(len(final_predictions))

    with open(args.pred_file, "w+") as out_file:
        for x in final_predictions:
            out_file.write(json.dumps(x)+"\n")
    return final_predictions


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

def get_gold_labels_evidence(args,jlr):
    evidences=[]
    with open(args.in_file,"r") as f:
        all_claims = jlr.process(f)
        gold=dict()
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels_ev:"):
            label=claim_full["label"]
            if not (label.lower()=="NOT ENOUGH INFO"):
                gold["label"]=label
                gold["evidence"]=claim_full["evidence"]
                evidences.append(gold)

    return evidences

def get_claim_evidence_sans_NEI(args,jlr):
    claims=[]
    evidences=[]

    with open(args.in_file,"r") as f:
        all_claims = jlr.process(f)
        gold=dict()
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels_ev:"):
            label=claim_full["label"]
            if not (label.lower()=="NOT ENOUGH INFO"):
                gold["label"]=label
                gold["evidence"]=claim_full["evidence"]
                evidences.append(gold)
                claims.append(claim_full)

    return claims,evidences

def get_gold_labels_small(args,jlr):
    labels = np.array([[]])

    counter=0
    with open(args.in_file,"r") as f, open(args.out_file, "w+") as out_file:
        all_claims = jlr.process(f)
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels:"):
            counter+=1
            label=claim_full["label"]
            if (label == "SUPPORTS"):
                labels = np.append(labels, 0)
            else:
                if (label == "REFUTES"):
                    labels = np.append(labels, 1)
                else:
                    if (label=="NOT ENOUGH INFO"):
                        labels = np.append(labels, 2)
            logging.debug(index)
            if (counter==10):
                return labels
    return labels


def uofa_dev(args, jlr, method, logger):


    gold_labels = get_gold_labels(args, jlr)
    logging.warning("got inside uofa_dev")
    #print_missed(args,gold_labels)
    #logging.warning("done printing")
    #sys.exit(1)

    combined_vector= read_json_create_feat_vec(load_ann_corpus,args)
    #print_cv(combined_vector, gold_labels)
    logging.info("done with generating feature vectors. Model loading and predicting next")
    logging.info("shape of cv:"+str(combined_vector.shape))
    logging.info("number of rows in label list is is:" + str(len(gold_labels)))
    logging.info("above two must match")
    trained_model=load_model()
    # logging.warning("weights:")
    # logging.warning(trained_model.coef_)
    pred=do_testing(combined_vector,trained_model)
    logging.debug(str(pred))
    logging.debug("and golden labels are:")
    logging.debug(str(gold_labels))
    logging.warning("done testing. and the accuracy is:")
    acc=accuracy_score(gold_labels, pred)*100
    logging.warning(str(acc)+"%")
    logging.info(classification_report(gold_labels, pred))
    logging.info(confusion_matrix(gold_labels, pred))



    # get number of support vectors for each class
    #logging.debug(trained_model.n_support_)
    logging.info("done with testing. going to exit")
    sys.exit(1)

