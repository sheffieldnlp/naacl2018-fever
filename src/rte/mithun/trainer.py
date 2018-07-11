from __future__ import division
from rte.mithun.log import setup_custom_logger
import logging
import sys
from sklearn import svm
import tqdm
import os
import numpy as np
from tqdm import tqdm
import time
from sklearn.externals import joblib
from processors import ProcessorsBaseAPI
from processors import Document
import json
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
my_out_dir = "poop-out"
n_cores = 2
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
RELATED = LABELS[0:3]
annotated_only_lemmas="ann_lemmas.json"
annotated_only_tags="ann_tags.json"
annotated_body_split_folder="split_body/"
annotated_head_split_folder="split_head/"
data_folder="/data/fever-data-ann/"


def read_json_feat_vec(load_ann_corpus_tr):


    logging.debug("value of load_ann_corpus_tph2:" + str(load_ann_corpus_tr))

    cwd=os.getcwd()
    bf=cwd+data_folder+annotated_body_split_folder
    bff=bf+annotated_only_lemmas
    bft=bf+annotated_only_tags

    hf=cwd+data_folder+annotated_head_split_folder
    hff=hf+annotated_only_lemmas
    hft=hf+annotated_only_tags


    logging.debug("hff:" + str(hff))
    logging.debug("bff:" + str(bff))
    logging.info("going to read heads_lemmas from disk:")

    heads_lemmas = read_json(hff)
    bodies_lemmas = read_json(bff)
    heads_tags = read_json(hft)
    bodies_tags = read_json(bft)


    logging.debug("size of heads_lemmas is: " + str(len(heads_lemmas)))
    logging.debug("size of bodies_lemmas is: " + str(len(bodies_lemmas)))
    logging.debug("size of data is: " + str(len(data)))


    if not (len(heads_lemmas) == len(bodies_lemmas)==len(data)):
        logging.debug("size of heads_lemmas and bodies_lemmas dont match")
        sys.exit(1)


    combined_vector = create_feature_vec(heads_lemmas, bodies_lemmas, heads_tags,
                                         bodies_tags)

    joblib.dump(combined_vector, 'combined_vector_testing_phase2.pkl')

    logging.debug("done generating feature vectors. Going to call classifier")

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(combined_vector, labels.ravel())

    joblib.dump(clf, 'model_trained_phase2.pkl')

    return;


def read_json(json_file):
    logging.debug("inside read_json_pyproc_doc")
    l = []
    counter=0

    with open(json_file) as f:
        for eachline in (f):
            d = json.loads(eachline)
            a=d["data"]
            just_lemmas=' '.join(str(r) for v in a for r in v)
            l.append(just_lemmas)
            logging.debug(counter)
            counter = counter + 1

    logging.debug("counter:"+str(counter))
    return l


def normalize_dummy(text):
    x = text.lower().translate(remove_punctuation_map)
    return x.split(" ")

def create_feature_vec(heads_lemmas,bodies_lemmas,heads_tags_related,bodies_tags_related):
    word_overlap_vector = np.empty((0, 1), float)
    hedging_words_vector = np.empty((0, 30), int)
    refuting_value_matrix = np.empty((0, 16), int)
    noun_overlap_vector = np.empty((0, 2), int)

    for head_lemmas, body_lemmas,head_tags_related,body_tags_related in tqdm(zip(heads_lemmas, bodies_lemmas,heads_tags_related,bodies_tags_related),
                           total=len(bodies_tags_related), desc="feat_gen:"):
        # h = Document.load_from_JSON(head)
        # b= Document.load_from_JSON(body)
        #logging.debug(h.lemmas)
        #logging.debug(b.lemmas)

        # lemmatized_headline=h.lemmas
        # lemmatized_body=b.lemmas
        # tagged_headline=h.tags
        # tagged_body=b.tags

        lemmatized_headline = head_lemmas
        lemmatized_body=body_lemmas
        tagged_headline=head_tags_related
        tagged_body=body_tags_related

        # logging.debug(lemmatized_headline)
        # logging.debug(lemmatized_body)
        # logging.debug(tagged_headline)
        # logging.debug(tagged_body)



        word_overlap_array, hedge_value_array, refuting_value_array, noun_overlap_array = add_vectors(
            lemmatized_headline, lemmatized_body, tagged_headline, tagged_body)

        word_overlap_vector = np.vstack([word_overlap_vector, word_overlap_array])
        hedging_words_vector = np.vstack([hedging_words_vector, hedge_value_array])
        refuting_value_matrix = np.vstack([refuting_value_matrix, refuting_value_array])
        noun_overlap_vector = np.vstack([noun_overlap_vector, noun_overlap_array])




    logging.debug("\ndone with all headline body.:")
    logging.debug("shape of  word_overlap_vector is:" + str(word_overlap_vector.shape))
    logging.debug("refuting_value_matrix.dtype=" + str(refuting_value_matrix.dtype))
    logging.debug("refuting_value_matrix is =" + str(refuting_value_matrix))

    combined_vector = np.hstack(
        [word_overlap_vector, hedging_words_vector, refuting_value_matrix, noun_overlap_vector])

    return combined_vector
