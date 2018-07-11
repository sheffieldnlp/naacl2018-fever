import logging
from rte.mithun.log import setup_custom_logger
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from nltk import word_tokenize
import logging
import re
import os
import numpy as np
import smtplib
import sys
from utils.process_input_data import tokenize
from utils.score import report_score
import  smtplib
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# from utils.fileWriter import writeToOutputFile
# from utils.fileWriter import appendToFile
import itertools
from utils.datastructures import indiv_headline_body
import scipy
from sklearn import svm
import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import itertools
from utils.file_functions import writeToOutputFile
from utils.process_input_data import cosine_sim_without_corpus_tfidf
from utils.process_input_data import create_corpus
from utils.process_input_data import cosine_sim_given_tfidf_vectors
from utils.process_input_data import get_cosine,normalize_dummy
from utils.process_input_data import create_corpus_articles


from utils.feature_engineering import refuting_features, polarity_features, hand_features,hedging_features
from utils.feature_engineering import word_overlap_features
from tqdm import tqdm
from utils.datastructures import indiv_headline_body
from utils.process_input_data import createAtfidfVectorizer
import itertools
from utils.process_input_data import doAllWordProcessing
import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import time
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import processors
from processors import ProcessorsBaseAPI
from processors import Document
import multiprocessing as mp
import json
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
my_out_dir = "poop-out"
n_cores = 2
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]
ann_head_test_ph2 = "ann_head_test_ph2.json"
ann_body_test_ph2 = "ann_body_test_ph2.json"
annotated_only_lemmas="ann_lemmas.json"
annotated_only_tags="ann_tags.json"
annotated_body_split_folder="split_body/"
annotated_head_split_folder="split_head/"
data_folder="/data/fever-data-ann/"


def read_json_feat_vec(load_ann_corpus_tr,gold_labels):

    related_matrix=[]
    un_related_matrix = []

    heads_lemmas_related=[]
    bodies_lemmas_related = []
    heads_tags_related = []
    bodies_tags_related = []


    total_pairs=0


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

    # vectorizer_only_wordoverlap = TfidfVectorizer(tokenizer=normalize_dummy, preprocessor=lambda x: x,
    #                                               stop_words='english')



    combined_vector = create_feature_vec(heads_lemmas, bodies_lemmas, heads_tags,
                                         bodies_tags)

    gold_int = joblib.load('gold_int_testing_phase2.pkl')

    joblib.dump(combined_vector, 'combined_vector_testing_phase2.pkl')

    logging.debug("done generating feature vectors. Going to call classifier")

    # give that vector to your svm for prediction.
    pred_class = svm_phase2.predict(combined_vector)
    joblib.dump(pred_class, 'pred_class.pkl')
    logging.debug("going to logging.debug pred_class")
    logging.debug("number of rows in pred_classis:" + str(pred_class.shape))

    # convert the predicted label to a regular list from numpy matrix

    pred_label_int = []
    value2_float = 2.0
    value1_float = 1.0
    value0_float = 0.0

    tuple_counter = 0

    # convert from float to int
    for x in (np.nditer(pred_class)):
        if (x == value2_float):
            pred_label_int.append(2)
        else:
            if (x == value1_float):
                pred_label_int.append(1)
            else:
                if (x == value0_float):
                    pred_label_int.append(0)

        tuple_counter = tuple_counter + 1

    logging.debug("total number of items in pred_label_int is:" + str(len(pred_label_int)))

    logging.debug("going to find number of rows in gold_int:")
    numrows = len(gold_int)
    logging.debug("number of rows in gold_int:" + str(numrows))

    return gold_int, pred_label_int


def  phase2_training_svm(tr_data, load_annotated_corpus, load_gold_labels_from_disk):
    logger = setup_custom_logger('root')
    logging.debug("inside phase2_training_svm")
    entire_corpus=[]
    labels = np.array([[]])

    word_overlap_vector = np.empty((0, 1), float)
    hedging_words_vector = np.empty((0, 30), int)
    refuting_value_matrix= np.empty((0, 16), int)
    noun_overlap_vector = np.empty((0, 2), int)

    combined_vector=None

    # Load the annotated corpus whcih was stored to disk in some of the previous runs.
    # i.e if you want to add a new feature to an existing data set you wont have to annotate the
    # entire data again.

    logging.debug("value of load_annotated_corpus:"+str(load_annotated_corpus))


    if(load_annotated_corpus== True):

        logging.debug("load_annotated_corpus== True")
        headlines_loaded = joblib.load('doc_headline_list.pkl')
        bodies_loaded = joblib.load('doc_body_list.pkl')

        for headline_loaded,body_loaded  in tqdm((zip(headlines_loaded,bodies_loaded)), total=len(tr_data), desc="annotated_data:"):

            #gold_stance = obj_indiv_headline_body.gold_stance


            lemmatized_headline=headline_loaded.lemmas
            lemmatized_body=body_loaded.lemmas
            tagged_headline=headline_loaded.tags
            tagged_body=body_loaded.tags


            word_overlap_array, hedge_value_array, refuting_value_array, noun_overlap_array=add_vectors(
                        lemmatized_headline, lemmatized_body, tagged_headline, tagged_body)

            word_overlap_vector = np.vstack([word_overlap_vector, word_overlap_array])
            hedging_words_vector = np.vstack([hedging_words_vector, hedge_value_array])
            refuting_value_matrix = np.vstack([refuting_value_matrix, refuting_value_array])
            noun_overlap_vector = np.vstack([noun_overlap_vector, noun_overlap_array])



            logging.debug("actual   word_overlap_vector is:" + str(word_overlap_vector))

            logging.debug("shape of  word_overlap_vector is:" + str(word_overlap_vector.shape))



        logging.info("number of rows in label list is is:" + str(len(labels)))
        logging.info("going to feed this vectorized tf to a classifier:")

        logging.info("shape of  word_overlap_vector is:" + str(word_overlap_vector.shape))

        logging.info("shape of  hedging_words_vector is:" + str(hedging_words_vector.shape))

        combined_vector = np.hstack([word_overlap_vector, hedging_words_vector, refuting_value_matrix, noun_overlap_vector])

        #todo: keep all these file names at one place
        joblib.dump(combined_vector, 'combined_vector_tr_phase2.pkl')

        if (load_gold_labels_from_disk):
            labels = joblib.load('labels_tr_phase2.pkl')




    logging.debug("shape of combined_vector is:" + str(combined_vector.shape))

    start_time = time.time()
    elapsed_time = time.time() - start_time
    logging.debug("writeToOutputFile time taken:" + str(elapsed_time))

    logging.debug(str(labels))
    logging.debug("shape of labels is:" + str(labels.shape))


    #feed the vectors to an an svm, with labels.
    start_time = time.time()
    clf = svm.SVC(kernel='linear', C=1.0)
    #feature_vector=feature_vector.reshape(-1, 1)
    clf.fit(combined_vector, labels.ravel())


    elapsed_time = time.time() - start_time
    logging.debug("svm time taken:" + str(elapsed_time))

    logging.debug("done training svm:" )

    return clf,vectorizer_phase2,combined_vector


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
