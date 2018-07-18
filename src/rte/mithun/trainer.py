from __future__ import division
import sys,logging
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
#data_root="/work/mithunpaul/fever/my_fork/fever-baselines/data"
data_root=""
data_folder_train=data_root+"/fever-data-ann/train/"
data_folder_dev=data_root+"/data/fever-data-ann/dev/"
model_trained="model_trained.pkl"

predicted_results="predicted_results.pkl"
combined_vector_training="combined_vector_testing_phase2.pkl"

def read_json_create_feat_vec(load_ann_corpus_tr,args):

    if (args.load_feat_vec==True):

        logging.info("going to load combined vector from disk")
        combined_vector = joblib.load(combined_vector_training)

    else:
        logging.debug("load_feat_vec is falsse. going to generate features")
        logging.debug("value of load_ann_corpus_tph2:" + str(load_ann_corpus_tr))

        cwd=os.getcwd()
        data_folder=None
        if(args.mode=="test"):
            data_folder=data_folder_dev
        else:
            if(args.mode=="train"):
                data_folder=data_folder_train

        bf=cwd+data_folder+annotated_body_split_folder
        bff=bf+annotated_only_lemmas
        bft=bf+annotated_only_tags

        hf=cwd+data_folder+annotated_head_split_folder
        hff=hf+annotated_only_lemmas
        hft=hf+annotated_only_tags


        logging.debug("hff:" + str(hff))
        logging.debug("bff:" + str(bff))
        logging.info("going to read heads_lemmas from disk:")

        heads_lemmas = read_json(hff,logging)
        bodies_lemmas = read_json(bff,logging)
        heads_tags = read_json(hft,logging)
        bodies_tags = read_json(bft,logging)


        logging.debug("size of heads_lemmas is: " + str(len(heads_lemmas)))
        logging.debug("size of bodies_lemmas is: " + str(len(bodies_lemmas)))


        if not (len(heads_lemmas) == len(bodies_lemmas)):
            logging.debug("size of heads_lemmas and bodies_lemmas dont match")
            sys.exit(1)


        combined_vector = create_feature_vec(heads_lemmas, bodies_lemmas, heads_tags,
                                             bodies_tags,logging)

        joblib.dump(combined_vector, combined_vector_training)
        logging.info("done generating feature vectors.")


    return combined_vector;

def print_nonzero_cv(combined_vector):
    # debug code: go through all the vectors last row and print the coordinates of non zero entries


    c=0

    logging.debug(" starting: combined vector"+str(combined_vector))
    while(c<20):
        ns = np.nonzero(combined_vector[c])
        logging.debug(ns)
        # for x in ns:
        #
        #     for y in x:
        #             if(y not in(0,50,51)):
        #                 logging.debug(x)
        c = c + 1

    sys.exit(1)



def do_training(combined_vector,gold_labels_tr,args):
    cvalue=float(args.svmc)
    k=args.kernel
    logging.debug("going to load the classifier:")
    clf = svm.SVC(kernel=k, C=cvalue)
    clf.fit(combined_vector, gold_labels_tr.ravel())
    #todo:print the weights.
    file=model_trained+"_"+str(cvalue)+"_"+str(k)+".pkl"
    joblib.dump(clf, file)
    logging.debug("done saving model to disk")

def load_model():
    model=joblib.load(model_trained)
    return model;

def do_testing(combined_vector,svm):
    logging.info("all value of combined_vector is:"+str(combined_vector))
    logging.info("going to predict...")
    p=svm.predict(combined_vector)
    joblib.dump(p, predicted_results)
    logging.debug("done with predictions")
    return p

def read_json(json_file,logging):
    logging.debug("inside read_json")
    l = []
    counter=0

    with open(json_file) as f:
        for eachline in (f):
            d = json.loads(eachline)
            a=d["data"]
            just_lemmas=' '.join(str(r) for v in a for r in v)
            l.append(just_lemmas)
            counter = counter + 1
    return l


def normalize_dummy(text):
    x = text.lower().translate(remove_punctuation_map)
    return x.split(" ")

def create_feature_vec(heads_lemmas,bodies_lemmas,heads_tags_related,bodies_tags_related,logging):
    #todo: dont hardcode. create this after u know the size
    word_overlap_vector = np.empty((0, 1), float)
    hedging_words_vector = np.empty((0, 30), int)
    refuting_value_matrix = np.empty((0, 19), int)
    noun_overlap_matrix = np.empty((0, 2), int)
    vb_overlap_matrix = np.empty((0, 2), int)

    counter=0

    for  head_lemmas, body_lemmas,head_tags_related,body_tags_related in tqdm((zip(heads_lemmas, bodies_lemmas,heads_tags_related,bodies_tags_related)),
                           total=len(bodies_tags_related), desc="feat_gen:"):

        lemmatized_headline = head_lemmas
        lemmatized_body=body_lemmas
        tagged_headline=head_tags_related
        tagged_body=body_tags_related


        #todo: remove stop words-bring in nltk list of stop words...and punctuation.

        word_overlap_array, hedge_value_array, refuting_value_array, noun_overlap_array, verb_overlap_array = add_vectors(
            lemmatized_headline, lemmatized_body, tagged_headline, tagged_body,logging)

        logging.info("inside create_feature_vec. just received verb_overlap_array is =" + str(verb_overlap_array))
        logging.info("inside create_feature_vec. vb_overlap_matrix is =" + str(vb_overlap_matrix))

        word_overlap_vector = np.vstack([word_overlap_vector, word_overlap_array])
        hedging_words_vector = np.vstack([hedging_words_vector, hedge_value_array])
        refuting_value_matrix = np.vstack([refuting_value_matrix, refuting_value_array])
        noun_overlap_matrix = np.vstack([noun_overlap_matrix, noun_overlap_array])
        vb_overlap_matrix== np.vstack([vb_overlap_matrix, verb_overlap_array])

        logging.info("  word_overlap_vector is:" + str(word_overlap_vector))
        logging.info("refuting_value_matrix" + str(refuting_value_matrix))
        logging.info("noun_overlap_matrix is =" + str(noun_overlap_matrix))
        logging.info("shape  noun_overlap_matrix is:" + str(noun_overlap_matrix.shape))
        logging.info("vb_overlap_matrix is =" + str(vb_overlap_matrix))
        logging.info("shape  vb_overlap_matrix is:" + str(vb_overlap_matrix.shape))


        counter = counter + 1




    logging.info("\ndone with all headline body.:")
    logging.info("shape of  word_overlap_vector is:" + str(word_overlap_vector.shape))
    logging.info("shape of  hedging_words_vector is:" + str(hedging_words_vector.shape))
    logging.info("shape of  refuting_value_matrix is:" + str(refuting_value_matrix.shape))
    logging.info("shape of  noun_overlap_matrix is:" + str(noun_overlap_matrix.shape))
    logging.info("shape of  vb_overlap_matrix is:" + str(vb_overlap_matrix.shape))

    combined_vector= np.hstack(
        [word_overlap_vector, hedging_words_vector, refuting_value_matrix, noun_overlap_matrix,vb_overlap_matrix])



    return combined_vector


def add_vectors(lemmatized_headline,lemmatized_body,tagged_headline,tagged_body,logging):


    #split everywhere based on space-i.e for word overlap etc etc..

    lemmatized_headline = lemmatized_headline.lower()
    lemmatized_body = lemmatized_body.lower()

    lemmatized_headline_split = lemmatized_headline.split(" ")
    headline_pos_split = tagged_headline.split(" ")
    lemmatized_body_split = lemmatized_body.split(" ")
    body_pos_split = tagged_body.split(" ")

    word_overlap = word_overlap_features_mithun(lemmatized_headline_split, lemmatized_body_split)
    word_overlap_array = np.array([word_overlap])

    hedge_value = hedging_features(lemmatized_headline_split, lemmatized_body_split)
    hedge_value_array = np.array([hedge_value])

    refuting_value = refuting_features_mithun(lemmatized_headline_split, lemmatized_body_split)
    refuting_value_array = np.array([refuting_value])

    noun_overlap = pos_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split, body_pos_split, "NN")
    noun_overlap_array = np.array([noun_overlap])

    vb_overlap = pos_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                        body_pos_split, "VB")
    logging.info("just after receiving vb_overlap in add_vectors. value of vb_overlap is:"+str(vb_overlap))
    vb_overlap_array = np.array([vb_overlap])
    logging.info("just after receiving vb_overlap in add_vectors. value of vb_overlap_array is:" + str(vb_overlap_array))




    return word_overlap_array,hedge_value_array,refuting_value_array,noun_overlap_array,vb_overlap_array


def word_overlap_features_mithun(clean_headline, clean_body):
    # todo: try adding word overlap features direction based, like noun overlap...i.e have 3 overall..one this, and 2 others.

    features = [
        len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]

    return features

def hedging_features(clean_headline, clean_body):

    #todo: do hedging features for headline. Have one for headline and one for body...note : have as separate vectors

    hedging_words = [
        'allegedly',
        'reportedly',
      'argue',
      'argument',
      'believe',
      'belief',
      'conjecture',
      'consider',
      'hint',
      'hypothesis',
      'hypotheses',
      'hypothesize',
      'implication',
      'imply',
      'indicate',
      'predict',
      'prediction',
      'previous',
      'previously',
      'proposal',
      'propose',
      'question',
      'speculate',
      'speculation',
      'suggest',
      'suspect',
      'theorize',
      'theory',
      'think',
      'whether'
    ]

    length_hedge=len(hedging_words)
    hedging_body_vector = [0] * length_hedge



    for word in clean_body:
        if word in hedging_words:
            index=hedging_words.index(word)
            hedging_body_vector[index]=1


    return hedging_body_vector

def refuting_features_mithun(clean_headline, clean_body):
    # todo: do hedging features for headline. Have one for headline and one for body...note : have as separate vectors

    refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny',
        'denies',
        'refute',
        'no',
        'neither',
        'nor',
        'not',
        'despite',
        'nope',
        'doubt',
        'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract',

    ]

    # todo: make sure nltk doesn't remove not as a stop word
    # todo: check the lamm form for 'n't and add it
    length_hedge=len(refuting_words)
    refuting_body_vector = [0] * length_hedge

    for word in clean_body:
        if word in refuting_words:
            index=refuting_words.index(word)
            refuting_body_vector[index]=1



    return refuting_body_vector

def pos_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split, body_pos_split, pos_in):
    # todo1: try adding just a simple plain noun overlap features ...not direction based, like noun overlap...i.e have 3 overall..one this, and 2 others.
    #todo:2: refer to excel sheet todo. add chunks. i.e entire one chunk and check how much of it overlaps.

        logging.info("inside " + pos_in + " overlap")
        h_nouns = []
        b_nouns = []

        noun_count_headline = 0
        for word, pos in zip(lemmatized_headline_split, headline_pos_split):
            logging.debug(str("pos:") + ";" + str((pos)))
            logging.debug(str("word:") + ";" + str((word)))

            if pos.startswith(pos_in):
                logging.debug("pos.startswith:"+str(pos_in))
                noun_count_headline = noun_count_headline + 1
                h_nouns.append(word)

        noun_count_body = 0
        for word, pos in zip(lemmatized_body_split, body_pos_split):
            if pos.startswith(pos_in):
                noun_count_body = noun_count_body + 1
                b_nouns.append(word)

        overlap = set(h_nouns).intersection(set(b_nouns))

        overlap_noun_counter = len(overlap)

        features = [0, 0]


        logging.info(str("h_nouns:") + ";" + str((h_nouns)))
        logging.info(str("b_nouns:") + ";" + str((b_nouns)))
        logging.info(str("overlap_noun_counter:") + ";" + str((overlap_noun_counter)))
        logging.info(str("overlap:") + ";" + str((overlap)))


        logging.debug(str("noun_count_body:") + ";" + str((noun_count_body)))
        logging.debug(str("noun_count_headline:") + ";" + str((noun_count_headline)))


        if (noun_count_body > 0 and noun_count_headline > 0):
            ratio_pos_dir1 = overlap_noun_counter / (noun_count_body)
            ratio_pos_dir2 = overlap_noun_counter / (noun_count_headline)

            if not ((ratio_pos_dir1==0) or (ratio_pos_dir2==0)):
                logging.debug("found  overlap")
                logging.debug(str(ratio_pos_dir1)+";"+str((ratio_pos_dir2)))

            features = [ratio_pos_dir1, ratio_pos_dir2]


        logging.info(str("features:") + ";" + str((features)))





        logging.debug("and value of features is:" + str((features)))

        return features