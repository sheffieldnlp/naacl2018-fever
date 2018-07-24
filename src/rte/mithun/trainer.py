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
from sklearn import linear_model
import json
import nltk
from nltk.corpus import wordnet
import itertools
from .proc_data import PyProcDoc


API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
my_out_dir = "poop-out"
n_cores = 2
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
RELATED = LABELS[0:3]
annotated_only_lemmas="ann_lemmas.json"
annotated_only_tags="ann_tags.json"
annotated_only_dep="ann_deps.json"
annotated_words="ann_words.json"
annotated_body_split_folder="split_body/"
annotated_head_split_folder="split_head/"
#pick based on which folder you are running from. if not on home folder:
data_root="/work/mithunpaul/fever/my_fork/fever-baselines/"
data_folder_train=data_root+"/data/fever-data-ann/train/"
data_folder_dev=data_root+"/data/fever-data-ann/dev/"
model_trained="model_trained.pkl"

predicted_results="predicted_results.pkl"
combined_vector_training="combined_vector_testing_phase2.pkl"
# if __name__ == "__main__":
#         logger = setup_custom_logger('root')
#         logger.debug('main message')


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

        bf=data_folder+annotated_body_split_folder
        bff=bf+annotated_only_lemmas
        bft=bf+annotated_only_tags
        bfd = bf + annotated_only_dep
        bfw=bf+annotated_words

        hf=data_folder+annotated_head_split_folder
        hff=hf+annotated_only_lemmas
        hft=hf+annotated_only_tags
        hfd=hf+annotated_only_dep
        hfw=hf+annotated_words


        logging.debug("hff:" + str(hff))
        logging.debug("bff:" + str(bff))
        logging.info("going to read heads_lemmas from disk:")

        heads_lemmas= read_json_with_id(hff)
        bodies_lemmas = read_json_with_id(bff)

        heads_tags= read_json_with_id(hft)
        bodies_tags = read_json_with_id(bft)

        heads_deps = read_json_deps(hfd)
        bodies_deps = read_json_deps(bfd)


        heads_words = read_json_with_id(hfw)
        bodies_words = read_json_with_id(bfw)

        logging.debug("type of heads_deps is: " + str(type(heads_deps)))
        logging.debug("size of heads_deps is: " + str(len(heads_deps)))
        logging.debug("type of bodies_deps is: " + str(type(bodies_deps)))
        logging.debug("size of bodies_deps is: " + str(len(bodies_deps)))


        if not ((len(heads_lemmas) == len(bodies_lemmas))or (len(heads_tags) == len(bodies_tags)) or
                    (len(heads_deps) == len(bodies_deps)) ):
            logging.debug("size of heads_lemmas and bodies_lemmas dont match. going to quit")
            sys.exit(1)


        combined_vector = create_feature_vec(heads_lemmas, bodies_lemmas, heads_tags,
                                             bodies_tags,heads_deps,bodies_deps,heads_words, bodies_words)

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




def do_training(combined_vector,gold_labels_tr):
    logging.debug("going to load the classifier:")
    clf=svm.NuSVC()
    clf.fit(combined_vector, gold_labels_tr.ravel())

    file = model_trained
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


def normalize_dummy(text):
    x = text.lower().translate(remove_punctuation_map)
    return x.split(" ")

def create_feature_vec(heads_lemmas_obj_list, bodies_lemmas_obj_list, heads_tags_obj_list, bodies_tags_obj_list, heads_deps_obj_list, bodies_deps_obj_list,heads_words_list, bodies_words_list):
    word_overlap_vector = np.empty((0, 1), float)
    hedging_words_vector = np.empty((0, 30), int)
    refuting_value_matrix = np.empty((0, 19), int)
    noun_overlap_matrix = np.empty((0, 2), float)
    vb_overlap_matrix = np.empty((0, 2), float)
    ant_overlap_matrix = np.empty((0, 2), float)
    neg_vb_matrix = np.empty((0, 4), float)


    counter=0
    for  (lemmatized_headline, lemmatized_body,tagged_headline,tagged_body,head_deps,body_deps,heads_words, bodies_words) \
            in tqdm(zip(heads_lemmas_obj_list, bodies_lemmas_obj_list, heads_tags_obj_list, bodies_tags_obj_list, heads_deps_obj_list,
                        bodies_deps_obj_list,heads_words_list, bodies_words_list),total=len(bodies_tags_obj_list),desc="feat_gen:"):

        word_overlap_array, hedge_value_array, refuting_value_array, noun_overlap_array, verb_overlap_array,antonym_overlap_array,\
        neg_vb_array  = add_vectors(lemmatized_headline, lemmatized_body, tagged_headline, tagged_body,head_deps,body_deps,heads_words, bodies_words)

        logging.info("inside create_feature_vec. just received verb_overlap_array is =" + repr(verb_overlap_array))
        logging.info(verb_overlap_array)
        logging.info("inside create_feature_vec. vb_overlap_matrix is =" + repr(vb_overlap_matrix))
        logging.info("inside create_feature_vec. just received noun_overlap_array is =" + repr(noun_overlap_array))
        logging.info("inside create_feature_vec. noun_overlap_matrix is =" + repr(noun_overlap_matrix))

        word_overlap_vector = np.vstack([word_overlap_vector, word_overlap_array])
        hedging_words_vector = np.vstack([hedging_words_vector, hedge_value_array])
        refuting_value_matrix = np.vstack([refuting_value_matrix, refuting_value_array])
        noun_overlap_matrix = np.vstack([noun_overlap_matrix, noun_overlap_array])
        vb_overlap_matrix=np.vstack([vb_overlap_matrix, verb_overlap_array])
        ant_overlap_matrix = np.vstack([ant_overlap_matrix, antonym_overlap_array])
        neg_vb_matrix= np.vstack([neg_vb_matrix, neg_vb_array])


        logging.info("  word_overlap_vector is:" + str(word_overlap_vector))
        logging.info("refuting_value_matrix" + str(refuting_value_matrix))

        logging.info("noun_overlap_matrix is =" + str(noun_overlap_matrix))
        logging.info("shape  noun_overlap_matrix is:" + str(noun_overlap_matrix.shape))
        logging.info("vb_overlap_matrix is =" + str(vb_overlap_matrix))
        logging.info("shape  vb_overlap_matrix is:" + str(vb_overlap_matrix.shape))

        counter=counter+1







    logging.info("\ndone with all headline body.:")
    logging.info("shape of  word_overlap_vector is:" + str(word_overlap_vector.shape))
    logging.info("shape of  hedging_words_vector is:" + str(hedging_words_vector.shape))
    logging.info("shape of  refuting_value_matrix is:" + str(refuting_value_matrix.shape))
    logging.info("shape of  noun_overlap_matrix is:" + str(noun_overlap_matrix.shape))
    logging.info("shape of  vb_overlap_matrix is:" + str(vb_overlap_matrix.shape))

    # combined_vector= np.hstack(
    #     [word_overlap_vector, hedging_words_vector, refuting_value_matrix, noun_overlap_matrix,vb_overlap_matrix])

    combined_vector = np.hstack(
        [word_overlap_vector, hedging_words_vector, refuting_value_matrix, noun_overlap_matrix,ant_overlap_matrix,neg_vb_matrix])

    return combined_vector


def add_vectors(lemmatized_headline_obj, lemmatized_body_obj, tagged_headline, tagged_body, head_deps, body_deps, head_words, body_words):



    lemmatized_headline_data = lemmatized_headline_obj.data
    lemmatized_body_data= lemmatized_body_obj.data

    #split everywhere based on space-i.e for word overlap etc etc..
    lemmatized_headline_data = lemmatized_headline_data.lower()
    lemmatized_body_data = lemmatized_body_data.lower()

    doc_id_hl=lemmatized_headline_obj.doc_id
    doc_id_bl=lemmatized_body_obj.doc_id
    doc_id_ht=tagged_headline.doc_id
    doc_id_bt=tagged_body.doc_id
    doc_id_hd=head_deps.doc_id
    doc_id_bd=body_deps.doc_id
    doc_id_hw=head_words.doc_id
    doc_id_bw=body_words.doc_id

    lemmatized_headline_split = lemmatized_headline_data.split(" ")
    lemmatized_body_split = lemmatized_body_data.split(" ")
    headline_pos_split = tagged_headline.data.split(" ")
    body_pos_split = tagged_body.data.split(" ")

    logging.debug(doc_id_hl)
    logging.debug(doc_id_bl)
    logging.debug(doc_id_ht)
    logging.debug(doc_id_bt)
    logging.debug(doc_id_hd)
    logging.debug(doc_id_bd)
    logging.debug(doc_id_hw)
    logging.debug(doc_id_bw)

    if not (doc_id_hl == doc_id_bl == doc_id_ht == doc_id_bt == doc_id_hd == doc_id_bd == doc_id_hw == doc_id_bw):
        logging.error("all doc ids dont match going to quit")
        sys.exit(1)

    logging.info(lemmatized_headline_data)
    logging.info(lemmatized_body_data)
    logging.debug(tagged_headline.data)
    logging.debug(tagged_body.data)
    logging.debug(head_deps.data)
    logging.debug(body_deps.data)
    logging.debug(head_words.data)
    logging.debug(body_words.data)

    logging.debug(tagged_headline)
    logging.debug(tagged_body)
    logging.debug(headline_pos_split)
    logging.debug(body_pos_split)







    neg_vb = negated_verbs_count(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                 body_pos_split, head_deps, body_deps, "VB", head_words,body_words)
    neg_vb_array = np.array([neg_vb])

    antonym_overlap = antonym_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                      body_pos_split, "NN")
    antonym_overlap_array = np.array([antonym_overlap])


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
    vb_overlap_array = np.array([vb_overlap])




    return word_overlap_array,hedge_value_array,refuting_value_array,noun_overlap_array,vb_overlap_array,\
           antonym_overlap_array,neg_vb_array


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
        logging.info(str("overlap_pos_counter:") + ";" + str((overlap_noun_counter)))
        logging.info(str("overlap:") + ";" + str((overlap)))


        logging.debug(str("count_body:") + ";" + str((noun_count_body)))
        logging.debug(str("count_headline:") + ";" + str((noun_count_headline)))


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

#find positions where all verbs occur in headline.
def find_pos_positions(headline_pos_split,pos_in):
    positions=[]
    for index, pos in enumerate(headline_pos_split):
        if pos.startswith(pos_in):
            logging.debug("pos.startswith:" + str(pos_in))
            positions.append(index)

    return positions




# Finds count of verbs (or another POS) which are positive in text1 and negated in text2
def count_different_polarity(text1_lemmas, text1_pos, text1_deps, text2_lemmas, text2_pos, text2_deps, pos_in):
        #find all  verbs in headline
        text1_list= get_all_verbs(text1_lemmas,text1_pos,pos_in)
        text2_list= get_all_verbs(text2_lemmas,text2_pos,pos_in)
        positions_text1=given_verb_find_positions(text1_list, text1_lemmas)
        positions_text2=given_verb_find_positions(text2_list, text2_lemmas)

        #for each of these verbs find which all are -ves and which all are positives in head itself
        [verbs_negated_text1, verbs_positive_text1]=partition_by_polarity(positions_text1,text1_deps,text1_lemmas)
        [verbs_negated_text2, verbs_positive_text2]=partition_by_polarity(positions_text2,text2_deps,text2_lemmas)

        pos_text1_neg_text2 = len(set(verbs_positive_text1).intersection(set(verbs_negated_text2)))
        neg_text1_pos_text2 = len(set(verbs_negated_text1).intersection(set(verbs_positive_text2)))

        return [pos_text1_neg_text2, neg_text1_pos_text2]

        # #for each +ve verb in head find how many of those were negated in body
        # # e.g. ['be', 'jump'...]
        # positions_vb_pos_text1_in_text2=given_verb_find_positions(verbs_positive_text1, text2_lemmas)
        #
        # #if the verb doesn't even exist pos_head_neg_body= zero
        # if (len(positions_vb_pos_text1_in_text2) > 0):
        #     # e.g.
        #     # if both 'be' and 'jump' are negated body:
        #     #  pos_head_neg_body == 2
        #     pos_text1_neg_text2=get_neg_count(positions_vb_pos_text1_in_text2,text2_deps,text2_lemmas)
        # else:
        #     pos_text1_neg_text2=0

        # return pos_text1_neg_text2

def count_same_polarity_both_texts(text1_lemmas, text1_pos, text1_deps, text2_lemmas, text2_pos, text2_deps, pos_in):
     #for each -ve verb in head, find how many were negated in body also. if all were negated the feature denoting same polarity==0
        text1_list= get_all_verbs(text1_lemmas,text1_pos,pos_in)
        text2_list = get_all_verbs(text2_lemmas, text2_pos, pos_in)
        positions_text1=given_verb_find_positions(text1_list, text1_lemmas)
        positions_text2=given_verb_find_positions(text2_list, text2_lemmas)

        #for each of these verbs find which all are -ves and which all are positives in head itself
        [verbs_negated_text1, verbs_positive_text1]=partition_by_polarity(positions_text1,text1_deps,text1_lemmas)
        [verbs_negated_text2, verbs_positive_text2]=partition_by_polarity(positions_text2,text2_deps,text2_lemmas)

        neg_text1_neg_text2 = len(set(verbs_negated_text1).intersection(set(verbs_negated_text2)))
        pos_text1_pos_text2 = len(set(verbs_positive_text1).intersection(set(verbs_positive_text2)))
        return [neg_text1_neg_text2, pos_text1_pos_text2]


'''number of verbs in sentence one that were negated in sentence 2
#find  all verbs that occur in headline.
        # then  for each of these verbs, check if this verb occurs in the body.
        # if it does then find the position of that verb in the body. then
        # take that position value, go through dependency parse # and find if any of the leading edges go through "neg"
        '''
def negated_verbs_count(lemmatized_headline_split, headline_pos_split, lemmatized_body_split, body_pos_split, head_deps,body_deps,pos_in,head_words,body_words):
        #+ve in head -ve in body=[1,0,0]
        #-ve in head -ve in body=[0,0,1]
        # #-ve in head +ve in body=[0,1,0]
        pos_head_neg_body=0
        neg_head_pos_body=0
        neg_head_neg_body=0
        pos_head_pos_body=0

        logging.info("inside negated_verbs_count")
        # pos_text1_neg_text2, neg_text1_pos_text2
        [pos_head_neg_body, neg_head_pos_body] = count_different_polarity(lemmatized_headline_split, headline_pos_split, head_deps,
                                                                          lemmatized_body_split, body_pos_split, body_deps, pos_in="VB")

        [neg_head_neg_body, pos_head_pos_body] = count_same_polarity_both_texts(lemmatized_headline_split, headline_pos_split, head_deps,
                                                                                lemmatized_body_split, body_pos_split, body_deps, pos_in="VB")

        # first two are different poalrity counts, second two are same polarity counts
        features = [pos_head_neg_body, neg_head_pos_body, neg_head_neg_body, pos_head_pos_body]

        # DEBUG
        logging.info(head_words.data)
        logging.info(body_words.data)
        logging.info(features)


        # if neg_head_pos_body > 0:
        #     logging.info("neg_head_pos_body>0")
        #     sys.exit(1)

        if pos_head_neg_body > 0:
            logging.info("pos_head_neg_body>0")
            sys.exit(1)

        if   neg_head_neg_body > 0:
            logging.info("neg_head_neg_body>0")
            sys.exit(1)





        return features


        # # #find all verbs in headline
        # # verb_head_list= get_all_verbs(lemmatized_headline_split,headline_pos_split,pos_in)
        # # vb_positions_head=given_verb_find_positions(verb_head_list, lemmatized_headline_split)
        # #
        # #
        # # #for each of these verbs find which all are -ves and which all are positives in head itself
        # # verbs_negated_head=get_neg_list(vb_positions_head,head_deps,lemmatized_headline_split)
        # # #list of positive verbs will be the ones that are not negated
        # # list_of_pos_verb_h=set(verb_head_list).difference(set(verbs_negated_head))
        # #
        # # #for each +ve verb in head find how many of those were negated in body
        # # # e.g. ['be', 'jump'...]
        # # positions_vb_pos_head_in_body=given_verb_find_positions(list_of_pos_verb_h, lemmatized_body_split)
        # #
        # # #if the verb doesn't even exist pos_head_neg_body= zero
        # # if (len(positions_vb_pos_head_in_body) > 0):
        # #     # e.g.
        # #     # if both 'be' and 'jump' are negated body:
        # #     #  pos_head_neg_body == 2
        # #     pos_head_neg_body=get_neg_count(positions_vb_pos_head_in_body,body_deps,lemmatized_body_split)
        # # else:
        # #     pos_head_neg_body=0
        #
        #
        #
        # # logging.info(verb_head_list)
        # # logging.info(vb_positions_head)
        # # logging.info(pos_head_neg_body)
        # # logging.info(features)
        # # logging.info(neg_head_pos_body)
        # # logging.info(verbs_negated_head)
        # # logging.info(list_of_pos_verb_h)
        #
        #
        # #if atleast one of the positive verbs in headline was negated, change the value to the count and the feature denoting same polarity==0
        # # if(pos_head_neg_body>0):
        # #     #[1,0,0]
        # #     features[0]=pos_head_neg_body
        # #     features[1]=0
        # #     features[2]=0
        # #     sys.exit(1)
        # # else:
        # #
        # #     #[0,0,1]
        # #     features[0]=0
        # #     features[1]=0
        # #     features[2]=1
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # #for each -ve verb in head, find how many were negated in body also. if all were negated the feature denoting same polarity==0
        # positions_vb_neg_head_in_body=given_verb_find_positions(verbs_negated_head, lemmatized_body_split)
        #
        #
        #
        # #if the verb doesn't even exist neg_head_pos_body= zero
        # if (len(positions_vb_neg_head_in_body) > 0):
        #     neg_head_neg_body=get_neg_count(positions_vb_neg_head_in_body,body_deps,lemmatized_body_split)
        # else:
        #     neg_head_neg_body=0
        #
        #
        #
        # logging.info(verb_head_list)
        # logging.info(verbs_negated_head)
        # logging.info(list_of_pos_verb_h)
        # logging.info(vb_positions_head)
        # logging.info(pos_head_neg_body)
        # logging.info(features)
        # logging.info(neg_head_pos_body)
        # logging.info(positions_vb_neg_head_in_body)
        #
        #
        #
        #
        # if(neg_head_pos_body>0):
        #     if(neg_head_pos_body==len(verbs_negated_head)):
        #         #[0,0,1]
        #         features[0]=0
        #         features[1]=0
        #         features[2]=1
        #
        #
        #     else:
        #         #[0,1,0]
        #         features[0]=0
        #         features[1]=neg_head_pos_body
        #         features[2]=0
        #
        #
        #
        #
        #
        # #
        # # #feature 2: find no of verbs in body that were negated in headline
        # # verb_body_list= get_all_verbs(lemmatized_body_split,body_pos_split,pos_in)
        # # vb_positions_head=given_verb_find_positions(verb_body_list, lemmatized_headline_split)
        # # neg_head_pos_body=get_neg_count(vb_positions_head,head_deps,lemmatized_headline_split)
        # # features[1]=neg_head_pos_body
        # #
        # #
        # #
        # #
        # #
        # #
        # #
        # # verb_head_list= get_all_verbs(lemmatized_headline_split,headline_pos_split,pos_in)
        # # vb_positions_head=given_verb_find_positions(verb_head_list, lemmatized_headline_split)
        # # verbs_negated_head=get_neg_list(vb_positions_head,head_deps,lemmatized_headline_split)
        # # verb_body_list= get_all_verbs(lemmatized_body_split,body_pos_split,pos_in)
        # # positions_vb_neg_head_in_body=given_verb_find_positions(verb_body_list, lemmatized_body_split)
        # # verbs_negated_body=get_neg_list(positions_vb_neg_head_in_body,body_deps,lemmatized_body_split)
        # #
        # #
        # #
        # # list_of_pos_verb_b=set(verb_body_list).difference(set(verbs_negated_body))
        # #
        # #
        # # logging.info(verb_head_list)
        # # logging.info(vb_positions_head)
        # # logging.info(verbs_negated_head)
        # # logging.info(verb_body_list)
        # # logging.info(positions_vb_neg_head_in_body)
        # # logging.info(verbs_negated_body)
        # # logging.info(verbs_negated_body)
        # # logging.info(list_of_pos_verb_h)
        # # logging.info(list_of_pos_verb_b)
        # # logging.info(len(list_of_pos_verb_h))
        # # logging.info(len(list_of_pos_verb_b))
        # #
        # #
        # # lph=len(list_of_pos_verb_h)
        # # lpb=len(list_of_pos_verb_b)
        # #
        # # # if the negative polarity status is same, add that as another feature. i.e if verb is negated in both headline and body, that is one
        # #
        # #
        # # if ((len(verbs_negated_head) > 0) and (len(verbs_negated_body) > 0)):
        # #     if(set(verbs_negated_head).intersection(set(verbs_negated_body))==0):
        # #         logging.info("found that verbs in both sentences have same polarity")
        # #         features[2]=1
        # #         logging.info(features)
        # #         sys.exit(1)
        # #
        # #
        # # # if both headline and body had same verb and its polarity is positive
        # #
        # #
        # # if((lph > 0) and (lpb > 0)):
        # #     if( len ( (list_of_pos_verb_h).intersection((list_of_pos_verb_b))) > 0):
        # #         logging.info("found that verbs in both sentences have same positive polarity")
        # #         features[3]=1
        # #
        # #
        # #
        # # if(features[0]>0  or features[2]>0  or features[3]>0):
        # #         logging.info(features)
        # #         sys.exit(1)
        # #
        # #
        # logging.info(features)
        #



        #proportion of verbs in headline that was negated in body and vice versa. not count, but proportion.





        # return features

'''given positions of verbs find how many of them are negated in the given sentence
inputs:
array/list of verb positions int[]
dependency parse of the sentence
'''
def get_neg_count(vb_positions, sent_deps, lemmatized_sent_split):
    vb_list=get_neg_list(vb_positions, sent_deps, lemmatized_sent_split)
    logging.debug("vb_list:"+str(vb_list))
    return len(vb_list)


'''given positions of verbs find which all were negated in the given sentence
inputs:
outputs:
    return two lists, the verbs that are negated and those that are not
'''
def partition_by_polarity(vb_positions, sent_deps,lemmatized_sent_split):
        vb_count_list_negated=[]
        #vb_count_list_positive=[]



        # take that position value, go through dependency parse # and find if any of the leading edges go through "neg"
        if(len(vb_positions)>0):
            logging.debug(vb_positions)
            for p in vb_positions:
                logging.debug(p)
                for edges in sent_deps.data:
                    #list [Dict]
                        logging.debug(edges)
                        dest = edges["destination"]
                        src = edges["source"]
                        rel = edges["relation"]
                        logging.debug(src)
                        logging.debug(rel)
                        logging.debug(dest)

                        if (p==src):
                            if (rel=="neg"):
                                logging.debug("found a verb having negative edge")
                                logging.debug(src)
                                logging.debug(rel)
                                logging.debug(dest)
                                # and find if any of the leading edges go through "neg"-add it as a feature
                                vb_count_list_negated.append(lemmatized_sent_split[p])
                            # else:
                            #   else  vb_count_list_positive.append(lemmatized_sent_split[p])
        vb_count_list_positive = [lemmatized_sent_split[p] for p in vb_positions if lemmatized_sent_split[p] not in vb_count_list_negated]
        return vb_count_list_negated, vb_count_list_positive


'''given a list of verbs find all the positions if and where they occur in the given sentence'''
def given_verb_find_positions(verb_list, lemmatized_sent):
        vb_positions_body=[]
        for vb_head in verb_list:
            for index,word2 in enumerate(lemmatized_sent):
                if (vb_head==word2):
                    vb_positions_body.append(index)
        return vb_positions_body

# find  all verbs that occur in a given sentence.
def get_all_verbs(lemmatized_headline_split, headline_pos_split,pos_in):
        verb_head_list = []
        for word1, pos in zip(lemmatized_headline_split, headline_pos_split):
            if pos.startswith(pos_in):
                verb_head_list.append(word1)
        return verb_head_list

#number of nouns in sentence 2 that were antonyms of anyword in sentence 1 and vice versa
def antonym_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split, body_pos_split, pos_in):

        logging.info("inside " + pos_in + " antonyms")
        logging.info("lemmatized_headline_split " +str(lemmatized_headline_split))
        logging.info("lemmatized_headline_split " + str(lemmatized_body_split))
        h_nouns = []
        b_nouns = []
        h_nouns_antonyms=[]
        b_nouns_antonyms = []

        noun_count_headline = 0
        for word1, pos in zip(lemmatized_headline_split, headline_pos_split):
            logging.debug(str("pos:") + str((pos)))
            logging.debug(str("word:")  + str((word1)))
            if pos.startswith(pos_in):
                logging.debug("pos.startswith:"+str(pos_in))
                noun_count_headline = noun_count_headline + 1
                h_nouns.append(word1)
                ant_h_list=get_ant(word1)
                logging.debug(ant_h_list)
                if(len(ant_h_list)>0):
                    logging.debug("ant_h_list:")
                    logging.debug(ant_h_list)
                    h_nouns_antonyms.append(ant_h_list)




        noun_count_body = 0
        for word2, pos in zip(lemmatized_body_split, body_pos_split):
            logging.debug(str("pos:") + str((pos)))
            logging.debug(str("word:") + str((word2)))
            if pos.startswith(pos_in):
                noun_count_body = noun_count_body + 1
                b_nouns.append(word2)
                ant_b_list = get_ant(word2)
                if (len(ant_b_list) > 0):
                    logging.debug("ant_b_list:")
                    logging.debug(ant_b_list)
                    b_nouns_antonyms.append(ant_b_list)




        overlap_dir1=0
        overlap_dir2=0

        #number of nouns in evidence that were antonyms of any word in claim
        if(len(h_nouns_antonyms)>0):

            logging.info(("len h_nouns_antonyms"))
            logging.info(len(h_nouns_antonyms))
            flatten_h = list(itertools.chain.from_iterable(h_nouns_antonyms))
            logging.info(" flatten_h1:" + str((flatten_h)))
            logging.info(str("b_nouns:") + ";" + str((b_nouns)))
            overlap = set(flatten_h).intersection(set(b_nouns))

            if(len(overlap)>0):
                logging.info("found overlap1")
                logging.info(overlap)
                overlap_dir1 = len(overlap)

        #vice versa
        if (len(b_nouns_antonyms) > 0):

            logging.info(("len b_nouns_antonyms"))
            logging.info(len(b_nouns_antonyms))
            flatten_b = list(itertools.chain.from_iterable(b_nouns_antonyms))
            logging.info(" flatten_b:" + str((flatten_b)))
            logging.info(str("h_nouns:") + ";" + str((h_nouns)))
            overlap2 = set(flatten_b).intersection(set(h_nouns))

            if (len(overlap2) > 0):
                logging.info("found overlap2")
                logging.info(overlap2)
                overlap_dir2 = len(overlap2)



        features = [overlap_dir1, overlap_dir2]
        logging.debug(str("features_ant:") + str((features)))


        return features

def read_json_deps(json_file):
    logging.debug("inside read_json_deps")
    l = []

    py_proc_doc_list=[]

    with open(json_file) as f:
        for eachline in (f):
            obj_doc=PyProcDoc()
            d = json.loads(eachline)
            a=d["data"]
            b = d["doc_id"]
            obj_doc.doc_id=b
            for e in a:
                edges=e["edges"]
                obj_doc.data=edges

            py_proc_doc_list.append(obj_doc)

    return py_proc_doc_list


def read_json_with_id(json_file):
    logging.debug("inside read_json_deps")

    py_proc_doc_list=[]

    with open(json_file) as f:
        for eachline in (f):
            obj_doc=PyProcDoc()
            d = json.loads(eachline)
            a=d["data"]
            just_lemmas=' '.join(str(r) for v in a for r in v)
            obj_doc.data=just_lemmas
            b = d["doc_id"]
            obj_doc.doc_id=b
            py_proc_doc_list.append(obj_doc)

    return py_proc_doc_list


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


def get_ant(word):
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return antonyms

