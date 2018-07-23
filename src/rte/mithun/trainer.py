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

        hf=data_folder+annotated_head_split_folder
        hff=hf+annotated_only_lemmas
        hft=hf+annotated_only_tags
        hfd=hf+annotated_only_dep




        logging.debug("hff:" + str(hff))
        logging.debug("bff:" + str(bff))
        logging.info("going to read heads_lemmas from disk:")

        #heads_lemmas = read_json(hff,logging)
        heads_lemmas= read_json_with_id(hff)
        bodies_lemmas = read_json_with_id(bff)

        heads_tags= read_json_with_id(hft)
        bodies_tags = read_json_with_id(hft)
        #
        # heads_tags = read_json(hft,logging)
        # bodies_tags = read_json(hft,logging)
        heads_deps = read_json_deps(hfd)
        bodies_deps = read_json_deps(bfd)


        logging.debug("type of heads_deps is: " + str(type(heads_deps)))
        logging.debug("size of heads_deps is: " + str(len(heads_deps)))
        logging.debug("type of bodies_deps is: " + str(type(bodies_deps)))
        logging.debug("size of bodies_deps is: " + str(len(bodies_deps)))


        if not ((len(heads_lemmas) == len(bodies_lemmas))or (len(heads_tags) == len(bodies_tags)) or
                    (len(heads_deps) == len(bodies_deps)) ):
            logging.debug("size of heads_lemmas and bodies_lemmas dont match. going to quit")
            sys.exit(1)


        combined_vector = create_feature_vec(heads_lemmas, bodies_lemmas, heads_tags,
                                             bodies_tags,heads_deps,bodies_deps)

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

def create_feature_vec(heads_lemmas_obj_list, bodies_lemmas_obj_list, heads_tags_obj_list, bodies_tags_obj_list, heads_deps_obj_list, bodies_deps_obj_list):
    word_overlap_vector = np.empty((0, 1), float)
    hedging_words_vector = np.empty((0, 30), int)
    refuting_value_matrix = np.empty((0, 19), int)
    noun_overlap_matrix = np.empty((0, 2), float)
    vb_overlap_matrix = np.empty((0, 2), float)
    ant_overlap_matrix = np.empty((0, 2), float)
    neg_vb_matrix = np.empty((0, 2), float)


    counter=0
    for  head_lemmas, body_lemmas,head_tags_related,body_tags_related,head_deps,body_deps in \
            tqdm((zip(heads_lemmas_obj_list, bodies_lemmas_obj_list, heads_tags_obj_list, bodies_tags_obj_list, heads_deps_obj_list, bodies_deps_obj_list)),
                 total=len(bodies_tags_obj_list), desc="feat_gen:"):

        lemmatized_headline = head_lemmas
        lemmatized_body=body_lemmas
        tagged_headline=head_tags_related
        tagged_body=body_tags_related

        word_overlap_array, hedge_value_array, refuting_value_array, noun_overlap_array, verb_overlap_array, \
        antonym_overlap_array,neg_vb_array = add_vectors(
            lemmatized_headline, lemmatized_body, tagged_headline, tagged_body,head_deps,body_deps)

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


def add_vectors(lemmatized_headline_obj, lemmatized_body_obj, tagged_headline, tagged_body, head_deps, body_deps):



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

    lemmatized_headline_split = lemmatized_headline_data.split(" ")
    lemmatized_body_split = lemmatized_body_data.split(" ")

    logging.debug(doc_id_hl)
    logging.debug(doc_id_bl)
    logging.debug(doc_id_ht)
    logging.debug(doc_id_bt)
    logging.debug(doc_id_hd)
    logging.debug(doc_id_bd)

    logging.debug(lemmatized_headline_split)
    logging.debug(lemmatized_body_split)
    logging.debug(tagged_headline.data)
    logging.debug(tagged_body.data)
    logging.debug(head_deps.data)
    logging.debug(body_deps.data)

    sys.exit(1)

    headline_pos_split = tagged_headline.split(" ")
    body_pos_split = tagged_body.split(" ")

    neg_vb = negated_verbs_count(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                 body_pos_split, head_deps, body_deps, "VB")
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

'''number of verbs in sentence one that were negated in sentence 2
#find  all verbs that occur in headline.
        # then  for each of these verbs, check if this verb occurs in the body.
        # if it does then find the position of that verb in the body. then
        # take that position value, go through dependency parse # and find if any of the leading edges go through "neg"
        '''
def negated_verbs_count(lemmatized_headline_split, headline_pos_split, lemmatized_body_split, body_pos_split, head_deps,body_deps,pos_in):
        logging.info("inside negated_verbs_count")
        h_nouns = []
        b_nouns = []


        id_h=head_deps.doc_id
        id_b=body_deps.doc_id
        e_h=head_deps.data
        e_b = body_deps.data

        logging.debug(id_h)
        logging.debug(id_b)
        logging.debug(e_h)
        logging.debug(e_b)

        sys.exit(1)

        #vb_positions=find_pos_positions(headline_pos_split,pos_in)


        # logging.debug("body_deps")
        # logging.debug(body_deps)
        logging.debug(lemmatized_headline_split)
        logging.debug(lemmatized_body_split)

        # find  all verbs that occur in headline.
        verb_head_list = []
        for word1, pos in zip(lemmatized_headline_split, headline_pos_split):
            if pos.startswith(pos_in):
                verb_head_list.append(word1)

        logging.debug(verb_head_list)
        logging.debug("verb_head_list")
        vb_positions_body=[]
        # then  for each of these verbs, check if this verb occurs in the body.
        for vb_head in verb_head_list:
            for index,word2 in enumerate(lemmatized_body_split):
                logging.debug(word2)
                if (vb_head==word2):
                    # if it does then find the position of that verb in the body. then
                    vb_positions_body.append(index)
                    logging.debug("found a verb which has same verb in headline and body")
                    logging.debug(index)
                    logging.debug(vb_head)




        # take that position value, go through dependency parse # and find if any of the leading edges go through "neg"
        if(len(vb_positions_body)>0):
            logging.debug(vb_positions_body)
            for p in vb_positions_body:
                logging.debug(p)
                for edges in body_deps.data:
                        logging.debug(edges)
                        dest = edges["destination"]
                        src = edges["source"]
                        rel = edges["relation"]
                        logging.debug(src)
                        logging.debug(rel)
                        logging.debug(dest)

                        if ((p==src) and (rel=="neg")):
                            logging.debug("found a verb having negative edge")
                            logging.debug(lemmatized_headline_split)
                            logging.debug(lemmatized_body_split)
                            logging.debug(src)
                            logging.debug(rel)
                            logging.debug(dest)
                            # and find if any of the leading edges go through "neg"
                            sys.exit(1)


        features = [0, 0]




        return features



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
    counter=0
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

