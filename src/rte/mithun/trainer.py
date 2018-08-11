from __future__ import division
import sys,logging
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import os
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
from processors import ProcessorsBaseAPI
import json
from nltk.corpus import wordnet
import itertools
from .proc_data import PyProcDoc
import torchwordemb


API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
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
data_folder_train_small=data_root+"/data/fever-data-ann/train_small/"
data_folder_dev=data_root+"/data/fever-data-ann/dev/"
data_folder_test=data_root+"/data/fever-data-ann/test/"
model_trained="model_trained.pkl"
path_glove_server="/data/nlp/corpora/glove/6B/glove.6B.300d.txt"

predicted_results="predicted_results.pkl"
combined_vector_pkl= "combined_vector.pkl"


def read_json_create_feat_vec(load_ann_corpus_tr,args):

    #just load feature vector alone. No dynamically adding new features
    if (args.load_feat_vec==True):
        logging.info("going to load combined vector from disk")
        combined_vector = joblib.load(combined_vector_pkl)



    else:
        logging.debug("load_feat_vec is falsse. going to generate features")
        logging.debug("value of load_ann_corpus_tph2:" + str(load_ann_corpus_tr))

        cwd=os.getcwd()
        data_folder=None
        if(args.mode=="dev"):
            data_folder=data_folder_dev
        else:
            if(args.mode=="train"):
                data_folder=data_folder_train
            else:
                   if(args.mode=="small"):
                        data_folder=data_folder_train_small
                   else:
                       if (args.mode == "test"):
                            data_folder = data_folder_test


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

        logging.info("going to load glove vectors...")
        vocab, vec = torchwordemb.load_glove_text(path_glove_server)

        #if we are doing dynamic cv addition.make sure load cv is true and load that cv
        if (args.dynamic_cv==True):
                logging.info("going to load combined vector from disk")
                logging.info("dynamic_cv=true, load_feat vec=true ")
                combined_vector_old = joblib.load(combined_vector_pkl)
                logging.info("shaped of combined_vector_old:"+str(combined_vector_old.shape))
                combined_vector = create_feature_vec_one_feature(heads_lemmas, bodies_lemmas, heads_tags,
                                             bodies_tags,heads_deps,bodies_deps,heads_words, bodies_words,combined_vector_old,vocab,vec)

        else:
            combined_vector = create_feature_vec(heads_lemmas, bodies_lemmas, heads_tags,
                                             bodies_tags,heads_deps,bodies_deps,heads_words, bodies_words,vocab,vec)

        joblib.dump(combined_vector, combined_vector_pkl)
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
    logging.debug("going to train the classifier:")
    clf=svm.NuSVC()
    clf.fit(combined_vector, gold_labels_tr.ravel())

    file = model_trained
    joblib.dump(clf, file)
    logging.debug("done saving model to disk")

def load_model():
    model=joblib.load(model_trained)
    return model;


def print_missed():
    pred=joblib.load(p, predicted_results)

def do_testing(combined_vector,svm):
    logging.info("all value of combined_vector is:"+str(combined_vector))
    logging.info("going to predict...")
    p=svm.predict(combined_vector)
    joblib.dump(p, predicted_results)
    logging.debug("done with predictions")
    return p

#
# def normalize_dummy(text):
#     x = text.lower().translate(remove_punctuation_map)
#     return x.split(" ")

def create_feature_vec (heads_lemmas_obj_list, bodies_lemmas_obj_list,
                        heads_tags_obj_list, bodies_tags_obj_list, heads_deps_obj_list, bodies_deps_obj_list,heads_words_list,
                        bodies_words_list,vocab,vec):
    word_overlap_vector = np.empty((0, 1), float)
    hedging_words_vector = np.empty((0, 30), int)
    refuting_value_matrix = np.empty((0, 19), int)
    noun_overlap_matrix = np.empty((0, 2), float)
    vb_overlap_matrix = np.empty((0, 2), float)
    ant_noun_overlap_matrix = np.empty((0, 2), float)
    ant_adj_overlap_matrix = np.empty((0, 2), float)
    ant_overlap_matrix = np.empty((0, 2), float)
    neg_vb_matrix = np.empty((0, 4), float)
    hedging_headline_matrix = np.empty((0, 30), int)
    num_overlap_matrix = np.empty((0, 2), float)
    emb_cos_sim_matrix = np.empty((0, 1), float)


    counter=0
    #debug:find total number of sentences which had numbers, either in headline, body or both
    num_o=0
    num_h=0
    num_b=0
    for  (lemmatized_headline, lemmatized_body,tagged_headline,tagged_body,head_deps,body_deps,head_words,body_words) \
            in tqdm(zip(heads_lemmas_obj_list, bodies_lemmas_obj_list, heads_tags_obj_list, bodies_tags_obj_list, heads_deps_obj_list,
                        bodies_deps_obj_list,heads_words_list, bodies_words_list),total=len(bodies_tags_obj_list),desc="feat_gen:"):

        word_overlap_array, hedge_value_array, refuting_value_array, noun_overlap_array, verb_overlap_array, \
        antonym_overlap_array,num_overlap_array,hedge_headline_array,neg_vb_array,antonym_adj_overlap_array,emb_cosine_sim_array \
            = add_vectors\
                (lemmatized_headline, lemmatized_body, tagged_headline, tagged_body,head_deps, body_deps,head_words,body_words,vocab,vec)

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
        hedging_headline_matrix = np.vstack([hedging_headline_matrix, hedge_headline_array])
        num_overlap_matrix = np.vstack([num_overlap_matrix, num_overlap_array])
        neg_vb_matrix= np.vstack([neg_vb_matrix, neg_vb_array])
        ant_adj_overlap_matrix = np.vstack([ant_adj_overlap_matrix, antonym_adj_overlap_array])
        ant_noun_overlap_matrix = np.vstack([ant_noun_overlap_matrix, antonym_overlap_array])
        emb_cos_sim_matrix = np.vstack([emb_cos_sim_matrix, emb_cosine_sim_array])






        logging.debug("  word_overlap_vector is:" + str(word_overlap_vector))
        logging.debug("refuting_value_matrix" + str(refuting_value_matrix))
        logging.debug("noun_overlap_matrix is =" + str(noun_overlap_matrix))
        logging.debug("shape  noun_overlap_matrix is:" + str(noun_overlap_matrix.shape))
        logging.debug("vb_overlap_matrix is =" + str(vb_overlap_matrix))
        logging.debug("shape  vb_overlap_matrix is:" + str(vb_overlap_matrix.shape))
        logging.debug("num_overlap matrix is =" + str(num_overlap_matrix))
        logging.debug("shape  num_overlap_matrix is:" + str(num_overlap_matrix.shape))

        counter=counter+1







    logging.info("\ndone with all headline body.:")
    logging.info("overall number count is:" + str(num_o))
    logging.info("headlines that have numbers is:" + str(num_h))
    logging.info("body that has numbers is:" + str(num_b))


    logging.info("shape of  word_overlap_vector is:" + str(word_overlap_vector.shape))
    logging.info("shape of  hedging_words_vector is:" + str(hedging_words_vector.shape))
    logging.info("shape of  refuting_value_matrix is:" + str(refuting_value_matrix.shape))
    logging.info("shape of  noun_overlap_matrix is:" + str(noun_overlap_matrix.shape))
    logging.info("shape of  vb_overlap_matrix is:" + str(vb_overlap_matrix.shape))
    logging.info("shape  num_overlap_matrix is:" + str(num_overlap_matrix.shape))

    #no antonym ones
    # combined_vector = np.hstack(
    #     [word_overlap_vector, hedging_words_vector, refuting_value_matrix,
    #      noun_overlap_matrix,hedging_headline_matrix,neg_vb_matrix,emb_cos_sim_matrix])

    #no negation related. has everything else
    #refuting_value_matrix,neg_vb_matrix
    # combined_vector = np.hstack(
    #     [word_overlap_vector, hedging_words_vector,
    #      noun_overlap_matrix, ant_overlap_matrix, hedging_headline_matrix, ant_noun_overlap_matrix,
    #      ant_adj_overlap_matrix, emb_cos_sim_matrix])

    #all vectors


    #remove hedging features only:hedging_words_vector
    # combined_vector = np.hstack(
    #     [word_overlap_vector , refuting_value_matrix,
    #      noun_overlap_matrix, ant_overlap_matrix, hedging_headline_matrix, neg_vb_matrix, ant_noun_overlap_matrix,
    #      ant_adj_overlap_matrix,emb_cos_sim_matrix])

    #removing all overlap features:word_overlap_vector,noun_overlap_matrix
    # combined_vector = np.hstack(
    #     [hedging_words_vector, refuting_value_matrix
    #      , ant_overlap_matrix, hedging_headline_matrix, neg_vb_matrix, ant_noun_overlap_matrix,
    #      ant_adj_overlap_matrix,emb_cos_sim_matrix])

    combined_vector = np.hstack(
        [word_overlap_vector, hedging_words_vector, refuting_value_matrix,
         noun_overlap_matrix, ant_overlap_matrix, hedging_headline_matrix, neg_vb_matrix, ant_noun_overlap_matrix,
         ant_adj_overlap_matrix, emb_cos_sim_matrix])

    logging.info("shape  combined_vector is:" + str(combined_vector.shape))
    return combined_vector


'''Overloaded version for: if and when you add a new feature, you won't have to go through feature creation of the previously existing features. Just load old ones, and
attach just the new one alone. Time saven.'''
def create_feature_vec_one_feature(heads_lemmas_obj_list,
                                   bodies_lemmas_obj_list, heads_tags_obj_list, bodies_tags_obj_list, heads_deps_obj_list,
                                   bodies_deps_obj_list,heads_words_list, bodies_words_list,combined_vector,vocab,vec):
    logging.info("inside create_feature_vec overloaded")

    new_feature_matrix = np.empty((0, 1), float)
    counter=0


    for  (lemmatized_headline, lemmatized_body,tagged_headline,tagged_body,head_deps,body_deps,head_words,body_words) in \
            tqdm(zip(heads_lemmas_obj_list, bodies_lemmas_obj_list, heads_tags_obj_list,
                     bodies_tags_obj_list
                , heads_deps_obj_list,bodies_deps_obj_list,heads_words_list, bodies_words_list),total=len(bodies_tags_obj_list),
                 desc="feat_gen_dynamic_cv:"):

        new_feature_array= add_vectors_one_feature (lemmatized_headline, lemmatized_body,
                                                    tagged_headline, tagged_body,head_deps, body_deps,head_words,body_words,vocab, vec)

        new_feature_matrix = np.vstack([new_feature_matrix, new_feature_array])


        logging.info("new_feature_matrix matrix is =" + str(new_feature_matrix))
        logging.info("shape  new_feature_matrix is:" + str(new_feature_matrix.shape))

        counter=counter+1




    logging.info("done with all headline body.:")


    logging.info("shape  combined_vector before stacking is:" + str(combined_vector.shape))


    combined_vector = np.hstack([combined_vector, new_feature_matrix])
    logging.info("shape  combined_vector after stacking is:" + str(combined_vector.shape))

    return combined_vector


'''overloaded version that will be used for generarting one feature at a time'''
def add_vectors_one_feature(lemmatized_headline_obj, lemmatized_body_obj, tagged_headline, tagged_body
                            , head_deps, body_deps, head_words, body_words,vocab, vec):
    logging.info("inside add_vectors_one_feature ")


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


        #remove stop words
    stop_words = {'some', 'didn', 'itself', 'how', 'an', 'in', 'about', 'to', 'a', 'through', 've', 'ours', 'wouldn',
                  'y', 'from',
      'weren', "you've", 'yourselves', 'ain', 'or', 'mustn', 'so', 'that', 'them', 'such', 'being', 'her', 'doesn',
      'if', 'of', 'by', 'for', 'shouldn', 'll', 'are', 'any', 'doing', 'my', 'just', 'hers', 'its', 'i', 'further',
      'myself', 'then', 'yours', 'the', 'there', "you're", 'can', 'ourselves', "you'll", 'with', 'as', 'him', "shan't",
      'own', 'than', 'aren', 'nor', 'you', 'at', 'mightn', 'hasn', 'am', 'shan', 'needn', 'this', 'having', 'hadn',
      'yourself', 'themselves', 'too', 'couldn', 'will', "aren't", "you'd", 'more', 'few', 'our', 'most', 'very', 'me',
      'into', 'their', 'those', 'wasn', 'all', 'here', 'been', 'your', 'on','isn','these', 'until', 'haven', 'we',
        'theirs', 'be', 'what', 'while', 'why', 'where', 'which', 'when', 'who','whom', 'his', 'they', 'she', 'himself',
                  'herself', 'has', 'have', 'do','and','is' , "weren't",'were', 'did', "did n't", 'it', "won't", "doesn't",
                  'had', "needn't", "wouldn't","that'll", "mightn't","hadn't","mustn't",'he',"don't","she's", "isn't","should've",
                  'should', "shouldn't",'does',"couldn't","wasn't","haven't","hasn't",'was', "it's"}


    lemmatized_headline_split_sw = [w for w in lemmatized_headline_split if not w in stop_words]
    lemmatized_body_split_sw = [w for w in lemmatized_body_split if not w in stop_words]



    emb_overlap = embed_cosine_sim_features(lemmatized_headline_split_sw, lemmatized_body_split_sw,vocab, vec)
    emb_overlap_array = np.array([emb_overlap])





    return emb_overlap_array



def add_vectors(lemmatized_headline_obj, lemmatized_body_obj, tagged_headline, tagged_body, head_deps, body_deps,
                head_words, body_words,vocab,vec):



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


        #remove stop words
    stop_words = {'some', 'didn', 'itself', 'how', 'an', 'in', 'about', 'to', 'a', 'through', 've', 'ours', 'wouldn',
                  'y', 'from',
      'weren', "you've", 'yourselves', 'ain', 'or', 'mustn', 'so', 'that', 'them', 'such', 'being', 'her', 'doesn',
      'if', 'of', 'by', 'for', 'shouldn', 'll', 'are', 'any', 'doing', 'my', 'just', 'hers', 'its', 'i', 'further',
      'myself', 'then', 'yours', 'the', 'there', "you're", 'can', 'ourselves', "you'll", 'with', 'as', 'him', "shan't",
      'own', 'than', 'aren', 'nor', 'you', 'at', 'mightn', 'hasn', 'am', 'shan', 'needn', 'this', 'having', 'hadn',
      'yourself', 'themselves', 'too', 'couldn', 'will', "aren't", "you'd", 'more', 'few', 'our', 'most', 'very', 'me',
      'into', 'their', 'those', 'wasn', 'all', 'here', 'been', 'your', 'on','isn','these', 'until', 'haven', 'we',
        'theirs', 'be', 'what', 'while', 'why', 'where', 'which', 'when', 'who','whom', 'his', 'they', 'she', 'himself',
                  'herself', 'has', 'have', 'do','and','is' , "weren't",'were', 'did', "did n't", 'it', "won't", "doesn't",
                  'had', "needn't", "wouldn't","that'll", "mightn't","hadn't","mustn't",'he',"don't","she's", "isn't","should've",
                  'should', "shouldn't",'does',"couldn't","wasn't","haven't","hasn't",'was', "it's"}

    logging.debug(stop_words)

    lemmatized_headline_split_sw = [w for w in lemmatized_headline_split if not w in stop_words]
    lemmatized_body_split_sw = [w for w in lemmatized_body_split if not w in stop_words]




    neg_vb = negated_verbs_count(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                 body_pos_split, head_deps, body_deps, "VB", head_words,body_words)
    neg_vb_array = np.array([neg_vb])



    num_overlap = num_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                      body_pos_split, "CD")
    num_overlap_array = np.array([num_overlap])


    antonym_noun_overlap = antonym_overlap_features(lemmatized_headline_split_sw, headline_pos_split, lemmatized_body_split_sw,
                                      body_pos_split, "NN")
    antonym_noun_overlap_array = np.array([antonym_noun_overlap])

    antonym_adj_overlap = antonym_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                      body_pos_split, "NN")
    antonym_adj_overlap_array = np.array([antonym_adj_overlap])

    antonym_overlap = antonym_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                               body_pos_split, "JJ")
    antonym_overlap_array = np.array([antonym_overlap])

    word_overlap = word_overlap_features_mithun(lemmatized_headline_split_sw, lemmatized_body_split_sw)
    word_overlap_array = np.array([word_overlap])

    hedge_value = hedging_features_body( lemmatized_body_split)
    hedge_value_array = np.array([hedge_value])

    hedge_headline = hedging_features_headline(lemmatized_headline_split)
    hedge_headline_array = np.array([hedge_headline])

    refuting_value = refuting_features_mithun(lemmatized_headline_split, lemmatized_body_split)
    refuting_value_array = np.array([refuting_value])

    noun_overlap = pos_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split, body_pos_split, "NN")
    noun_overlap_array = np.array([noun_overlap])

    vb_overlap = pos_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split,
                                        body_pos_split, "VB")
    vb_overlap_array = np.array([vb_overlap])


    emb_overlap = embed_cosine_sim_features(lemmatized_headline_split_sw, lemmatized_body_split_sw,vocab, vec)
    emb_overlap_array = np.array([emb_overlap])

    return word_overlap_array,hedge_value_array,refuting_value_array,noun_overlap_array,vb_overlap_array\
        ,antonym_overlap_array,num_overlap_array,hedge_headline_array,neg_vb_array,\
           antonym_adj_overlap_array,emb_overlap_array


def word_overlap_features_mithun(clean_headline, clean_body):
    # todo: try adding word overlap features direction based, like noun overlap...i.e have 3 overall..one this, and 2 others.

    features = [
        len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]

    return features

def hedging_features_body(clean_body):

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

def hedging_features_headline(clean_headline):

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
    hedging_h_vector = [0] * length_hedge


    for word in clean_headline:
        if word in hedging_words:
            index=hedging_words.index(word)
            hedging_h_vector[index]=1

    return hedging_h_vector


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
        logging.info(lemmatized_headline_split)
        logging.info(lemmatized_body_split)
        logging.info(features)


        # if pos_head_pos_body > 0:
        #     logging.info("pos_head_pos_body>0")
        #     sys.exit(1)


        # if neg_head_pos_body > 0:
        #     logging.info("neg_head_pos_body>0")
        #     sys.exit(1)
        #
        # if pos_head_neg_body > 0:
        #     logging.info("pos_head_neg_body>0")
        #     sys.exit(1)

        # if   neg_head_neg_body > 0:
        #     logging.info("neg_head_neg_body>0")
        #     sys.exit(1)





        return features


'''given positions of verbs find how many of them are negated in the given sentence
inputs:
array/list of verb positions int[]
dependency parse of the sentence
'''
# def get_neg_count(vb_positions, sent_deps, lemmatized_sent_split):
#     vb_list=get_neg_list(vb_positions, sent_deps, lemmatized_sent_split)
#     logging.debug("vb_list:"+str(vb_list))
#     return len(vb_list)


'''given positions of verbs find which all were negated in the given sentence
inputs:
outputs:
    return two lists, the verbs that are negated and those that are not
'''
def partition_by_polarity(vb_positions, sent_deps,lemmatized_sent_split):
        vb_count_list_negated=[]
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

def get_sum_vector_embedding(vocab,vec, sent):
    sum = None
    very_first_time=True;

    for index, x in tqdm(enumerate(sent), total=len(sent),desc="load_embed"):
        if (x in vocab):
            logging.info("index:"+str(index))
            logging.info("x:" + str(x))
            emb = vec[vocab[x]]
            logging.info(emb.shape)
            q = emb.numpy()
            logging.info(q.shape)
            if (very_first_time):
                sum=q
                logging.debug(sum)
                very_first_time=False
            else:
                logging.debug(q)
                sum = sum + q
                logging.info(sum.shape)
                logging.debug(sum)

    return sum

def embed_cosine_sim_features(lemmatized_headline_split_sw, lemmatized_body_split_sw,vocab, vec):
    logging.info(" got inside embed_cosine_sim_features  ")


    sum_h=get_sum_vector_embedding(vocab,vec,lemmatized_headline_split_sw)
    sum_b = get_sum_vector_embedding(vocab, vec, lemmatized_body_split_sw)


    logging.info(" lemmatized_headline_split_sw vector ")
    logging.info(str((lemmatized_headline_split_sw)))
    logging.info(" lemmatized_body_split_sw vector ")
    logging.info(str((lemmatized_body_split_sw)))


    logging.info(" size vector for body is ")
    logging.info(str(len(sum_b)))

    logging.info(" size vector for head is ")
    logging.info(str(len(sum_h)))



    sum_h_r= sum_h.reshape(1,-1)
    sum_b_r = sum_b.reshape(1,-1)

    c=cosine_similarity(sum_h_r,sum_b_r)
    logging.debug(" cosine:"+str(c[0][0]))

    logging.info(" size of vector for headline is ")
    logging.info(str((sum_h.shape)))
    logging.info(" size vector for body is ")
    logging.info(str((sum_b.shape)))

    features=[c[0][0]]
    return features


#Of all the numbers/digits are mentioned in headline. how many are mentioned in body
def num_overlap_features(lemmatized_headline_split, headline_pos_split, lemmatized_body_split, body_pos_split, pos_in):

        logging.info("inside " + pos_in + " features")
        logging.info("lemmatized_headline_split " +str(lemmatized_headline_split))
        logging.info("lemmatized_headline_split " + str(lemmatized_body_split))
        h_numbers = []
        b_numbers = []
        h_nouns_antonyms=[]
        b_nouns_antonyms = []
        overall=False
        hc=False
        bc=False

        count_headline = 0
        for word1, pos in zip(lemmatized_headline_split, headline_pos_split):
            logging.debug(str("pos:") + str((pos)))
            logging.debug(str("word:")  + str((word1)))
            if pos.startswith(pos_in):
                overall=True
                hc=True
                logging.debug("pos.startswith:"+str(pos_in))
                count_headline = count_headline + 1
                h_numbers.append(word1)


        count_body = 0
        for word2, pos in zip(lemmatized_body_split, body_pos_split):
            logging.debug(str("pos:") + str((pos)))
            logging.debug(str("word:") + str((word2)))
            if pos.startswith(pos_in):
                overall=True
                bc=True
                count_body = count_body + 1
                b_numbers.append(word2)




        overlap_intersection = set(h_numbers).intersection(set(b_numbers))
        overlap_diff = set(h_numbers).difference(set(b_numbers))


        features = [len(overlap_intersection), len(overlap_diff)]
        logging.debug(str("features_ant:") + str((features)))



        return features



def get_ant(word):
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return antonyms

