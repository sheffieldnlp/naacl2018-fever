from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import hstack

from common.features.feature_function import FeatureFunction
from common.util.array import flatten

import numpy as np

class TermFrequencyFeatureFunction(FeatureFunction):

    stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]


    def __init__(self,doc_db,lim_unigram=5000,gold=True):
        super().__init__()
        self.doc_db = doc_db
        self.lim_unigram = lim_unigram

        if gold:
            self.ename = "evidence"
        else:
            self.ename = "predicted"



    def inform(self,train,dev=None,test=None):
        claims = self.claims(train)
        bodies = self.bodies(train)

        if dev is not None:
            dev_claims = self.claims(dev)
            dev_bodies = self.bodies(dev)
        else:
            dev_claims = []
            dev_bodies = []

        if test is not None:
            test_claims = self.claims(test)
            test_bodies = self.bodies(test)
        else:
            test_claims = []
            test_bodies = []

        self.bow_vectorizer = CountVectorizer(max_features=self.lim_unigram,
                                         stop_words=TermFrequencyFeatureFunction.stop_words)
        self.bow = self.bow_vectorizer.fit_transform(claims + bodies)

        self.tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(self.bow)

        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.lim_unigram,
                                           stop_words=TermFrequencyFeatureFunction.stop_words). \
            fit(claims + bodies + dev_claims + dev_bodies + test_claims + test_bodies)


    def lookup(self,data):
        size = len(data)
        return self.process(data)

    def process(self,data):
        claim_bow = self.bow_vectorizer.transform(self.claims(data))
        claim_tfs = self.tfreq_vectorizer.transform(claim_bow)
        claim_tfidf = self.tfidf_vectorizer.transform(self.claims(data))

        body_texts = self.texts(data)
        body_bow = self.bow_vectorizer.transform(body_texts)
        body_tfs = self.tfreq_vectorizer.transform(body_bow)
        body_tfidf = self.tfidf_vectorizer.transform(body_texts)

        cosines = np.array([cosine_similarity(c, b)[0] for c,b in zip(claim_tfidf,body_tfidf)])

        return hstack([body_tfs,claim_tfs,cosines])


    def claims(self,data):
        return [datum["claim"] for datum in data]

    def bodies(self,data):
        return [self.doc_db.get_doc_text(id) for id in set(flatten(self.body_ids(data)))]

    def texts(self,data):
        return [" ".join([self.doc_db.get_doc_text(page) for page in instance]) for instance in self.body_ids(data)]


    def body_ids(self,data):
        return [datum[self.ename] for datum in data]




