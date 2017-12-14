from common.util.array import flatten
from rte.riedel.fever_features import TermFrequencyFeatureFunction


class FeverOrFNCTermFrequencyFeatureFunction(TermFrequencyFeatureFunction):

    def __init__(self,fever_db,fnc_db,lim_unigram=5000):
        self.fnc_db = fnc_db
        self.fever_db =fever_db
        super().__init__(fever_db,lim_unigram)
        self.ename = "evidence"

    def bodies(self,data):
        ret = []
        for datum in data:
            if isinstance(datum[self.ename], list):
                ret.extend([self.fever_db.get_doc_text(id) for id in set(flatten(self.body_ids([datum])))])
            else:
                ret.extend([self.fnc_db.get_doc_text(id) for id in set(self.body_id([datum]))])
        return list(set(ret))

    def body_id(self,data):
        return [datum[self.ename] for datum in data]

    def texts(self,data):
        ret = []
        for datum in data:
            if isinstance(datum[self.ename], list):
                ret.extend([" ".join([self.fever_db.get_doc_text(page) for page in instance]) for instance in self.body_ids([datum])])
            else:
                ret.extend([self.fnc_db.get_doc_text(id) for id in set(self.body_id([datum]))])
        return ret



    def body_ids(self,data):
        return [datum[self.ename] for datum in data]
