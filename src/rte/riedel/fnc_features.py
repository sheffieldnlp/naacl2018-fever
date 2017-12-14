from rte.riedel.fever_features import TermFrequencyFeatureFunction


class FNCTermFrequencyFeatureFunction(TermFrequencyFeatureFunction):

    def __init__(self,doc_db,lim_unigram=5000):
        super().__init__(doc_db,lim_unigram)
        self.ename = "evidence"

    def bodies(self,data):
        return [self.doc_db.get_doc_text(id) for id in set(self.body_id(data))]

    def texts(self,data):
        return [self.doc_db.get_doc_text(id) for id in self.body_id(data)]

    def body_id(self,data):
        return [datum[self.ename] for datum in data]


