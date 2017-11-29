
class LabelSchema:
    def __init__(self,labels):
        self.labels = {self.preprocess(val):idx for idx,val in enumerate(labels)}
        self.idx = {idx:self.preprocess(val) for idx,val in enumerate(labels)}

    def get_id(self,label):
        if self.preprocess(label) in self.labels:
            return self.labels[self.preprocess(label)]
        return None

    def preprocess(self,item):
        return item.lower()



class SNLILabelSchema(LabelSchema):
    def __init__(self):
        super(SNLILabelSchema, self).__init__(["neither","contradiction","entailment"])

