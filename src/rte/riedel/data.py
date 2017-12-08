from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema


def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]



class FEVERGoldFormatter(Formatter):

    def __init__(self,index,label_schema):
        super().__init__(label_schema)
        self.index = index
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]


        pages = [preprocess(ev[1]) for ev in line["evidence"]]
        return {"claim":line["claim"], "evidence": pages, "label":self.label_schema.get_id(annotation)}


class FEVERPredictionsFormatter(Formatter):

    def __init__(self,index,label_schema):
        super().__init__(label_schema)
        self.index = index
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]


        pages = [preprocess(ev[0]) for ev in line["predicted_pages"]]
        return {"claim":line["claim"], "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERPredictions2Formatter(Formatter):

    def __init__(self,index,label_schema):
        super().__init__(label_schema)
        self.index = index
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        if 'predicted_pages' in line:
            pages = [preprocess(ev[0]) for ev in line["predicted_pages"]]

        elif 'evidence' in line:
            pages = [preprocess(ev[1]) for ev in line["evidence"]]

        else:
            pages = []

        return {"claim":line["claim"], "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["supports","refutes","not enough info"])
