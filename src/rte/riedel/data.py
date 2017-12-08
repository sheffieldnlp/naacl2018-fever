from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from nltk import word_tokenize

def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]

class FeverFormatter(Formatter):
    def __init__(self, index, label_schema, tokenizer=None):
        super().__init__(label_schema)
        self.index=index
        self.tokenize = tokenizer if tokenizer is not None else self.nltk_tokenizer

    def nltk_tokenizer(self,text):
        return " ".join(word_tokenize(text))

class FEVERGoldFormatter(FeverFormatter):
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        pages = [preprocess(ev[1]) for ev in line["evidence"]]
        return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERPredictionsFormatter(FeverFormatter):
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        pages = [preprocess(ev[0]) for ev in line["predicted_pages"]]
        return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERPredictions2Formatter(FeverFormatter):
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

        return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["supports","refutes","not enough info"])
