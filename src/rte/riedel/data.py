import importlib.util
import os

from nltk import word_tokenize

from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from retrieval.filter_uninformative import uninformative


def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]

class FeverFormatter(Formatter):
    def __init__(self, index, label_schema, tokenizer=None,filtering=None):
        super().__init__(label_schema)
        self.index=index
        self.tokenize = tokenizer if tokenizer is not None else self.nltk_tokenizer
        self.filtering = None

        def import_module(filename):
            spec = importlib.util.spec_from_file_location('filter_doc', filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        if filtering is not None:
            if filtering:
                self.filtering = import_module(filtering).preprocess

    def nltk_tokenizer(self,text):
        return " ".join(word_tokenize(text))

class FEVERGoldFormatter(FeverFormatter):
    def format_line(self,line):
        annotation = None
        if "label" in line:
            annotation = line["label"]
        pages = []

        if 'predicted_sentences' in line:
            pages.extend([(ev[0], ev[1]) for ev in line["predicted_sentences"]])
        elif 'predicted_pages' in line:
            pages.extend([(ev[0],-1) for ev in line["predicted_pages"]])
        else:
            for evidence_group in line["evidence"]:
                pages.extend([(ev[2],ev[3]) for ev in evidence_group])

        if self.filtering is not None:
            for page,_ in pages:
                if self.filtering({"id":page}) is None:
                    return None
        if annotation is not None:
            return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}
        else:
            return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":None,"label_text":None}
 

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
            pages = [ev[0] for ev in line["predicted_pages"]]

        elif 'evidence' in line:
            pages = [ev[1] for ev in line["evidence"]]

        else:
            pages = []


        return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["supports","refutes","not enough info"])


