from common.util.array import flatten
from rte.riedel.data import FeverFormatter, preprocess, FEVERLabelSchema


class FEVERSentenceFormatter(FeverFormatter):
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        pages = []
        if 'evidence' in line:
            pages = [[(ev[2],ev[3]) for ev in annotation if ev[2] is not None] for annotation in line["evidence"]]

        return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERSentenceTextFormatter(FeverFormatter):
    def __init__(self,idx, db,ls):
        super().__init__(idx, ls)
        self.db = db

    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        return lines.split("\n")[line].split("\t")[1]

    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        newpages = []
        docs = []
        if 'evidence' in line:
            pages = set(flatten([[(ev[2],ev[3]) for ev in annotation if ev[2] is not None] for annotation in line["evidence"]]))
            docs = set(flatten([[ev[2] for ev in annotation if ev[2] is not None] for annotation in line["evidence"]]))

            for page in pages:
                newpages.append((page[0],page[1],self.get_doc_line(page[0],page[1])))

        return {"claim":self.tokenize(line["claim"]), "docs": docs, "evidence": newpages, "label":self.label_schema.get_id(annotation),"label_text":annotation}



class FEVERSentenceRelatednessFormatter(FeverFormatter):

    def __init__(self,idx, db,ls):
        super().__init__(idx, ls)
        self.label_schema = ls
        self.ols = FEVERLabelSchema()
        self.db = db

    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        if self.ols.get_id(annotation) != self.ols.get_id("not enough info"):
            annotation = "related"
        else:
            annotation = "unrelated"

        evidence_texts = []
        claim = self.tokenize(line['claim']).strip()
        for page in set([ev[2] for ev in line['evidence']]):
            evidences = set([ev[3] for ev in line['evidence'] if ev[1] == page])
            lines = self.db.get_doc_lines(page)
            if any(ev<0 for ev in evidences):
                evidence_texts = [""]
            else:
                evidence_texts = [lines.split("\n")[line].split("\t")[1].split() for line in evidences]

        return {"claim":claim, "sentences": evidence_texts, "label":self.label_schema.get_id(annotation),"label_text":annotation}



