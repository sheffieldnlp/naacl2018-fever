from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema


def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]



class FEVERGoldFormatter(Formatter):

    def __init__(self,index,label_schema):
        super().__init__(label_schema)
        self.index = index
    def format_line(self,line):
        annotation = line["verdict"]

        if not isinstance(line['evidence'][0],list):
            return None

        pages = [preprocess(ev[1]) for ev in line["evidence"]]

        if any(map(lambda p: p not in self.index, pages)):
            return None

        return {"claim":line["claim"], "evidence": pages, "label":self.label_schema.get_id(annotation)}


class FEVERPredictionsFormatter(Formatter):

    def __init__(self,index,label_schema):
        super().__init__(label_schema)
        self.index = index
    def format_line(self,line):
        annotation = line["verdict"]

        if not isinstance(line['predicted_pages'][0],list):
            return None

        if not isinstance(line['evidence'][0],list):
            return None


        pages = [preprocess(ev[0]) for ev in line["predicted_pages"]]

        if any(map(lambda p: p not in self.index, pages)):
            return None

        return {"claim":line["claim"], "evidence": pages, "label":self.label_schema.get_id(annotation)}



class FEVERLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["supported","refuted","not enough information"])
