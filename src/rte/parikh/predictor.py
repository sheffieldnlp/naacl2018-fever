from typing import List

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors import Predictor

from common.util.random import SimpleRandom


@Predictor.register('drwiki-te')
class TextualEntailmentPredictor(Predictor):
    @overrides
    def _batch_json_to_instances(self, json: List[JsonDict]) -> List[Instance]:
        instances = []
        for blob in json:
            instances.extend(self._json_to_instances(blob))
        return instances

    def set_docdb(self,db):
        self.db = db


    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1]
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]


    def _json_to_instances(self,json):
        hypothesis_text = json["claim"]
        instances = []


        premise_texts = []

        flattened_evidence = [evidence for evidence_group in json["evidence"] for evidence in evidence_group]
        for _,_, page,sentence in flattened_evidence:
            premise_texts = self.get_doc_line(page,sentence)

        instances.append(self._dataset_reader.text_to_instance(" ".join(premise_texts), hypothesis_text))
        return instances



@Predictor.register('drwiki-te-pred-all')
class TextualEntailmentPredictor(Predictor):
    @overrides
    def _batch_json_to_instances(self, json: List[JsonDict]) -> List[Instance]:
        instances = []
        for blob in json:
            instances.extend(self._json_to_instances(blob))
        return instances

    def set_docdb(self,db):
        self.db = db

    def _json_to_instances(self,json):
        hypothesis_text = json["claim"]
        instances = []
        premise_texts = []
        for page,score in json["predicted_pages"]:
            premise_texts.append(self.db.get_doc_text(page))

        instances.append(self._dataset_reader.text_to_instance(" ".join(premise_texts), hypothesis_text))
        return instances
