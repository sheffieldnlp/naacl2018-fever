from typing import Dict
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERPredictions2Formatter, FEVERLabelSchema, FEVERGoldFormatter
from common.dataset.data_set import DataSet as MultiNLIDataset
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MultiNLIFormatter(Formatter):
    def format_line(self,line):

        if line["gold_label"] in ["entailment","contradiction","neutral"]:
            return {"premise":line["sentence1"], "hypothesis":line["sentence2"], "label_text":line["gold_label"]}
        return None

class MultiNLILabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["entail","contradict","neutral"])


@DatasetReader.register("multinli")
class MultiNLIReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.formatter = MultiNLIFormatter(MultiNLILabelSchema())
        self.reader = JSONLineReader()


    @overrides
    def read(self, file_path: str):
        instances = []
        ds = MultiNLIDataset(file_path,reader=self.reader, formatter=self.formatter)
        ds.read()

        for instance in tqdm.tqdm(ds.data):
            hypothesis = instance["hypothesis"]
            premise = instance["premise"]
            label = instance["label_text"]
            instances.append(self.text_to_instance(premise, hypothesis, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


    @classmethod
    def from_params(cls, params: Params) -> 'MultiNLIReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return MultiNLIReader(tokenizer=tokenizer,
                              token_indexers=token_indexers)

