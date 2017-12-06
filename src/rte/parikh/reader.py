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
from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERPredictions2Formatter, FEVERLabelSchema
from common.dataset.data_set import DataSet as FEVERDataSet
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fever")
class FEVERReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 db: FeverDocDB,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.db = db

        self.formatter = FEVERPredictions2Formatter(set(self.db.get_doc_ids()), FEVERLabelSchema())
        self.reader = JSONLineReader()


    @overrides
    def read(self, file_path: str):

        instances = []


        ds = FEVERDataSet(file_path,reader=self.reader, formatter=self.formatter)
        ds.read()

        for instance in tqdm.tqdm(ds.data):
            if instance is None:
                continue

            premise = " ".join([self.db.get_doc_text(ev) for ev in set(instance["evidence"])])
            hypothesis = instance["claim"]
            label = instance["label"]
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
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'FEVERReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        db = FeverDocDB(params.pop("db_path","data/fever.db"))
        params.assert_empty(cls.__name__)
        return FEVERReader(db=db,
                           tokenizer=tokenizer,
                           token_indexers=token_indexers)
