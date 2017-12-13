from typing import Dict, List, Tuple
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter
from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB
from retrieval.sentence import FEVERSentenceFormatter
from rte.riedel.data import FEVERLabelSchema, FeverFormatter, preprocess
from common.dataset.data_set import DataSet as FEVERDataSet
from allennlp.data.dataset_readers.reading_comprehension import util
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name





@DatasetReader.register("fever-sentence")
class FEVERSentenceReader(DatasetReader):
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
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.db = db

        self.formatter = FEVERSentenceFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema())
        self.reader = JSONLineReader()


    @overrides
    def read(self, file_path: str):

        instances = []

        ds = FEVERDataSet(file_path,reader=self.reader, formatter=self.formatter)
        ds.read()

        for instance in tqdm.tqdm(ds.data):
            if instance is None:
                continue

            for page in set([ev[0] for ev in instance['evidence']]):
                claim = instance['claim'].strip()
                paragraph = self.db.get_doc_text(page)
                tokenized_paragraph = self._wiki_tokenizer.tokenize(paragraph)

                evidences = set([ev[1] for ev in instance['evidence'] if ev[0] == page])


                lines = self.db.get_doc_lines(page)
                if any(ev<0 for ev in evidences):
                    span_ends = [0]
                    span_starts = [0]
                    evidence_texts = [""]

                else:
                    evidence_texts = [lines.split("\n")[line].split("\t")[1] for line in evidences]

                    span_starts = [paragraph.index(evidence_text) for evidence_text in evidence_texts]
                    span_ends =  [start + len(evidence_texts) for start, evidence_text in zip(span_starts, evidence_texts)]
                inst = self.text_to_instance(claim,
                                                 paragraph,
                                                 zip(span_starts, span_ends),
                                                 evidence_texts,
                                                 tokenized_paragraph)
                instances.append(inst)
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
        premise_tokens = self._wiki_tokenizer.tokenize(premise)
        hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'FEVERSentenceReader':
        claim_tokenizer = Tokenizer.from_params(params.pop('claim_tokenizer', {}))
        wiki_tokenizer = Tokenizer.from_params(params.pop('wiki_tokenizer', {}))

        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        db = FeverDocDB(params.pop("db_path","data/fever/fever.db"))
        params.assert_empty(cls.__name__)
        return FEVERSentenceReader(db=db,
                           claim_tokenizer=claim_tokenizer,
                           wiki_tokenizer=wiki_tokenizer,
                           token_indexers=token_indexers)


    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return util.make_reading_comprehension_instance(self._claim_tokenizer.tokenize(question_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts)

