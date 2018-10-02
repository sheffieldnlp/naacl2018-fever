from typing import Dict
import json,os
import logging

from overrides import overrides
import tqdm
import sys

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
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from retrieval.read_claims import UOFADataReader
from rte.riedel.data import FEVERPredictions2Formatter, FEVERLabelSchema, FEVERGoldFormatter
from common.dataset.data_set import DataSet as FEVERDataSet
from rte.mithun.trainer import UofaTrainTest

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
                 sentence_level = False,
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 filtering: str = None) -> None:
        self._sentence_level = sentence_level
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.db = db

        self.formatter = FEVERGoldFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema(),filtering=filtering)
        self.reader = JSONLineReader()


    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1]
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]

    @overrides
    def read(self, file_path: str, run_name,do_annotation):

        instances = []

        ds = FEVERDataSet(file_path,reader=self.reader, formatter=self.formatter)
        ds.read()
        counter=0

        objUOFADataReader = UOFADataReader()

        if (run_name == "train"):
            head_file = objUOFADataReader.ann_head_tr
            body_file = objUOFADataReader.ann_body_tr
        else:
            if (run_name == "dev"):
                head_file = objUOFADataReader.ann_head_dev
                body_file = objUOFADataReader.ann_body_dev


        if(do_annotation):
          # DELETE THE annotated file IF IT EXISTS every time before the loop
            self.delete_if_exists(head_file)
            self.delete_if_exists(body_file)

            for instance in tqdm.tqdm(ds.data):
                counter=counter+1
                if instance is None:
                    continue

                if not self._sentence_level:
                    pages = set(ev[0] for ev in instance["evidence"])
                    premise = " ".join([self.db.get_doc_text(p) for p in pages])
                else:
                    lines = set([self.get_doc_line(d[0],d[1]) for d in instance['evidence']])
                    premise = " ".join(lines)

                if len(premise.strip()) == 0:
                    premise = ""

                hypothesis = instance["claim"]
                label = instance["label_text"]
                #replacing hypothesis with the annotated one-either load pre-nnotated data
                # from disk or do live annotation (Takes more time)


                premise_ann,hypothesis_ann =self.uofa_annotate(hypothesis, premise, counter,objUOFADataReader,head_file,body_file)


                premise= " ".join(premise_ann)
                hypothesis = " ".join(hypothesis_ann)


                instances.append(self.text_to_instance(premise, hypothesis, label))

        else:
            # load it from the disk
            print("going to load annotated data from the disk")
            premise_ann, hypothesis_ann = self.uofa_load_ann_disk(objUOFADataReader,run_name)


            #for each line in the annotated data:
            #premise_ann, hypothesis_ann = self.uofa_load_ann_disk(self, objUOFADataReader, head_file, body_file)

            premise = " ".join(premise_ann)
            hypothesis = " ".join(hypothesis_ann)

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
        premise_tokens = self._wiki_tokenizer.tokenize(premise) if premise is not None else None
        hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers) if premise is not None else None
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)



    def uofa_load_ann_disk(self,objUOFADataReader,run_name):

        objUofaTrainTest = UofaTrainTest()

        if (run_name == "dev"):
            data_folder = objUofaTrainTest.data_folder_dev
        else:
            if (run_name == "train"):
                data_folder = objUofaTrainTest.data_folder_train
            else:
                if (run_name == "small"):
                    data_folder = objUofaTrainTest.data_folder_train_small
                else:
                    if (run_name== "test"):
                        data_folder = objUofaTrainTest.data_folder_test

        bf = data_folder + objUofaTrainTest.annotated_body_split_folder
        bfl = bf + objUofaTrainTest.annotated_only_lemmas
        bfw = bf + objUofaTrainTest.annotated_words
        bfe=bf+objUofaTrainTest.annotated_only_entities

        hf = data_folder + objUofaTrainTest.annotated_head_split_folder
        hfl = hf + objUofaTrainTest.annotated_only_lemmas
        hfw = hf + objUofaTrainTest.annotated_words
        hfe = hf + objUofaTrainTest.annotated_only_entities

        print(f"hfl:{hfl}")
        print(f"bfl:{bfl}")
        logging.info("going to read heads_lemmas from disk:")

        hl = objUofaTrainTest.read_json(hfl)
        bl = objUofaTrainTest.read_json(bfl)
        he = objUofaTrainTest.read_json(hfe)
        be = objUofaTrainTest.read_json(bfe)
        hw = objUofaTrainTest.read_json(hfw)
        bw = objUofaTrainTest.read_json(bfw)

        print(f"hl:{hl}")
        print(f"bl:{bl}")

        premise, hyp = objUofaTrainTest.convert_NER_form_per_sent(he, be, hl, bl, hw, bw)
        print(f'premise:{premise}')
        print(f'hyp:{hyp}')
        sys.exit(1)



        # print(premise,hyp)
        return premise, hyp

    def uofa_annotate(self, claim, evidence, index,objUOFADataReader,head_file,body_file):
        doc1,doc2 = objUOFADataReader.annotate_and_save_doc\
            (claim, evidence, index, objUOFADataReader.API,head_file,body_file,logger)

        he=doc1._entities
        hl=doc1.lemmas
        hw=doc1.words
        be = doc2._entities
        bl = doc2.lemmas
        bw = doc2.words
        objUofaTrainTest=UofaTrainTest()
        # print(f'{he}{hl}{hw}{be}{bl}{bw}')
        premise, hyp= objUofaTrainTest.convert_NER_form_per_sent( he, be, hl, bl, hw, bw)
        # print(premise,hyp)
        return premise,hyp

    def delete_if_exists(self, name):

        if os.path.exists(name):
            append_write = 'w'  # make a new file if not
            with open(name, append_write) as outfile:
                outfile.write("")

    @classmethod
    def from_params(cls, params: Params) -> 'FEVERReader':
        claim_tokenizer = Tokenizer.from_params(params.pop('claim_tokenizer', {}))
        wiki_tokenizer = Tokenizer.from_params(params.pop('wiki_tokenizer', {}))

        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        sentence_level = params.pop("sentence_level",False)
        db = FeverDocDB(params.pop("db_path","data/fever.db"))
        params.assert_empty(cls.__name__)
        return FEVERReader(db=db,
                           sentence_level=sentence_level,
                           claim_tokenizer=claim_tokenizer,
                           wiki_tokenizer=wiki_tokenizer,
                           token_indexers=token_indexers)

