import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.models import Model, archive_model, load_archive
from allennlp.service.predictors import Predictor
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader

import argparse
import logging
import sys
import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def eval_model(db: FeverDocDB, args) -> Model:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device, overrides=args.overrides)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()

    reader = FEVERReader(db,
                                 sentence_level=ds_params.pop("sentence_level",False),
                                 wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                                 claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                                 token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

    logger.info("Reading training data from %s", args.in_file)
    data = reader.read(args.in_file)


    for item in data:
        print(model.forward_on_instance(item, args.cuda_device))
        

    return model



if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=argparse.FileType('r'), help='/path/to/saved/db.db')


    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')



    args = parser.parse_args()

    db = FeverDocDB(args.db)

    params = Params.from_file(args.param_path)
    eval_model(db,params,args.cuda_device,args.logdir)