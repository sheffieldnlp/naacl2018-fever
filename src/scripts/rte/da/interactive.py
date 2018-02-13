from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from drqa import retriever
from allennlp.data import Tokenizer, TokenIndexer
from allennlp.models import Model, load_archive
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader

import argparse
import numpy as np


from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from scripts.retrieval.sentence.process_tfidf import XTermFrequencyFeatureFunction

LogHelper.setup()
logger = LogHelper.get_logger(__name__)  # pylint: disable=invalid-name


def tf_idf_sim(claim, lines):
    test = []
    for line in lines:
        test.append({"claim": claim, "text": line})

    return tf.lookup(test).reshape(-1).tolist()

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


    while True:

        claim = input("enter claim (or q to quit) >>")
        if claim.lower() == "q":
            break

        ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

        p_lines = []
        pages,_ = ranker.closest_docs(claim,5)

        for page in pages:
            lines = db.get_doc_lines(page)
            lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in lines.split("\n")]

            p_lines.extend(zip(lines, [page] * len(lines), range(len(lines))))

        scores = tf_idf_sim(claim, [pl[0] for pl in p_lines])
        scores = list(zip(scores, [pl[1] for pl in p_lines], [pl[2] for pl in p_lines], [pl[0] for pl in p_lines]))
        scores = list(filter(lambda score: len(score[3].strip()), scores))
        sentences_l = list(sorted(scores, reverse=True, key=lambda elem: elem[0]))

        sentences = [s[3] for s in sentences_l[:5]]
        evidence = " ".join(sentences)


        print("Best pages: {0}".format(repr(pages)))

        print("Evidence:")
        for idx,sentence in enumerate(sentences_l[:5]):
            print("{0}\t{1}\t\t{2}\t{3}".format(idx+1, sentence[0], sentence[1],sentence[3]) )

        item = reader.text_to_instance(evidence, claim)

        prediction = model.forward_on_instance(item, args.cuda_device)
        cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]
        print("PREDICTED: {0}".format(cls))
        print()


if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    logger = LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('archive_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument("--model",type=str,help="model")
    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')



    args = parser.parse_args()

    logger.info("Load DB")
    db = FeverDocDB(args.db)

    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    logger.info("Read datasets")
    train_ds = DataSet(file="data/fever/train.ns.pages.p{0}.jsonl".format(1), reader=jlr, formatter=formatter)
    dev_ds = DataSet(file="data/fever/dev.ns.pages.p{0}.jsonl".format(1), reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()

    logger.info("Generate vocab for TF-IDF")
    tf = XTermFrequencyFeatureFunction(db)
    tf.inform(train_ds.data, dev_ds.data)

    logger.info("Eval")
    eval_model(db,args)