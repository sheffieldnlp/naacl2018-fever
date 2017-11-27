from contextlib import ExitStack
from typing import Optional, IO, List

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from features.word_splitter import IndexedSpaces
from tqdm import tqdm
import numpy as np
import json
import argparse

from retrieval.fever_doc_db import FeverDocDB



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

    def _json_to_instances(self,json):
        hypothesis_text = json["claim"]
        instances = []
        for page,score in json["predicted_pages"]:
            premise_text = self.db.get_doc_text(page)
            instances.append(self._dataset_reader.text_to_instance(premise_text, hypothesis_text))
        return instances



def _run(predictor: Predictor,
         input_file: IO,
         output_file: Optional[IO],
         cuda_device: int) -> None:

    def _run_predictor(item):
        results = predictor.predict_batch_json(item, cuda_device)
        model_input = item[0]

        for idx, output in enumerate(results):
            vers = ["NEUTRAL","SUPPORTED", "REFUTED"]
            a = vers[np.argmax(output['label_logits'])]
            model_input["predicted_pages"][idx].append(a)
        output_file.write(json.dumps(model_input) + "\n")

    batch_json_data = []
    for line in tqdm(input_file):
        if not line.isspace():
            # Collect batch size amount of data.
            json_data = json.loads(line)
            _run_predictor([json_data])


def predict(args: argparse.Namespace,docdb) -> None:
    print(args.archive_file)
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device, overrides=args.overrides)
    predictor = Predictor.from_archive(archive, "drwiki-te")
    predictor.set_docdb(docdb)

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)  # type: ignore
        output_file = stack.enter_context(args.output_file)  # type: ignore

        _run(predictor, input_file, output_file, args.cuda_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')
    parser.add_argument('output_file', type=argparse.FileType('w'), help='path to output file')

    cuda_device = parser.add_mutually_exclusive_group(required=False)
    cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    cuda_device.add_argument('--cuda_device', type=int, help=argparse.SUPPRESS)


    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')


    args = parser.parse_args()
    parser.set_defaults(func=predict)
    db = FeverDocDB(args.db)
    predict(args,db)