import argparse
import json
from contextlib import ExitStack
from typing import Optional, IO

import numpy as np
from tqdm import tqdm
from rte.parikh.predictor import TextualEntailmentPredictor
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from retrieval.fever_doc_db import FeverDocDB


def _run(predictor: Predictor,
         input_file: IO,
         output_file: Optional[IO],
         cuda_device: int) -> None:
    correct = 0
    total = 0

    def _run_predictor(item):
        results = predictor.predict_batch_json(item, cuda_device)
        model_input = item[0]

        global correct
        global total

        vers = ["SUPPORTS","REFUTES", "NOT ENOUGH INFO"]
        a = vers[np.argmax(results[0]['label_logits'])]
        model_input["predicted"] = a
        output_file.write(json.dumps(model_input) + "\n")

        if a == model_input["label"]:
            correct += 1
        total +=1


    batch_json_data = []
    for line in tqdm(input_file):
        if not line.isspace():
            # Collect batch size amount of data.
            json_data = json.loads(line)
            _run_predictor([json_data])

    print (correct/total)

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