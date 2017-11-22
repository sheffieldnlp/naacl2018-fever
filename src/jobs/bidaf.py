import argparse
from contextlib import ExitStack
import json
from typing import Optional, IO, Dict, List

from overrides import overrides

from allennlp.common import Params
from allennlp.data import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.models.archival import Archive, load_archive
from allennlp.service.predictors import Predictor


@WordSplitter.register('indexed_spaces')
class JustSpacesWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.  We use a somewhat odd name here to avoid coming too close to the more
    commonly used ``SpacyWordSplitter``.

    Note that we use ``sentence.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        tokens = [Token(text=t,idx=0) for t in sentence.split()]
        for id,token in enumerate(tokens):
            if id == 0:
                continue
            token.idx = tokens[id-1].idx + len(tokens[id-1].text) + 1
        return tokens
    @classmethod
    def from_params(cls, params: Params) -> 'WordSplitter':
        params.assert_empty(cls.__name__)
        return cls()

def _run(predictor: Predictor,
         input_file: IO,
         output_file: Optional[IO],
         batch_size: int,
         cuda_device: int) -> None:

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            result = predictor.predict_json(batch_data[0], cuda_device)
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data, cuda_device)

        for model_input, output in zip(batch_data, results):
            string_output = json.dumps(output)

            print("input: ", model_input)
            print("prediction: ", string_output)

            if output_file:
                output_file.write(string_output + "\n")

    batch_json_data = []
    for line in input_file:
        if not line.isspace():
            # Collect batch size amount of data.
            json_data = json.loads(line)
            batch_json_data.append(json_data)
            if len(batch_json_data) == batch_size:
                _run_predictor(batch_json_data)
                batch_json_data = []

    # We might not have a dataset perfectly divisible by the batch size,
    # so tidy up the scraps.
    if batch_json_data:
        _run_predictor(batch_json_data)


def predict(args: argparse.Namespace) -> None:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device, overrides=args.overrides)
    predictor = Predictor.from_archive(archive, "machine-comprehension")



    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)  # type: ignore
        output_file = stack.enter_context(args.output_file)  # type: ignore

        _run(predictor, input_file, output_file, args.batch_size, args.cuda_device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    parser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')
    parser.add_argument('output_file', type=argparse.FileType('w'), help='path to output file')

    batch_size = parser.add_mutually_exclusive_group(required=False)
    batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')
    batch_size.add_argument('--batch_size', type=int, help=argparse.SUPPRESS)

    cuda_device = parser.add_mutually_exclusive_group(required=False)
    cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    cuda_device.add_argument('--cuda_device', type=int, help=argparse.SUPPRESS)

    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')

    parser.set_defaults(func=predict)
    predict(parser.parse_args())