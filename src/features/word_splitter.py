from typing import Optional, IO, Dict, List
from overrides import overrides
from allennlp.common import Params
from allennlp.data import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter

@WordSplitter.register('indexed_spaces')
class IndexedSpaces(WordSplitter):
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