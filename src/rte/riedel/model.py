from common.dataset import JSONLineReader
from common.features.feature_function import Features
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.features import TermFrequencyFeatureFunction

if __name__ == "__main__":
    db = FeverDocDB("data/fever/drqa.db")
    f = Features([TermFrequencyFeatureFunction(db)])


    f.inform([{"claim":"This is a claim about japan",
                  "evidence":["Japan","Pakistan"]
                  },
                 {"claim": "This is a claim about Estonia",
                  "evidence": ["Estonia"]
                  }
                 ])

    print(f.load([{"claim": "This is a claim about japan",
                  "evidence": ["Japan", "Pakistan"],
                   "label":"B"
                  },{"claim": "This is a claim about Estonia",
                  "evidence": ["Estonia"],
                     "label":"A"
                  },{"claim": "This is a second claim about the French Republic France France",
                  "evidence": ["France"],
                     "label": "H"

                  }]))


    jlr = JSONLineReader()
    formatter = TextAnnotationFormatter(FEVERLabelSchema())

    DataSet(file=sexism_file, reader=jlr, formatter=formatter),