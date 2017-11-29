from common.features.feature_function import Features
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.features import TermFrequencyFeatureFunction

def simple_preproc(text):
    return text.lower()

if __name__ == "__main__":
    db = FeverDocDB("data/fever/drqa.db")
    f = Features([TermFrequencyFeatureFunction(db)],preprocessing=simple_preproc)


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
