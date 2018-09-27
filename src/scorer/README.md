# FEVER Scorer

[![Build Status](https://travis-ci.org/sheffieldnlp/fever-scorer.svg?branch=master)](https://travis-ci.org/sheffieldnlp/fever-scorer)

Scoring function for the Fact Extraction and VERification shared task. Tested for Python 3.6 and 2.7.

This scorer produces five outputs: 
 * The strict score considering the requirement for evidence (**primary scoring metric for shared task**)
 * The label accuracy
 * The macro-precision of the evidence for supported/refuted claims
 * The macro-recall of the evidence supported/refuted claims where an instance is scored if and only if at least one complete evidence group is found
 * The F1 score of the evidence, using the above metrics.
 
The evidence is considered to be correct if there exists a complete list of actual evidence that is a subset of the predicted evidence.

In the FEVER Shared Task, we will consider only only the first `5` sentences of predicted_evidence that the candidate system provies for scoring. This is configurable through the `max_evidence` parameter for the scorer. When too much evidence is provided. It is removed, without penalty.

## Find out more

Visit [http://fever.ai](http://fever.ai) to find out more about the shared task.

## Example 1
```python
from fever.scorer import fever_score

instance1 = {"label": "REFUTES", "predicted_label": "REFUTES", "predicted_evidence": [ #is not strictly correct - missing (page2,2)
        ["page1", 1]                                    #page name, line number
    ], 
    "evidence":
    [
        [
            [None, None, "page1", 1],           #[(ignored) annotation job, (ignored) internal id, page name, line number]
            [None, None, "page2", 2],
        ]
    ]
}

instance2 = {"label": "REFUTES", "predicted_label": "REFUTES", "predicted_evidence": [
        ["page1", 1],                                   
        ["page2", 2],
        ["page3", 3]                                    
    ], 
    "evidence":
    [
        [
            [None, None, "page1", 1],   
            [None, None, "page2", 2],
        ]
    ]
}

predictions = [instance1, instance2]
strict_score, label_accuracy, precision, recall, f1 = fever_score(predictions)

print(strict_score)     #0.5
print(label_accuracy)   #1.0
print(precision)        #0.833 (first example scores 1, second example scores 2/3)
print(recall)           #0.5 (first example scores 0, second example scores 1)
print(f1)               #0.625 
```



## Example 2 - (e.g. blind test set)
```python
from fever.scorer import fever_score

instance1 = {"predicted_label": "REFUTES", "predicted_evidence": [ #is not strictly correct - missing (page2,2)
    ["page1", 1]                                    #page name, line number
]}

instance2 = {"predicted_label": "REFUTES", "predicted_evidence": [
    ["page1", 1],                                   #page name, line number
    ["page2", 2],
    ["page3", 3]
]}

actual = [
    {"label": "REFUTES", "evidence":
        [
            [
                [None, None, "page1", 1],
                [None, None, "page2", 2],
            ]
        ]},
    {"label": "REFUTES", "evidence":
        [
            [
                [None, None, "page1", 1],
                [None, None, "page2", 2],
            ]
        ]}
]

predictions = [instance1, instance2]
strict_score, label_accuracy, precision, recall, f1 = fever_score(predictions,actual)

```
