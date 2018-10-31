PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/train.jsonl --out-file data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode train --lmode WARNING

PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/dev.jsonl --out-file data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode dev --lmode INFO


