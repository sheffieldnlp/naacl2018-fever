PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev
PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split dev
PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 10 --split dev
PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 25 --split dev
PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 50 --split dev
PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 100 --split dev
PYTHONPATH=src python src/scripts/retrieval/document/eval_recall.py --split dev --count 1
PYTHONPATH=src python src/scripts/retrieval/document/eval_recall.py --split dev --count 5
PYTHONPATH=src python src/scripts/retrieval/document/eval_recall.py --split dev --count 10
PYTHONPATH=src python src/scripts/retrieval/document/eval_recall.py --split dev --count 25
PYTHONPATH=src python src/scripts/retrieval/document/eval_recall.py --split dev --count 50
PYTHONPATH=src python src/scripts/retrieval/document/eval_recall.py --split dev --count 100
PYTHONPATH=src python src/scripts/retrieval/document/eval_oracle.py --split dev --count 1
PYTHONPATH=src python src/scripts/retrieval/document/eval_oracle.py --split dev --count 5
PYTHONPATH=src python src/scripts/retrieval/document/eval_oracle.py --split dev --count 10
PYTHONPATH=src python src/scripts/retrieval/document/eval_oracle.py --split dev --count 25
PYTHONPATH=src python src/scripts/retrieval/document/eval_oracle.py --split dev --count 50
PYTHONPATH=src python src/scripts/retrieval/document/eval_oracle.py --split dev --count 100