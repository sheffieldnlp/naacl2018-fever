#Predict pages for each data split using DrQA

#Top 1 page
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev > logs/ir.dev.1 &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split test > logs/ir.test.1 &

#Top 5 pages
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split dev > logs/ir.dev.5 &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split test > logs/ir.test.5 &

#Top 10 pages
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 10 --split dev > logs/ir.dev.10 &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 10 --split test > logs/ir.test.10 &

#Top 25 pages
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 25 --split dev > logs/ir.dev.25 &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 25 --split test > logs/ir.test.25 &

#Top 100 pages
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 100 --split dev > logs/ir.dev.100 &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 100 --split test > logs/ir.test.100 &

