#Predict pages for each data split using DrQA

#Top 1 page
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/batch_ir.py  --model data/fever/drqatfidf/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/batch_ir.py  --model data/fever/drqatfidf/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split test

#Top 5 pages
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/batch_ir.py  --model data/fever/drqatfidf/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/batch_ir.py  --model data/fever/drqatfidf/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split test

#Top 100 pages
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/batch_ir.py  --model data/fever/drqatfidf/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 100 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/batch_ir.py  --model data/fever/drqatfidf/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 100 --split test


#Sample random pages for training not enough info
PYTHONPATH=src:lib/DrQA python src/scripts/dataset/neg_sample_evidence.py

#Select top 5 pages for training not enough info
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/batch_ir_ns.py  --model data/fever/drqatfidf/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split train
