#Predict pages for each data split using DrQA


PYTHONPATH=src:lib/DrQA python src/scripts/dataset/neg_sample_evidence.py > logs/ir.nse &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir_ns.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split train > logs/ir.ns.p5.train &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir_ns.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split train > logs/ir.ns.p1.train &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir_ns.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split dev > logs/ir.ns.p5.dev &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir_ns.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev > logs/ir.ns.p1.dev &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir_ns.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split test > logs/ir.ns.p5.test &
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/batch_ir_ns.py  --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split test > logs/ir.ns.p1.test &


