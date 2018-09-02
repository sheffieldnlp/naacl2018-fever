PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/train.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode train --lmode WARNING
PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/dev.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5  --mode dev --lmode WARNING



