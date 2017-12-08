#Predict pages for each data split using DrQA

#Top 1 page
echo "k=1"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_mrr.py --count 1 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_macro.py --count 1 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_oracle.py --count 1 --split dev

echo "k=5"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_mrr.py --count 5 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_macro.py --count 5 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_oracle.py --count 5 --split dev

echo "k=10"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_mrr.py --count 10 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_macro.py --count 10 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_oracle.py --count 10 --split dev

echo "k=100"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_mrr.py --count 100 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_macro.py --count 100 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/eval_p_oracle.py --count 100 --split dev
