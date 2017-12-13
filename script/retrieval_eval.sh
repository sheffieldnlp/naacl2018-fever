#Predict pages for each data split using DrQA

#Top 1 page
echo "k=1"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall.py --count 1 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall_all.py --count 1 --split dev

echo "k=5"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall.py --count 5 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall_all.py --count 5 --split dev

echo "k=10"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall.py --count 10 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall_all.py --count 10 --split dev

echo "k=25"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall.py --count 25 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall_all.py --count 25 --split dev

echo "k=100"
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall.py --count 100 --split dev
PYTHONPATH=src:lib/DrQA python src/scripts/retrieval/document/eval_recall_all.py --count 100 --split dev
