rm -rf logs/
PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/da_nn_sent --cuda-device $CUDA_DEVICE
mkdir -p data/models
cp logs/da_nn_sent/model.tar.gz data/models/decomposable_attention.tar.gz
PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/dev.ns.pages.p1.small.jsonl




