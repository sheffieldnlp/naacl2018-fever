FILES=`ls data/ablation/* | xargs -n1 basename`
for f in $FILES;
do
    echo "Train $f";
    export GPU=1
    export CUDA_DEVICE=0
    RANDOM_SEED=1234 PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/ablation/$f data/fever/dev.ns.pages.p1.jsonl --model ablation$f --sentence true
    RANDOM_SEED=2345 PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/ablation/$f data/fever/dev.ns.pages.p1.jsonl --model ablation$f --sentence true
    RANDOM_SEED=3456 PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/ablation/$f data/fever/dev.ns.pages.p1.jsonl --model ablation$f --sentence true
    RANDOM_SEED=4567 PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/ablation/$f data/fever/dev.ns.pages.p1.jsonl --model ablation$f --sentence true
    RANDOM_SEED=5678 PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/ablation/$f data/fever/dev.ns.pages.p1.jsonl --model ablation$f --sentence true
done
