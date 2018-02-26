FILES=`ls data/ablation/* | xargs -n1 basename`
for f in $FILES;
do
    echo "Train $f";
    export GPU=1
    export CUDA_DEVICE=0
    PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db $f data/fever/dev.ns.pages.p1.jsonl --model ablation$f --sentence true
done
