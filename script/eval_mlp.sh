FILES=`ls data/ablation/* | xargs -n1 basename`
for f in $FILES;
do
    echo "Eval $f";
    export GPU=1
    export CUDA_DEVICE=0
    PYTHONPATH=src python src/scripts/rte/mlp/eval_mlp.py data/fever/fever.db data/fever/dev.ns.pages.p1.jsonl --model ablation$f --sentence true
done
