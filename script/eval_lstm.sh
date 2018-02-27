FILES=`ls data/ablation/* | xargs -n1 basename`
for f in $FILES;
do
    echo "Eval $f";
    export GPU=1
    export CUDA_DEVICE=0
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db logs/ablation/$f/model.tar.gz data/fever/dev.ns.pages.p1.jsonl
done
