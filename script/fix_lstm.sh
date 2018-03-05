FILES=`ls data/ablation/* | xargs -n1 basename`
for f in $FILES;
do
    echo "Eval $f";
    export GPU=1
    export CUDA_DEVICE=0
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db logs/ablation/seed1-$f/model.tar.gz data/fever/dev.ns.pages.p1.jsonl
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db logs/ablation/seed2-$f/model.tar.gz data/fever/dev.ns.pages.p1.jsonl
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db logs/ablation/seed3-$f/model.tar.gz data/fever/dev.ns.pages.p1.jsonl
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db logs/ablation/seed4-$f/model.tar.gz data/fever/dev.ns.pages.p1.jsonl
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db logs/ablation/seed5-$f/model.tar.gz data/fever/dev.ns.pages.p1.jsonl
done
