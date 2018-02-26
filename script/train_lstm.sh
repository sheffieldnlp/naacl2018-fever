FILES=`ls data/ablation/* | xargs -n1 basename`
for f in $FILES;
do
    echo "Train $f";
    export CUDA_DEVICE=0 #or cuda device id

    PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/ablation/$f --cuda-device $CUDA_DEVICE --overrides {"train_data_path":"data/ablation/$f"}
done
