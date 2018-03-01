FILES=`ls data/ablation/* | xargs -n1 basename`
for f in $FILES;
do
    echo "Train $f";
    export CUDA_DEVICE=0 #or cuda device id

    RANDOM_SEED=1234 PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/ablation/seed1-$f --cuda-device $CUDA_DEVICE --overrides {"train_data_path":"data/ablation/$f"}
    RANDOM_SEED=2345 PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/ablation/seed2-$f --cuda-device $CUDA_DEVICE --overrides {"train_data_path":"data/ablation/$f"}
    RANDOM_SEED=3456 PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/ablation/seed3-$f --cuda-device $CUDA_DEVICE --overrides {"train_data_path":"data/ablation/$f"}
    RANDOM_SEED=4567 PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/ablation/seed4-$f --cuda-device $CUDA_DEVICE --overrides {"train_data_path":"data/ablation/$f"}
    RANDOM_SEED=5678 PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/ablation/seed5-$f --cuda-device $CUDA_DEVICE --overrides {"train_data_path":"data/ablation/$f"}
done
