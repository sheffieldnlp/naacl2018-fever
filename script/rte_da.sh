PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/da/train.py data/fever/drqa.db config/fever_rs_k5.json logs/rs_k5
PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/da/train.py data/fever/drqa.db config/fever_rs_k1.json logs/rs_k1
PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/da/train.py data/fever/drqa.db config/fever_rs_ora.json logs/rs_ora

PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/da/train.py data/fever/drqa.db config/fever_nn1_k5.json logs/nn1_k5
PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/da/train.py data/fever/drqa.db config/fever_nn1_k1.json logs/nn1_k1
PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/da/train.py data/fever/drqa.db config/fever_nn1_ora.json logs/nn1_ora
