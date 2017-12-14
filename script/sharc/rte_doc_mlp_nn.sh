#$ -N fever-doc-mlp-nn
#$ -V
#$ -cwd
#$ -q rse.q
#$ -P rse
#$ -l rmem=22G
#$ -l gpu=1
#$ -l h_rt=95:00:00

source /etc/profile.d/modules.sh
source ~/.bashrc

module load libs/CUDA/8.0.44/binary

source activate dgx

cp data/fever/fever.db $TMPDIR/fever.db
TRAIN=1 PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/mlp/document_mlp.py $TMPDIR/fever.db data/fever/train.ns.pages.p1.jsonl data/fever/dev.ns.pages.p1.jsonl --model ns_nn_doc