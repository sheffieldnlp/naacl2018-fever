#$ -N fever-sent-lstm-da-rs
#$ -V
#$ -cwd
#$ -q rse.q
#$ -P rse
#$ -l rmem=40G
#$ -l gpu=1
#$ -l h_rt=95:00:00

source /etc/profile.d/modules.sh
source ~/.bashrc

module load libs/CUDA/8.0.44/binary

source activate dgx

cp data/fever/fever.db $TMPDIR/fever.db
PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/rte/da/train_da.py $TMPDIR/fever.db config/fever_rs_ora_sent.json logs/sent_rs