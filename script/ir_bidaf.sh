#$ -N fever-bidaf
#$ -V
#$ -cwd
#$ -l rmem=16G
#$ -l gpu=1
#$ -l h_rt= 95:00:00
module load libs/CUDA/8.0.44/binary
source activate fever
PYTHONPATH=src:lib/DrQA:lib/allennlp python src/scripts/retrieval/sentence/train.py data/fever/fever.db config/fever_bidaf.json logs/bidaf
