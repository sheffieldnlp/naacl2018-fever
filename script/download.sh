#$ -N fever-download
#$ -V
#$ -t 1-51
#$ -cwd
#$ -l rmem=10G
module load apps/python/anaconda3-4.2.0
source activate fever

block=$SGE_TASK_ID
PYTHONPATH=src S3_BUCKET=$bucket python src/scripts.dataset/download_dataset.py $block
