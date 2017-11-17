#$ -N fever-index
#$ -V
#$ -cwd
#$ -l rmem=10G
module load apps/python/anaconda3-4.2.0
source activate fever

PYTHONPATH=src S3_BUCKET=$bucket python src/jobs/index_pages.py
