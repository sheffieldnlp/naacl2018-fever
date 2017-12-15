#$ -N wmd
#$ -V
#$ -cwd
#$ -q rse.q
#$ -P rse
#$ -l rmem=8G
#$ -pe openmp 4
#$ -l h_rt=95:00:00

source /etc/profile.d/modules.sh
source ~/.bashrc

source activate dgx

cp data/fever/fever.db $TMPDIR/fever.db
python -m spacy download en
PYTHONPATH=src:lib/DrQA/:lib/allennlp/ python src/scripts/retrieval/sentence/process.py $TMPDIR/fever.db data/fever/dev.pages.p100.jsonl data/fever/dev.sentences.${k}.${l} ${k} ${l}