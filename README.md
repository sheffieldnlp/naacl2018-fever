# Fact Extraction and VERification

This is the PyTorch imp0lementation of the FEVER baselines described in th NAACL2018 paper.

## Links
Installation
Data preparation
Training
Evaluation
Demo

## Pre-requisites

This was tested and evaluated using the Python 3.6 verison of Anaconda 5.0.1 which can be downloaded from [anaconda.com](https://www.anaconda.com/download/)

To train the LSTM-based models, it is highly recommended to use a GPU. Training will take about 3 hours on a GTX 1080Ti whereas training on a CPU will take days. We offer a pre-trained model.tar.gz that can be downloaded [from Google Drive](https://drive.google.com/file/d/1aoiC5lJl_X8Oo_IFdowCJG3zF7W1X3X8/view) 

## Installation

Mac OSX users may have to install xcode before running git or installing packages (gcc may fail). See this post: [https://apple.stackexchange.com/questions/254380/macos-sierra-invalid-active-developer-path]

Create a virtual environment for FEVER with Python 3.6 and activate it

    conda create -n fever python=3.6
    source activate fever

Manually Install PyTorch (different distributions should follow instructions from [pytorch.org](http://pytorch.org/))

    conda install pytorch torchvision -c pytorch

Clone the repository

    git clone https://github.com/sheffieldnlp/fever-baselines
    cd fever-baselines

Install requirements (run export LANG=C.UTF-8 if installation of DrQA fails)

    pip install -r requirements.txt

Download the DrQA document retrieval code and the fever dataset (as a git submodule)

    git submodule init
    git submodule update

## Data Preparation

Download Wikipedia data: https://drive.google.com/file/d/1BMnxxIcoC8VRL5p3E6kamgpVmAyALH2x/view and unzip it into the data folder.

    unzip wiki.zip -d data

Copy Wikipedia pages into SQLite DB and build TF-IDF index (go grab a coffee while this runs)

    PYTHONPATH=src python src/scripts/build_db.py data/wiki data/fever/fever.db
    mkdir data/index
    PYTHONPATH=lib/DrQA/scripts/retriever python lib/DrQA/scripts/retriever/build_tfidf.py data/fever/fever.db data/index/

Sample training data for the NotEnoughInfo class (may also take a while)

    #Using nearest neighbor method
    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split train
    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev

    #Using random sampling method
    PYTHONPATH=src python src/scripts/dataset/neg_sample_evidence.py data/fever/fever.db

Download pretrained GloVe Vectors

    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip glove.6B.zip -d data/glove
    gzip data/glove/*.txt
    
## Training

Model 1: Multilayer Perceptron

    #If using a GPU, set
    export GPU=1
    #If more than one GPU,
    export CUDA_DEVICE=0 #(or any CUDA device id. default is 0)

    # Using nearest neighbor sampling method for NotEnoughInfo class (better)
    PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/fever/train.ns.pages.p1.jsonl data/fever/dev.ns.pages.p1.jsonl --model ns_nn_sent --sentence true

    #Or, using random sampled data for NotEnoughInfo (worse)
    PYTHONPATH=src python src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/fever/train.ns.rand.jsonl data/fever/dev.ns.rand.jsonl --model ns_rand_sent --sentence true


Model 2: LSTM with Decomposable Attention

    #if using a CPU, set
    export CUDA_DEVICE=-1

    #if using a GPU, set
    export CUDA_DEVICE=0 #or cuda device id

    # Using nearest neighbor sampling method for NotEnoughInfo class (better)
    PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/da_nn_sent --cuda-device $CUDA_DEVICE

    #Or, using random sampled data for NotEnoughInfo (worse)
    PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_rs_ora_sent.json logs/da_rs_sent --cuda-device $CUDA_DEVICE


## Evaluation

### Oracle Evaluation (no evidence retrieval):
    
Model 1: Multi-layer perceptron

    PYTHONPATH=src python src/scripts/rte/mlp/eval_mlp.py data/fever/fever.db --model ns_nn_sent --sentence true --log logs/mlp_nn_sent
    
Model 2: LSTM with decomposable attention 
    
    PYTHONPATH=src:lib/allennlp python src/scripts/rte/da/eval_da.py data/fever/fever.db logs/da_nn_sent/model.tar.gz data/fever/dev.ns.pages.p1.jsonl
    
 
### Evidence Retrieval Evaluation:

Preprocessing (for both models):

    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split dev
    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 5 --split test
    
    PYTHONPATH=src python src/scripts/retrieval/sentence/process_tfidf.py data/fever/fever.db data/fever/dev.pages.p5.jsonl --max_page 5 --max_sent 5 --split dev
    PYTHONPATH=src ptyhon src/scripts/retrieval/sentence/process_tfidf.py data/fever/fever.db data/fever/test.pages.p5.jsonl --max_page 5 --max_sent 5 --split test

Model 1: Multi-layer perceptron

    PYTHONPATH=src python src/scripts/rte/mlp/eval_mlp.py data/fever/fever.db data/fever/dev.sentences.p5.s5.jsonl --model ns_nn_sent --sentence true --log logs/mlp_nn_sent_dev
    PYTHONPATH=src python src/scripts/rte/mlp/eval_mlp.py data/fever/fever.db data/fever/test.sentences.p5.s5.jsonl --model ns_nn_sent --sentence true --log logs/mlp_nn_sent_test
    
Model 2: LSTM with decomposable attention 
    
    PYTHONPATH=src:lib/allennlp src/scripts/rte/da/eval_da.py data/fever/fever.db logs/da_nn_sent/model.tar.gz data/fever/dev.sentences.p5.s5.jsonl  --log logs/da_nn_sent_dev
    PYTHONPATH=src:lib/allennlp src/scripts/rte/da/eval_da.py data/fever/fever.db logs/da_nn_sent/model.tar.gz data/fever/test.sentences.p5.s5.jsonl  --log logs/da_nn_sent_test
    
 
## Interactive Demo (LSTM with decomposable attention)

    PYTHONPATH=src python src/scripts/rte/dta/interactive.py data/fever/fever.db logs/da_nn_sent/model.tar.gz --model data/fever/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
    
    
