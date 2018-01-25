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

## Installation

Create a virtual environment for FEVER with Python 3.6 and activate it

    conda create -n fever python=3.6
    source activate fever

Manually Install PyTorch (different distributions should follow instructions from [pytorch.org](http://pytorch.org/))

    conda install pytorch torchvision -c pytorch

Clone the repository

    git clone https://github.com/sheffieldnlp/fever-baselines
    cd fever-baselines

Install requirements

    pip install -r requirements.txt

Download the DrQA document retrieval code and the fever dataset (as a git submodule)

    git submodule init
    git submodule update

## Data Preparation

Download Wikipedia data

    wget https://tbd
    unzip tbd.zip data/wiki

Copy Wikipedia pages into SQLite DB and build TF-IDF index

    PYTHONPATH=src python src/scripts/build_db.py data/wiki data/fever/fever.db
    PYTHONPATH=lib/DrQA/scripts/retriever python lib/DrQA/scripts/retriever/build_tfidf.py data/fever/fever.db data/index/

Sample training data for the NotEnoughInfo class

    #Using nearest neighbor method
    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split train
    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev

    #Using random sampling method
    PYTHONPATH=src python src/scripts/dataset/neg_sample_evidence.py data/fever/fever.db

## Training

Multilayer Perceptron

    #If using a GPU, set
    export GPU=1
    #If more than one GPU,
    export CUDA_DEVICE=0 #(or any CUDA device id. default is 0)

    # Using nearest neighbor sampling method for NotEnoughInfo class (better)
    PYTHONPATH=src src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/fever/train.ns.pages.p1.jsonl data/fever/dev.ns.pages.p1.jsonl --model ns_nn_sent --sentence true

    #Or, using random sampled data for NotEnoughInfo (worse)
    PYTHONPATH=src src/scripts/rte/mlp/train_mlp.py data/fever/fever.db data/fever/train.ns.rand.jsonl data/fever/dev.ns.rand.jsonl --model ns_rand_sent --sentence true


LSTM with Decomposable Attention

    #if using a CPU, set
    export CUDA_DEVICE=-1

    #if using a GPU, set
    export CUDA_DEVICE=0 #or cuda device id

    # Using nearest neighbor sampling method for NotEnoughInfo class (better)
    PYTHONPATH=src src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/cpu --cuda-device $CUDA_DEVICE

    #Or, using random sampled data for NotEnoughInfo (worse)
    PYTHONPATH=src src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_rs_ora_sent.json logs/cpu --cuda-device $CUDA_DEVICE


## Evaluation

## Interactive mode


