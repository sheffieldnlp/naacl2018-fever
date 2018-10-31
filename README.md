
# UOFA- Fact Extraction and VERification
## Smart NER: replace tokens with NER tags but checking if they exists in the claim 

To run the the training and evaluation using the smartNER either just do `./run_all_train_test.sh`
or use these commands below
@server@jenny

`rm -rf logs/`

`PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/da_nn_sent --cuda-device $CUDA_DEVICE`

`mkdir -p data/models`

`cp logs/da_nn_sent/model.tar.gz data/models/decomposable_attention.tar.gz`

`PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/dev.ns.pages.p1.jsonl`

This assumes that you are on the same folder. If your data folder is somewhere else, use this 

for training:
`PYTHONPATH=src python src/scripts/rte/da/train_da.py /net/kate/storage/work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db config/fever_nn_ora_sent.json logs/da_nn_sent --cuda-device $CUDA_DEVICE`
for dev:
`PYTHONPATH=src python src/scripts/rte/da/eval_da.py /net/kate/storage/work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db data/models/decomposable_attention.tar.gz /net/kate/storage/work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.ns.pages.p1.jsonl`






`source activate fever`
`PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/dev.ns.pages.p1.jsonl`
    
# Fact Extraction and VERification


- To annotate data once you have Docker you need to pull pyprocessors using :docker pull myedibleenso/processors-server:latest

- Then run this image using: docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server

note: the docker run command is for the very first time you create this container. Second time onwards use: docker start procserv

- source activate fever

## to run training from my_fork folder on jenny
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/train.jsonl --out-file data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode train --lmode WARNING`


## to run training from another folder on jenny
PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/train.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode train --lmode WARNING

## to run training on a smaller data set from another folder on jenny
PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db
--model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/train.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode small  --dynamic_cv True


 ## To run our entailment trainer on training data alone :

data_root="/work/mithunpaul/fever/my_fork/fever-baselines/data"

## To run on dev

`PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/dev.jsonl --out-file data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode dev --lmode WARNING`

## to run dev in a  folder branch_myfork in server but feeding from same data fold
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/dev.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode dev --lmode INFO`

## to run testing
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/dev.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode test --dynamic_cv True`

## to run dev after running the nearest neighbors algo for not enough info class (note that this assumes that you have run the NEI code mentioned below by sheffield)
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.ns.pages.p1.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl  --max-page 5 --max-sent 5 --mode dev --lmode INFO`


## Copy of Instructions from sheffield :might not be updated. use their instructions [page](https://github.com/sheffieldnlp/fever-baselines#evaluation)
This is the PyTorch implementation of the FEVER pipeline baseline described in the NAACL2018 paper: [FEVER: A large-scale dataset for Fact Extraction and VERification.]()

> Unlike other tasks and despite recent interest, research in textual claim verification has been hindered by the lack of large-scale manually annotated datasets. In this paper we introduce a new publicly available dataset for verification against textual sources, FEVER: Fact Extraction and VERification. It consists of 185,441 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo by annotators achieving 0.6841 in Fleiss κ. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment. To characterize the challenge of the dataset presented, we develop a pipeline approach using both baseline and state-of-the-art components and compare it to suitably designed oracles. The best accuracy we achieve on labeling a claim accompanied by the correct evidence is 31.87%, while if we ignore the evidence we achieve 50.91%. Thus we believe that FEVER is a challenging testbed that will help stimulate progress on claim verification against textual sources

The baseline model constists of two components: Evidence Retrieval (DrQA) + Textual Entailment (Decomposable Attention).

## Find Out More

 * Visit [http://fever.ai](http://fever.ai) to find out more about the shared task and download the data.

## Quick Links
 * [Docker Install](#docker-install)
 * [Manual Install](#manual-install)
 * [Download Data](#download-data)
 * [Data Preparation](#data-preparation)
 * [Train](#training)
 * [Evaluate](#evaluation)
 * [Score and Upload to Codalab](#scoring)
 
 

 
## Pre-requisites

This was tested and evaluated using the Python 3.6 verison of Anaconda 5.0.1 which can be downloaded from [anaconda.com](https://www.anaconda.com/download/)

Mac OSX users may have to install xcode before running git or installing packages (gcc may fail). 
See this post on [StackExchange](https://apple.stackexchange.com/questions/254380/macos-sierra-invalid-active-developer-path)

Support for Windows operating systems is not provided.

To train the Decomposable Attention models, it is highly recommended to use a GPU. Training will take about 3 hours on a GTX 1080Ti whereas training on a CPU will take days. We offer a pre-trained model.tar.gz that can be [downloaded](https://jamesthorne.co/fever/model.tar.gz). To use the pretrained model, simply replace any path to a model.tar.gz file with the path to the file you downloaded. (e.g. `logs/da_nn_sent/model.tar.gz` could become `~/Downloads/model.tar.gz`) 


## Change Log
 
* **v0.3** - Added the ability to read unlabelled data (i.e. the blind dataset for the competition). **You must update to this version to take part in the competition**
* **v0.2** - updated the Information Retrieval component to use a modified version of DrQA that allows multi-threaded document/sentence retrieval. This yields a >10x speed-up the in IR stage of the pipeline as I/O waits are no longer blocking computation of TF*IDF vectors  
* **v0.1** - original implementation (tagged as naacl2018)

 
## Docker Install

Download and run the latest FEVER. 

    docker volume create fever-data
    docker run -it -v fever-data:/fever/data sheffieldnlp/fever-baselines
    
To enable GPU acceleration (run with `--runtime=nvidia`) once [NVIDIA Docker has been installed](https://github.com/NVIDIA/nvidia-docker)
 
## Manual Install

Installation using docker is preferred. If you are unable to do this, you can manually create the python environment following instructions here: 
[Wiki/Manual-Install](https://github.com/sheffieldnlp/fever-baselines/wiki/Manual-Install)

Remember that if you manually installed, you should run `source activate fever` and `cd` to the directory before you run any commands.

## Download Data

### Wikipedia

To download a pre-processed Wikipedia dump ([license](https://s3-eu-west-1.amazonaws.com/fever.public/license.html)):

    bash scripts/download-processed-wiki.sh

Or download the raw data and process yourself

    bash scripts/download-raw-wiki.sh
    bash scripts/process-wiki.sh


### Dataset

Download the FEVER dataset from [our website](https://sheffieldnlp.github.io/fever/data.html) into the data directory:

    bash scripts/download-data.sh


(note that if you want to replicate the paper, run `scripts/download-paper.sh` instead of `scripts/download-data`).  
 
 
### Word Embeddings 
  
Download pretrained GloVe Vectors

    bash scripts/download-glove.sh



## Data Preparation

Sample training data for the NotEnoughInfo class. There are two sampling methods evaluated in the paper: using the nearest neighbour (similarity between TF-IDF vectors) and random sampling.

    #Using nearest neighbor method
    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split train
    PYTHONPATH=src python src/scripts/retrieval/document/batch_ir_ns.py --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split dev

Or random sampling

    #Using random sampling method
    PYTHONPATH=src python src/scripts/dataset/neg_sample_evidence.py data/fever/fever.db
    
## Training

We offer a pretrained model that can be downloaded by running the following command: 

    bash scripts/download-model.sh
    
    
Skip to [evaluation](#evaluation) if you are using the pretrained model.


### Train DA
Train the Decomposable Attention model

    #if using a CPU, set
    export CUDA_DEVICE=-1

    #if using a GPU, set
    export CUDA_DEVICE=0 #or cuda device id

Then either train the model with Nearest-Page Sampling for the NEI class 

    # Using nearest neighbor sampling method for NotEnoughInfo class (better)
    PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/da_nn_sent --cuda-device $CUDA_DEVICE
    mkdir -p data/models
    cp logs/da_nn_sent/model.tar.gz data/models/decomposable_attention.tar.gz
    
Or with Random Sampling for the NEI class

    # Using random sampled data for NotEnoughInfo (worse)
    PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_rs_ora_sent.json logs/da_rs_sent --cuda-device $CUDA_DEVICE
    mkdir -p data/models
    cp logs/da_rs_sent/model.tar.gz data/models/decomposable_attention.tar.gz


 


### Train MLP
The MLP model can be trained following instructions from the Wiki: [Wiki/Train-MLP](https://github.com/sheffieldnlp/fever-baselines/wiki/Train-MLP)


## Evaluation

These instructions are for the decomposable attention model. The MLP model can be evaluated following instructions from the Wiki: [Wiki/Evaluate-MLP](https://github.com/sheffieldnlp/fever-baselines/wiki/Evaluate-MLP)

### Oracle Evaluation (no evidence retrieval):
    
Run the oracle evaluation for the Decomposable Attention model on the dev set (requires sampling the NEI class for the dev dataset - see [Data Preparation](#data-preparation))
    
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/dev.ns.pages.p1.jsonl
    

### Evidence Retrieval Evaluation:

First retrieve the evidence for the dev/test sets:

    #Dev
    PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/dev.jsonl --out-file data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5
    
    #Test
    PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/test.jsonl --out-file data/fever/test.sentences.p5.s5.jsonl --max-page 5 --max-sent 5

Then run the model:
    
    #Dev
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/dev.sentences.p5.s5.jsonl  --log data/decomposable_attention.dev.log
    
    #Test
    PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/decomposable_attention.tar.gz data/fever/test.sentences.p5.s5.jsonl  --log logs/decomposable_attention.test.log


## Scoring
### Score locally (for dev set)  
Score:

    PYTHONPATH=src python src/scripts/score.py --predicted_labels data/decomposable_attention.dev.log --predicted_evidence data/fever/dev.sentences.p5.s5.jsonl --actual data/fever-data/dev.jsonl

### Or score on Codalab (for dev/test)

Prepare Submission for Codalab (dev):

    PYTHONPATH=src python src/scripts/prepare_submission.py --predicted_labels logs/decomposable_attention.dev.log --predicted_evidence data/fever/dev.sentences.p5.s5.jsonl --out_file predictions.jsonl
    zip submission.zip predictions.jsonl

Prepare Submission for Codalab (test):

    PYTHONPATH=src python src/scripts/prepare_submission.py --predicted_labels logs/decomposable_attention.test.log --predicted_evidence data/fever/test.sentences.p5.s5.jsonl --out_file predictions.jsonl
    zip submission.zip predictions.jsonl

          
