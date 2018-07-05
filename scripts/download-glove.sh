#!/bin/bash
mkdir -p data
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip -d data/glove
gzip data/glove/*.txt