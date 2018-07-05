#!/bin/bash
mkdir -p data
mkdir -p data/index
mkdir -p data/fever

wget -O data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever-tfidf-ngram%3D2-hash%3D16777216-tokenizer%3Dsimple.npz
wget -O data/fever/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db