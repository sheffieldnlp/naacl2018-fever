#!/bin/bash
mkdir -p data
mkdir -p data/fever-data

#To replicate the paper, download paper_dev and paper_test files. These are concatenated for the shared task
wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl
