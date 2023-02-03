#!/bin/bash
mkdir -p data
mkdir -p data/fever-data

#To replicate the paper, download paper_dev and paper_test files. These are concatenated for the shared task
wget -O data/fever-data/train.jsonl https://fever.ai/download/fever/train.jsonl
wget -O data/fever-data/dev.jsonl https://fever.ai/download/fever/paper_dev.jsonl
wget -O data/fever-data/test.jsonl https://fever.ai/download/fever/paper_test.jsonl
