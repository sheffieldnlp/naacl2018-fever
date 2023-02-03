#!/bin/bash
mkdir -p data
mkdir -p data/fever-data
wget -O data/fever-data/train.jsonl https://fever.ai/download/fever/train.jsonl
wget -O data/fever-data/dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
wget -O data/fever-data/test.jsonl https://fever.ai/download/fever/shared_task_test.jsonl
