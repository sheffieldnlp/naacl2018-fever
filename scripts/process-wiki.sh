#!/bin/bash
PYTHONPATH=src python src/scripts/build_db.py data/wiki-pages data/fever/fever.db
PYTHONPATH=src python src/scripts/build_tfidf.py data/fever/fever.db data/index/