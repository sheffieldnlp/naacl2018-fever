#!/bin/bash
mkdir -p data
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
unzip wiki-pages.zip -d data