FROM continuumio/miniconda3

#ENTRYPOINT ["/bin/bash"]

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN mkdir /fever/
RUN mkdir /fever/src
RUN mkdir /fever/config
RUN mkdir /fever/scripts

VOLUME /fever/

ADD requirements.txt /fever/
ADD src /fever/src/
ADD config /fever/config/
ADD scripts /fever/scripts/
ADD data /fever/data/

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip

RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n fever python=3.6

WORKDIR /fever/
RUN . activate fever
RUN conda install pytorch torchvision -c pytorch
RUN conda install cython nltk scikit-learn
RUN pip install -r requirements.txt
RUN python src/scripts/prepare_nltk.py
ENV PYTHONPATH src
CMD ["python", "src/scripts/retrieval/ir.py"]
#--db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/dev.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode dev --lmode INFO
