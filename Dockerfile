FROM continuumio/miniconda3

ENTRYPOINT ["/bin/bash"]

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
RUN conda install -y pytorch=0.3.1 torchvision -c pytorch
RUN pip install -r requirements.txt
RUN python src/scripts/prepare_nltk.py
