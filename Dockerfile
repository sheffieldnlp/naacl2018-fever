FROM continuumio/miniconda3

ENTRYPOINT ["/bin/bash"]


RUN mkdir /fever/
RUN mkdir /fever/src
RUN mkdir /fever/config

VOLUME /fever/

ADD requirements.txt /fever/
ADD src /fever/src/
ADD config /fever/config/

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    zip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config

RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n fever python=3.6

WORKDIR /fever/
RUN . activate fever
RUN conda install -y pytorch=0.3.1 torchvision -c pytorch
RUN pip install -r requirements.txt
