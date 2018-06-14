FROM continuumio/miniconda3

ENTRYPOINT ["/bin/bash", "-c"]


RUN mkdir /fever/
VOLUME /fever/
ADD requirements.txt /fever/
ADD setup.py /fever/
ADD src /fever/
ADD config /fever/
ADD requirements.txt /fever/
ADD setup.py /fever/

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
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
RUN python setup.py install

