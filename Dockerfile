FROM continuumio/miniconda3

ENTRYPOINT ["/bin/bash", "-c"]

RUN mkdir /fever/
VOLUME /fever/
ADD requirements.txt /fever/
ADD setup.py /fever/
ADD src /fever/
ADD config /fever/

RUN apt-get update
RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n fever python=3.6
RUN source activate fever
RUN pip install -r requirements.txt
RUN python setup.py install

