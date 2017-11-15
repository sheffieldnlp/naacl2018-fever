import logging

import boto3
import os
import pickle
from botocore.handlers import disable_signing
from dataset.s3.download_directory import download_dir
from botocore import UNSIGNED
from botocore.client import Config

FORMAT = '[%(levelname)s] - %(asctime)s - %(name)s - %(message)s'

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(FORMAT))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(ch)


class Writer(object):


    def __init__(self,name,path,limit=250000):
        self.volume = 0
        self.name = name
        self.path = path
        self.data = dict()
        self.size = 0
        self.limit = limit

    def save(self,name,data):

        self.data[name] = data
        self.size += 1

        if self.size % 10000 == 0:
            print(self.size)

        if self.size >= self.limit:
            self.write()

    def write(self):
        print("Write block " + str(self.volume))
        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p")), "wb+") as f:
            pickle.dump(self.data, f)

        with open((os.path.join(self.path, self.name + "-" + str(self.volume) + ".p.idx")), "wb+") as f:
            pickle.dump(set(self.data.keys()), f)

        self.data = dict()
        self.volume += 1
        self.size = 0

    def close(self):
        self.write()

if __name__ == "__main__":
    logger.info("Preparing Dataset")

    intro_path = os.path.join("data","fever")
    if not os.path.exists(intro_path):
        os.makedirs(intro_path)

    writer = Writer("wiki",intro_path)

    #Use boto3 to download all pages from intros section from s3
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    resource = boto3.resource("s3")
    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    download_dir(client,resource,"wiki-dump/intro/",os.getenv("S3_BUCKET"),writer)
    writer.close()
