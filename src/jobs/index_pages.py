import boto3
import os
import pickle
from botocore import UNSIGNED
from botocore.client import Config
from botocore.handlers import disable_signing

from dataset.s3.iterator import s3_iterator
from util.log_helper import LogHelper


class Indexer:
    def __init__(self,file):
        self.pages = []
        self.file = ""
        self.logger = LogHelper.get_logger(__name__)
        self.logger.info("Indexing Pages")

    def index_page(self,key):

        logger.debug("Index Page: {0}".format(key))
        self.pages.append(key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Saving index")
        pickle.dump(self.pages,self.file)

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("Index Pages in Dataset")

    #Use boto3 to download all pages from intros section from s3
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    resource = boto3.resource("s3")
    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

    with open(os.path.join("data","fever","pagse.p"),"wb+") as f:
        with Indexer(f) as indexer:
            s3_iterator(client,resource,"wiki-dump/intro/",os.getenv("S3_BUCKET"),indexer.index_page)
