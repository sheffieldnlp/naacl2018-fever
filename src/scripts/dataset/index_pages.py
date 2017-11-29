import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.handlers import disable_signing

from common.dataset import Indexer
from common.dataset import s3_iterator
from common.util import LogHelper

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("Index Pages in Dataset")

    logger.debug("Checking if fever data dir exists")
    intro_path = os.path.join("data", "fever")
    if not os.path.exists(intro_path):
        logger.debug("Creating data dir")
        os.makedirs(intro_path)


    #Use boto3 to download all pages from intros section from s3
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    resource = boto3.resource("s3")
    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

    with open(os.path.join("data","fever","pages.p"),"wb+") as f:
        with Indexer(f) as indexer:
            s3_iterator(client,resource,"wiki-dump/intro/","wiki-dump/intro/",os.getenv("S3_BUCKET"),indexer.index_page)

