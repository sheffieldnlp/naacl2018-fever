import boto3
import os
from botocore.handlers import disable_signing
from dataset.s3.iterator import download_dir
from botocore import UNSIGNED
from botocore.client import Config
from dataset.s3.local_writer import Writer
from util.log_helper import LogHelper

if __name__ == "__main__":
    logger = LogHelper.get_logger(__name__)
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
