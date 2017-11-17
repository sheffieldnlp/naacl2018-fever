from util.log_helper import LogHelper

if __name__ == "__main__":
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    logger.info("Prepare dataset")

