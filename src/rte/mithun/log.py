import logging

def setup_custom_logger(name):

    log_mode="logging.INFO"
    logging.basicConfig(level=log_mode,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='log_fever.txt',
                    filemode='w')

    logger = logging.getLogger(name)
    '''critical, error > warning,info, debug'''

    ch=logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logging.getLogger('').addHandler(ch)


    return logger
