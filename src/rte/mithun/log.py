import logging

def setup_custom_logger(name):

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='log_fever_verb.txt',
                    filemode='w')

    logger = logging.getLogger(name)
    '''critical, error > warning,info, debug'''

    ch=logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logging.getLogger('').addHandler(ch)


    return logger
