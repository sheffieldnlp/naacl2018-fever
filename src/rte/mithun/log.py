import logging

def setup_custom_logger(name, args):
    log_mode=logging.DEBUG

    if(args.lmode=="DEBUG"):
        log_mode = logging.DEBUG
    else:

        if (args.lmode == "WARNING"):
            log_mode = logging.WARNING

        else:

            if (args.lmode == "INFO"):
                log_mode = logging.INFO

            else:

                if (args.lmode == "ERROR"):
                    log_mode = logging.ERROR

    logging.basicConfig(level=log_mode,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='log_fever.txt',
                    filemode='w')

    logger = logging.getLogger(name)
    '''critical, error > warning,info, debug'''

    #ch=logging.StreamHandler()
    #ch.setLevel(logging.WARNING)
    #logging.getLogger('').addHandler(ch)


    return logger
