import logging
class LogHelper():
    handler = None
    handler2 = None
    @staticmethod
    def setup():
        logging.basicConfig(level=logging.WARNING,
                            format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='log_fever.log',
                            filemode='w')
        LogHelper.handler = logging.StreamHandler()
        LogHelper.handler.setLevel(logging.DEBUG)

        # LogHelper.handler2 = logging.FileHandler(filename)
        # LogHelper.handler2.setLevel(logging.DEBUG)

        LogHelper.get_logger(LogHelper.__name__).info("Log Helper set up")


    @staticmethod
    def get_logger(name,level=logging.DEBUG):
        l = logging.getLogger(name)
        l.setLevel(level)
        l.addHandler(LogHelper.handler)
        l.addHandler(LogHelper.handler2)

        return l
