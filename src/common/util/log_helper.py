import logging
class LogHelper():
    handler = None
    @staticmethod
    def setup():
        FORMAT = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
        LogHelper.handler = logging.StreamHandler()
        LogHelper.handler.setLevel(logging.DEBUG)
        LogHelper.handler.setFormatter(logging.Formatter(FORMAT))

        LogHelper.get_logger(LogHelper.__name__).info("Log Helper set up")

    @staticmethod
    def get_logger(name,level=logging.DEBUG):
        l = logging.getLogger(name)
        l.setLevel(level)
        l.addHandler(LogHelper.handler)
        return l
