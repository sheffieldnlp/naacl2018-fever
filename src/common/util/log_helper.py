import logging
class LogHelper():
    handler = None
    handler2 = None
    @staticmethod
    def setup():

        FORMAT = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
        LogHelper.handler = logging.StreamHandler()
        LogHelper.handler.setLevel(logging.INFO)
        LogHelper.handler.setFormatter(logging.Formatter(FORMAT))

        LogHelper.handler2 = logging.FileHandler("old_log.log" ,mode='w')
        LogHelper.handler2.setLevel(logging.INFO)
        LogHelper.handler.setFormatter(logging.Formatter(FORMAT))

        LogHelper.get_logger(LogHelper.__name__)


    @staticmethod
    def get_logger(name,level=logging.DEBUG):
        l = logging.getLogger(name)
        l.setLevel(level)
        l.addHandler(LogHelper.handler)
        l.addHandler(LogHelper.handler2)

        return l
