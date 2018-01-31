import torch

from common.util.log_helper import LogHelper


class EarlyStopping():
    def __init__(self,name,patience=8):
        self.patience = patience
        self.best_model = None
        self.best_score = None

        self.best_epoch = 0
        self.epoch = 0

        self.name = name
        self.logger = LogHelper.get_logger(EarlyStopping.__name__)

    def __call__(self, model, acc):
        self.epoch += 1

        if self.best_score is None:
            self.best_score = acc

        if acc >= self.best_score:
            torch.save(model.state_dict(),"models/{0}.best.save".format(self.name))
            self.best_score = acc
            self.best_epoch = self.epoch
            self.logger.info("Saving best weights from round {0}".format(self.epoch))
            return False

        elif self.epoch > self.best_epoch+self.patience:
            self.logger.info("Early stopping: Terminate")
            return True

        self.logger.info("Early stopping: Worse Round")
        return False

    def set_best_state(self,model):
        self.logger.info("Loading weights from round {0}".format(self.best_epoch))
        model.load_state_dict(torch.load("models/{0}.best.save".format(self.name)))
