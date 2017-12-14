import torch


class EarlyStopping():
    def __init__(self,name,patience=8):
        self.patience = patience
        self.best_model = None
        self.best_score = None

        self.best_epoch = 0
        self.epoch = 0

        self.name = name

    def __call__(self, model, acc):
        self.epoch += 1

        if self.best_score is None:
            self.best_score = acc

        if acc >= self.best_score:
            torch.save(model.state_dict(),"models/{0}.best.save".format(self.name))
            self.best_score = acc
            self.best_epoch = self.epoch
            return False

        elif self.epoch > self.best_epoch+self.patience:
            print("Early stopping: Terminate")
            return True

        print("Early stopping: Worse Round")
        return False

    def set_best_state(self,model):
        print("Loading weights from round {0}".format(self.best_epoch))
        model.load_state_dict(torch.load("models/{0}.best.save".format(self.name)))
