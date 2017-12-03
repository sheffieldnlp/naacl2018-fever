

class EarlyStopping():
    def __init__(self,patience=8):
        self.patience = patience
        self.best_model = None
        self.best_score = None

        self.best_epoch = 0
        self.epoch = 0

    def __call__(self, model, acc):
        self.epoch += 1
        if acc >= self.best_score:
            self.best_model = model
            self.best_score = acc
            return False
        elif self.epoch > self.best_epoch+self.patience:
            return True
        return False