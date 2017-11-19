
class Task:
    def score(self,data):
        self.do_scoring(data)

    def do_scoring(self):
        raise NotImplementedError("Not Implemented Here")


class IRTask(Task):
    def do_scoring(self,data):
        pass


class InferenceTask(Task):
    pass
