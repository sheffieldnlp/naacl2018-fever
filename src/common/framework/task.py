
class Task:
    def score(self,data,labels):
        self.do_scoring(data,labels)

    def do_scoring(self):
        raise NotImplementedError("Not Implemented Here")


class IRTask(Task):
    def do_scoring(self,data,labels):
        pass






class InferenceTask(Task):
    pass
