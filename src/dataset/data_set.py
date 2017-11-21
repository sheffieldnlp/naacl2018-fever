class DataSet():
    def __init__(self,file,reader,formatter):
        self.reader = reader
        self.file = file
        self.formatter = formatter
        self.data = []


    def read(self):
        self.data.extend(self.formatter.format(self.reader.read(self.file)[:10]))



