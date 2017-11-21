class Formatter():
    def __init__(self,label_schema):
        self.label_schema = label_schema

    def format(self,lines):
        formatted = []
        for line in lines:
            formatted.append(self.format_line(line))
        return formatted

    def format_line(self,line):
        pass



class SNLIFormatter(Formatter):
    def format_line(self,line):

        annotation = line["gold_label"]

        if annotation == "-":
            return None

        print(line['sentence1_binary_parse'])


        return None