class Formatter():
    def __init__(self,label_schema):
        self.label_schema = label_schema

    def format(self,lines):
        formatted = []
        for line in lines:
            fl = self.format_line(line)
            if fl is not None:
                if isinstance(fl,list):
                    formatted.extend(fl)
                else:
                    formatted.append(fl)

        return formatted

    def format_line(self,line):
        pass
