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

    def extract_words(self,parse):
        toks = [tok.strip() for tok in parse.replace(")","").split("(")  if " " in tok.strip() and len(tok.strip())]
        return list(zip(*[tok.split(" ") for tok in toks]))

    def format_line(self,line):

        annotation = line["gold_label"]

        if annotation == "-":
            return None

        s1_pos, s1_words = self.extract_words(line['sentence1_parse'])
        s2_pos, s2_words = self.extract_words(line['sentence1_parse'])

        return {"data":{"s1_pos":s1_pos,"s1_words":s1_words, "s2_pos":s2_pos, "s2_words": s2_words}, "label":annotation}