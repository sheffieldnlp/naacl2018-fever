import os
import json

from drqa.retriever.utils import normalize
from retrieval.fever_doc_db import FeverDocDB

data_folder = "data"
redirects_file = "redirect.txt"

def get_redirects():
    redirs = os.path.join(data_folder, redirects_file)
    rd = dict()
    with open(redirs, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break

            bits = line.strip().split("\t")
            if len(bits) == 2:
                frm, to = bits
                rd[frm] = to

    return rd

def clean(filename):
    return filename.replace("(", "-LRB-").replace(")", "-RRB-").replace("[", "-LSB-") \
        .replace("]", "-RSB-").replace(":", "-COLON-").replace(" ", "_")


def recursive_redirect_lookup(redirects, word):
    if word in redirects:
        try:
            return recursive_redirect_lookup(redirects, redirects[word])
        except RecursionError:
            return word
    else:
        return word

def get_wiki_entry(name):

    if normalize(clean(name)) in idx:
        return normalize(clean(name))
    else:
        try:
            if name[0].islower():
                return normalize(clean(get_wiki_entry(name[0].upper()+name[1:])))
            else:
                return normalize(clean(get_wiki_entry(recursive_redirect_lookup(redirects,redirects[name]))))
        except:
            return None


if __name__ == "__main__":
    db = FeverDocDB("data/fever/fever.db")
    idx = set(db.get_doc_ids())
    redirects = get_redirects()

    split = os.getenv("SPLIT")

    fixed = 0
    errs = 0
    with open("data/fever-data/{0}.jsonl".format(split),"r") as f:

        with open("data/fever-data/{0}.resolved.jsonl".format(split), "w+") as g:
            for line in f:
                js = json.loads(line)

                evidence = js["evidence"]

                for i, annotation in enumerate(evidence):
                    for j,ev in enumerate(annotation):
                        if ev[2] is not None:
                            ev[2] = ev[2].split("#")[0]
                            m = clean(normalize(ev[2]))
                            n = get_wiki_entry(ev[2])

                            evidence[i][j][2] = n

                g.write(json.dumps(js)+"\n")


