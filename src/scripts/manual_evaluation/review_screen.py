import json
import sys
import os
#python review_screen.py dump100.json


def pprint(dump,hl=None):

    for idx, line in enumerate(dump.split("\n")):
        bts = line.split("\t")
        print("{2}\t{0}\t{1}".format(bts[0],bts[1] if len(bts)>1 else "","*" if hl is not None and int(hl) == idx else ""))


# Read dump
with open(sys.argv[1],"r") as f:
    data = json.load(f)


with open("notes.jsonl","a+") as f:
    for item in data["annotations"]:
        os.system("clear")

        print("ID: \t{0}\t\t\t{1}".format(item["id"],item["text"]))
        print("Original Page: \t{0}\t\t\tIs Oracle: {1}\t\t\tIs Reval: {2}".format(item["original_page"], item["isOracle"], item["isReval"]))
        print("")


        for idx,annotation in enumerate(item["annotations"]):
            print("Annotation {0}".format(idx+1))
            print("\tPage: {0}\t\tLine: {1}".format(annotation["page"], annotation["line"]))
            print("\t{0}\t{1}\t\t(isOracleMaster: {2})".format(annotation["verifiable"],annotation["label"],annotation["isOracleMaster"]))

            print("")
            print("Selected Text")
            pprint(data["texts"][annotation["page"]],annotation["line"])

        print("")
        print("")

        print("Original Page:")

        pprint(data["texts"][item["original_page"]])


        print("")
        print("")

        correct = input("Correct annotation (y/n/q/s): ")
        if correct=="q":
            break
        elif correct=="s":
            continue

        notes = input("Notes: ")

        f.write(json.dumps({"id":item['id'],"correct":correct,"notes":notes})+"\n")

