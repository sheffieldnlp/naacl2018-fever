import json
import random
from collections import defaultdict
import heapq
import pymysql.cursors
import os
# Connect to the database
import sys

connection = pymysql.connect(host=os.getenv("DB_HOST", "localhost"),
                             user=os.getenv("DB_USER", "root"),
                             password=os.getenv("DB_PASS", ""),
                             db=os.getenv("DB_SCHEMA", "fever"),
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


def evidence(claim_id):
    cl_support = [ev for ev in claim_evidence[claim_id] if ev["label"] == "SUPPORTS" ]
    cl_refutes = [ev for ev in claim_evidence[claim_id] if ev["label"] == "REFUTES" ]
    cl_notenough = [ev for ev in claim_evidence[claim_id]  if ev["verifiable"] == "NOT ENOUGH INFO"]
    return cl_support,cl_refutes,cl_notenough



claim_evidence = defaultdict(lambda: [])
try:

    with connection.cursor() as cursor:
        sql = """

        select claim.id, claim.text,
          CASE
            WHEN annotation.verifiable =1 THEN 'VERIFIABLE'
            WHEN annotation.verifiable =2 THEN 'NOT ENOUGH INFO'
            WHEN annotation.verifiable =3 THEN 'NOT VERIFIABLE'
            WHEN annotation.verifiable =4 THEN 'TYPO'
          END as verifiable,
          CASE
            WHEN verdict=1 THEN 'SUPPORTS'
            WHEN verdict=2 THEN 'REFUTES'
          END as label, sentence.entity_id as entity, annotation.id as aid, annotation_verdict.id as vid, verdict_line.page, verdict_line.line_number, testing, isOracle,isReval, isTestMode,isOracleMaster,isDiscounted from annotation
        inner join claim on annotation.claim_id = claim.id
        left join annotation_verdict on annotation.id = annotation_verdict.annotation_id
        left join verdict_line on annotation_verdict.id = verdict_line.verdict_id
        inner join sentence on claim.sentence_id = sentence.id
        where isForReportingOnly = 0 and isTestMode = 0 and testing= 0

        """
        cursor.execute(sql)
        result = cursor.fetchall()

        for res in result:
            claim_evidence[res['id']].append(res)

finally:
    connection.close()


def process(ids):
    data = []
    print(len(ids))
    for id in ids:

        cl0 = claim_evidence[id][0]
        support_evidence, refute_evidence, not_enough_info_evidence = evidence(id)


        if len(set([ev["aid"] for ev in support_evidence])) > len(set([ev["aid"] for ev in not_enough_info_evidence])):
            not_enough_info_evidence = []

        if len(set([ev["aid"] for ev in refute_evidence])) > len(set([ev["aid"] for ev in not_enough_info_evidence])):
            not_enough_info_evidence = []

        if len(set([ev["aid"] for ev in refute_evidence])) < len(set([ev["aid"] for ev in not_enough_info_evidence])):
            support_evidence = []

        if len(set([ev["aid"] for ev in refute_evidence])) < len(set([ev["aid"] for ev in not_enough_info_evidence])):
            refute_evidence = []

        s_s = defaultdict(lambda:[])
        s_r = defaultdict(lambda:[])
        s_nei = defaultdict(lambda:[])

        for e in support_evidence:
            s_s[e['vid']].append((e['aid'],e['vid'],e['page'],e['line_number']))

        for e in refute_evidence:
            s_r[e['vid']].append((e['aid'],e['vid'],e['page'],e['line_number']))

        for e in not_enough_info_evidence:
            s_nei[e['vid']].append((e['aid'],e['vid'],e['page'],e['line_number']))

        if len(support_evidence):
            data.append({"id":id, "verifiable":"VERIFIABLE", "label":"SUPPORTS","claim":cl0['text'],"evidence":list(s_s.values())})
        if len(refute_evidence):
            data.append({"id": id, "verifiable":"VERIFIABLE", "label": "REFUTES", "claim": cl0['text'], "evidence":list(s_r.values())})
        if len(not_enough_info_evidence):
            data.append({"id": id, "verifiable":"NOT ENOUGH INFO", "label": None, "claim": cl0['text'], "evidence":list(s_nei.values())})
    return data


cnt=0

with open("train.ids.json", "r") as f:
    train_ids = json.load(f)
    print(train_ids[:10])
    train = process(train_ids)

with open("dev.ids.json", "r") as f:
    dev_ids = json.load(f)
    print(dev_ids[:10])
    dev = process(dev_ids)

with open("test.ids.json", "r") as f:
    test_ids = json.load(f)
    print(test_ids[:10])
    test = process(test_ids)

with open("train.jsonl","w+") as f:
    for line in train:
        f.write(json.dumps(line)+"\n")

with open("dev.jsonl","w+") as f:
    for line in dev:
        f.write(json.dumps(line)+"\n")

with open("test.jsonl","w+") as f:
    for line in test:
        f.write(json.dumps(line)+"\n")

print(len(train),len(dev),len(test))