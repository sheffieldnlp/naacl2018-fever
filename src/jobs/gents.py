import json
import random
from collections import defaultdict

import pymysql.cursors
import os
# Connect to the database
connection = pymysql.connect(host=os.getenv("DB_HOST","localhost"),
                             user=os.getenv("DB_USER","root"),
                             password=os.getenv("DB_PASS",""),
                             db=os.getenv("DB_SCHEMA","fever"),
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

claim_evidence = defaultdict(lambda:[])
page_evidence = defaultdict(lambda: defaultdict(lambda: []))
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
          END as label, sentence.entity_id as entity, annotation.id as aid, verdict_line.page, verdict_line.line_number, testing, isOracle,isReval, isTestMode,isOracleMaster,isDiscounted from annotation
        inner join claim on annotation.claim_id = claim.id
        left join annotation_verdict on annotation.id = annotation_verdict.annotation_id
        left join verdict_line on annotation_verdict.id = verdict_line.verdict_id
        inner join sentence on claim.sentence_id = sentence.id
        where isForReportingOnly = 0 and isTestMode = 0 and testing= 0
        
        """
        cursor.execute(sql)
        result = cursor.fetchall()



        for res in result:
            page_evidence[res['entity']][res['id']].append(res)
            claim_evidence[res['id']].append(res)

finally:
    connection.close()



ts_refutes = set()
ts_support = set()
ts_notenough = set()

ds_refutes = set()
ds_support = set()
ds_notenough = set()

pages = list(page_evidence.keys())
r = random.Random(12453)
r.shuffle(pages)


tsr_done = False
tsn_done = False
tss_done = False

for page in pages:
    print(page)

    if page is None:
        continue


    if len(ts_support) < 3333:
        for claim in page_evidence[page].keys():
            for evidence in page_evidence[page][claim]:
                if evidence["label"] == "SUPPORTS":
                    ts_support.add(claim)
    else:
        tss_done = True

    if len(ts_refutes) < 3333:
        for claim in page_evidence[page].keys():
            for evidence in page_evidence[page][claim]:
                if evidence["label"] == "REFUTES":
                    ts_refutes.add(claim)
    else:
        tsr_done = True

    if len(ts_notenough) < 3333:
        for claim in page_evidence[page].keys():
            for evidence in page_evidence[page][claim]:
                if evidence["verifiable"] == "NOT ENOUGH INFO":
                    ts_notenough.add(claim)
                    print(len(ts_notenough))
    else:
        tsn_done = True

    if len(ds_support) < 3333 and tss_done and tsr_done and tsn_done:
        for claim in page_evidence[page].keys():
            for evidence in page_evidence[page][claim]:
                if evidence["label"] == "SUPPORTS":
                    ds_support.add(claim)


    if len(ds_refutes) < 3333 and tss_done and tsr_done and tsn_done:
        for claim in page_evidence[page].keys():
            for evidence in page_evidence[page][claim]:
                if evidence["label"] == "REFUTES":
                    ds_refutes.add(claim)

    if len(ds_notenough) < 3333 and tss_done and tsr_done and tsn_done:
        for claim in page_evidence[page].keys():
            for evidence in page_evidence[page][claim]:
                if evidence["verifiable"] == "NOT ENOUGH INFO":
                    ds_notenough.add(claim)


print(len(ts_support))
print(len(ts_refutes))
print(len(ts_notenough))

print(len(ds_support))
print(len(ds_refutes))
print(len(ds_notenough))


with(open("data/fever/fever.test.jsonl","w+")) as f:
    added = set()

    for claim in ts_support:
        f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"SUPPORTED","evidence":[(ev["aid"],ev["page"],ev["line_number"]) for ev in claim_evidence[claim] if ev["label"]=="SUPPORTS"]})+"\n")
        added.add(claim)

    for claim in ts_refutes:
        f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"REFUTED","evidence":[(ev["aid"],ev["page"],ev["line_number"]) for ev in claim_evidence[claim] if ev["label"]=="REFUTES"]})+"\n")
        added.add(claim)

    for claim in ts_notenough:
        f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"NOT ENOUGH INFO","evidence":[(ev["aid"]) for ev in claim_evidence[claim] if ev["verifiable"]=="NOT ENOUGH INFO"]})+"\n")
        added.add(claim)


    for clid in added:
        del claim_evidence[clid]

with(open("data/fever/fever.dev.jsonl","w+")) as f:
    added = set()

    for claim in ds_support:
        f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"SUPPORTED","evidence":[(ev["aid"],ev["page"],ev["line_number"]) for ev in claim_evidence[claim] if ev["label"]=="SUPPORTS"]})+"\n")
        added.add(claim)

    for claim in ds_refutes:
        f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"REFUTED","evidence":[(ev["aid"],ev["page"],ev["line_number"]) for ev in claim_evidence[claim] if ev["label"]=="REFUTES"]})+"\n")
        added.add(claim)

    for claim in ds_notenough:
        f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"NOT ENOUGH INFO","evidence":[(ev["aid"]) for ev in claim_evidence[claim] if ev["verifiable"]=="NOT ENOUGH INFO"]})+"\n")
        added.add(claim)

    for clid in added:
        del claim_evidence[clid]

with(open("data/fever/fever.train.jsonl","w+")) as f:
    added = set()

    for claim in claim_evidence.keys():
        if len([ev for ev in claim_evidence[claim] if ev["label"]=="SUPPORTS"]) > 0:
            f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"SUPPORTED","evidence":[(ev["aid"],ev["page"],ev["line_number"]) for ev in claim_evidence[claim] if ev["label"]=="SUPPORTS"]})+"\n")
        if len([ev for ev in claim_evidence[claim] if ev["label"] == "REFUTES"]) > 0:
            f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"REFUTED","evidence":[(ev["aid"],ev["page"],ev["line_number"]) for ev in claim_evidence[claim] if ev["label"]=="REFUTES"]})+"\n")
        if len([ev for ev in claim_evidence[claim] if ev["verifiable"] == "NOT ENOUGH INFO"]) > 0:
            f.write(json.dumps({"id":claim,"claim":claim_evidence[claim][0]["text"],"verdict":"NOT ENOUGH INFO","evidence":[(ev["aid"]) for ev in claim_evidence[claim] if ev["verifiable"]=="NOT ENOUGH INFO"]})+"\n")
