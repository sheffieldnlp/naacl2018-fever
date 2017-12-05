import json
import random
from collections import defaultdict

import pymysql.cursors
import os
# Connect to the database
import sys

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



ts_refutes = []
ts_support = []
ts_notenough = []

ds_refutes = []
ds_support = []
ds_notenough = []

pages = list(page_evidence.keys())
r = random.Random(12453)
r.shuffle(pages)


tsr_done = False
tsn_done = False
tss_done = False


target = 3333


def fits_s(page, target, supports):
    claim_ids = page_evidence[page].keys()
    cl_support = [id for id in claim_ids if any(ev["label"] == "SUPPORTS" for ev in page_evidence[page][id])]
    return len(supports)+len(cl_support) <= target

def fits_r(page, target, refutes):
    claim_ids = page_evidence[page].keys()
    cl_refutes = [id for id in claim_ids if any(ev["label"] == "REFUTES" for ev in page_evidence[page][id])]
    return len(refutes)+len(cl_refutes) <= target


def fits_n(page, target, notenough):
    claim_ids = page_evidence[page].keys()
    cl_notenough = [id for id in claim_ids if any(ev["verifiable"] == "NOT ENOUGH INFO" for ev in page_evidence[page][id])]
    return len(notenough)+len(cl_notenough) <= target






def add(page, added, supports, refutes, notenough):
    claim_ids = page_evidence[page].keys()

    cl_support = [id for id in claim_ids if any(ev["label"] == "SUPPORTS" for ev in page_evidence[page][id])]
    cl_refutes = [id for id in claim_ids if any(ev["label"] == "REFUTES" for ev in page_evidence[page][id])]
    cl_notenough = [id for id in claim_ids if any(ev["verifiable"] == "NOT ENOUGH INFO" for ev in page_evidence[page][id])]

    supports.extend(cl_support)
    refutes.extend(cl_refutes)
    notenough.extend(cl_notenough)

    added.append(page)




def costs(page):
    claim_ids = page_evidence[page].keys()

    cl_support = set([id for id in claim_ids if any(ev["label"] == "SUPPORTS" for ev in page_evidence[page][id])])
    cl_refutes = set([id for id in claim_ids if any(ev["label"] == "REFUTES" for ev in page_evidence[page][id])])
    cl_notenough = set([id for id in claim_ids if any(ev["verifiable"] == "NOT ENOUGH INFO" for ev in page_evidence[page][id])])

    return len(cl_support),len(cl_refutes),len(cl_notenough)

added = []

print("3")
print("{0} {0} {0}".format(target))
print(len(pages))
for page in pages:
    print("{0} {1} {2} 1".format(*costs(page)))



