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

claim_evidence = defaultdict(lambda: [])
page_evidence = defaultdict(lambda: defaultdict(lambda: []))

def evidence(claim_id):
    cl_support = [ev for ev in claim_evidence[claim_id] if ev["label"] == "SUPPORTS" ]
    cl_refutes = [ev for ev in claim_evidence[claim_id] if ev["label"] == "REFUTES" ]
    cl_notenough = [ev for ev in claim_evidence[claim_id]  if ev["verifiable"] == "NOT ENOUGH INFO"]
    return cl_support,cl_refutes,cl_notenough


def acceptable(id):
    s,r,n = evidence(id)
    return (len(set([ev["aid"] for ev in s])) == len(set([ev["aid"] for ev in n])) and len(s)) \
           or (len(set([ev["aid"] for ev in r])) == len(set([ev["aid"] for ev in n])) and len(r)) \
           or (len(s) and len(r))

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


def costs(cl_support,cl_refutes,cl_notenough):
    return len(cl_support),len(cl_refutes),len(cl_notenough)


def claims(page):
    claim_ids = list(filter(lambda id: not acceptable(id), page_evidence[page].keys()))

    cl_support = set([id for id in claim_ids if any(ev["label"] == "SUPPORTS" for ev in page_evidence[page][id])])
    cl_refutes = set([id for id in claim_ids if any(ev["label"] == "REFUTES" for ev in page_evidence[page][id])])
    cl_notenough = set([id for id in claim_ids if any(ev["verifiable"] == "NOT ENOUGH INFO" for ev in page_evidence[page][id])])


    for claim in cl_support:
        if claim in cl_notenough and len(set([ev["aid"] for ev in claim_evidence[claim] if ev["label"] == "SUPPORTS"])) > len(set([ev["aid"] for ev in claim_evidence[claim] if ev["verifiable"] == "NOT ENOUGH INFO"])):
            cl_notenough.remove(claim)

    for claim in cl_support:
        if claim in cl_notenough and len(set([ev["aid"] for ev in claim_evidence[claim] if ev["label"] == "REFUTES"])) > len(set([ev["aid"] for ev in claim_evidence[claim] if ev["verifiable"] == "NOT ENOUGH INFO"])):
            cl_notenough.remove(claim)


    for claim in cl_notenough:
        if claim in cl_support and len(set([ev["aid"] for ev in claim_evidence[claim] if ev["label"] == "SUPPORTS"])) < len(set([ev["aid"] for ev in claim_evidence[claim] if ev["verifiable"] == "NOT ENOUGH INFO"])):
            cl_support.remove(claim)

        if claim in cl_refutes and len(set([ev["aid"] for ev in claim_evidence[claim] if ev["label"] == "REFUTES"])) < len(set([ev["aid"] for ev in claim_evidence[claim] if ev["verifiable"] == "NOT ENOUGH INFO"])):
            cl_refutes.remove(claim)

    return cl_support,cl_refutes,cl_notenough




def balancing_heuristic(page):
    s,r,n = costs(*claims(page))
    return 4*s-r-n


pages = list(page_evidence.keys())
r = random.Random(33259)
r.shuffle(pages)

pq = []

for page in pages:
    heapq.heappush(pq,(balancing_heuristic(page),page))

test_pages = []
dev_pages = []
train_pages = []

train_sup = []
dev_sup  =[]
test_sup = []


train_ref = []
dev_ref  =[]
test_ref = []


train_not = []
dev_not  =[]
test_not = []


cs, cr, cn = 0,0,0
csd, crd, cnd = 0,0,0

while True:
    _,page = heapq.heappop(pq)

    cl = claims(page)
    s,r,n = costs(*cl)

    test_sup.extend(cl[0])
    test_ref.extend(cl[1])
    test_not.extend(cl[2])

    cs+= s
    cr+= r
    cn+= n

    test_pages.append(page)

    if cr>3333 and cn>3333:
        break

print(cs,cr,cn)


while True:
    _,page = heapq.heappop(pq)

    cl = claims(page)
    s,r,n = costs(*cl)

    dev_sup.extend(cl[0])
    dev_ref.extend(cl[1])
    dev_not.extend(cl[2])

    csd+= s
    crd+= r
    cnd+= n

    dev_pages.append(page)

    if crd>3333 and cnd>3333:
        break

while len(pq):
    _, page = heapq.heappop(pq)
    train_pages.append(page)

    s,r,n = claims(page)
    train_sup.extend(s)
    train_ref.extend(r)
    train_not.extend(n)

print(csd,crd,cnd)

test_s_remove = len(test_sup)-3333
test_r_remove = len(test_ref)-3333
test_n_remove = len(test_not)-3333

dev_s_remove = len(dev_sup)-3333
dev_r_remove = len(dev_ref)-3333
dev_n_remove = len(dev_not)-3333


r = random.Random(2149)
r.shuffle(test_sup)
r = random.Random(2129)
r.shuffle(test_ref)
r = random.Random(2649)
r.shuffle(test_not)

dropped_test = test_sup[:test_s_remove]+test_ref[:test_r_remove]+test_not[:test_n_remove]

test_sup = test_sup[test_s_remove:]
test_ref = test_ref[test_r_remove:]
test_not = test_not[test_n_remove:]

r = random.Random(3149)
r.shuffle(dev_sup)
r = random.Random(3129)
r.shuffle(dev_ref)
r = random.Random(3649)
r.shuffle(dev_not)

dropped_dev = dev_sup[:dev_s_remove]+dev_ref[:dev_r_remove]+dev_not[:dev_n_remove]

dev_sup = dev_sup[dev_s_remove:]
dev_ref = dev_ref[dev_r_remove:]
dev_not = dev_not[dev_n_remove:]

print(len(test_sup),len(test_ref),len(test_not))
print(len(dev_sup),len(dev_ref),len(dev_not))
print(len(train_sup),len(train_ref),len(train_not))

print(sum((len(test_sup),len(test_ref),len(test_not),len(dev_sup),len(dev_ref),len(dev_not),len(train_sup),len(train_ref),len(train_not))))


train = train_sup+train_ref+train_not
dev = dev_sup+dev_ref+dev_not
test = test_sup+test_ref+test_not


r = random.Random(13842)
r.shuffle(train)

r = random.Random(13843)
r.shuffle(dev)

r = random.Random(13844)
r.shuffle(test)

with open("train.ids.json","w+") as f:
    json.dump(train,f)
with open("dev.ids.json","w+") as f:
    json.dump(dev,f)
with open("test.ids.json","w+") as f:
    json.dump(test,f)