from collections import defaultdict

import pickle
import pymysql.cursors
import os

import random

from common.dataset.corpus import Corpus
from common.util.log_helper import LogHelper
def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]


lut = dict()

LogHelper.setup()

pages = Corpus("page", "data/fever", 50,lambda x :x )
for page,doc in pages:
    lut[page] = doc



# Connect to the database
connection = pymysql.connect(host=os.getenv("DB_HOST", "localhost"),
                             user=os.getenv("DB_USER", "root"),
                             password=os.getenv("DB_PASS", ""),
                             db=os.getenv("DB_SCHEMA", "fever"),
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


data = []
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
            data.append(dict(res))
finally:
    connection.close()



r = random.Random()

r.shuffle(data)

done = []
for datum in data[:500]:
    page = datum['entity']
    try:
        datum["original_page"] = lut[preprocess(page)]
        datum["verdict_page"] = lut[preprocess(datum['page'])] if datum['page'] is not None else None
        done.append(datum)
    except:
        continue
import json


with open("data/dump1.json","w+") as f:
    json.dump(done[:100],f)

with open("data/dump2.json", "w+") as f:
    json.dump(done[100:200], f)

with open("data/dump3.json", "w+") as f:
    json.dump(done[200:300], f)

with open("data/dump4.json", "w+") as f:
    json.dump(done[300:400], f)

with open("data/dump5.json", "w+") as f:
    json.dump(done[400:500], f)


