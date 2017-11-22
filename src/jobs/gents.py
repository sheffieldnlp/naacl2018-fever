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

claim_evidence = defaultdict(lambda: [])
page_evidence = defaultdict(lambda: [])
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
          END as label, verdict_line.page, verdict_line.line_number, testing, isOracle,isReval, isTestMode,isOracleMaster,isDiscounted from annotation
        inner join claim on annotation.claim_id = claim.id
        left join annotation_verdict on annotation.id = annotation_verdict.annotation_id
        left join verdict_line on annotation_verdict.id = verdict_line.verdict_id
        where isForReportingOnly = 0 and isTestMode = 0 and testing= 0
        
        """
        cursor.execute(sql)
        result = cursor.fetchall()



        for res in result:
            claim_evidence[res['id']].append(res)
            page_evidence[res['page']].append(res)

finally:
    connection.close()



ts_refutes = []
ts_support = []
ts_notenough = []

pages = list(page_evidence.keys())
r = random.Random(123)
r.shuffle(pages)

for page in pages:
    print(page)