from collections import defaultdict

import pymysql
import os

from statsmodels.stats.inter_rater import fleiss_kappa

connection = pymysql.connect(host=os.getenv("DB_HOST", "localhost"),
                             user=os.getenv("DB_USER", "root"),
                             password=os.getenv("DB_PASS", ""),
                             db=os.getenv("DB_SCHEMA", "fever_final"),
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

claims_dict = defaultdict(lambda:[])
with connection.cursor() as cursor:

    cursor.execute(
        """
select id, user, verifiable,verdict from (
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
  END as verdict, annotation.user, verdict_line.page, verdict_line.line_number, annotation.id as aid, testing, isOracle,isReval, isTestMode,isOracleMaster,isDiscounted from annotation
inner join claim on annotation.claim_id = claim.id
left join annotation_verdict on annotation.id = annotation_verdict.annotation_id
left join verdict_line on annotation_verdict.id = verdict_line.verdict_id
where isForReportingOnly = 0 and isTestMode = 0 and testing= 0 and isReval=1)
as a group by id, user

        """)


    def row_ct(row):
        rowct = []
        for i in range(3):
            rowct.append(row.count(i))

        return rowct

    claims = cursor.fetchall()

    for claim in claims:
        if claim['verifiable'] == "NOT ENOUGH INFO":
            claims_dict[claim['id']].append(0)
        elif claim['verifiable'] == "VERIFIABLE":
            claims_dict[claim['id']].append(1 if claim["verdict"]=="SUPPORTS" else 2)


    fkt1 = [row_ct(claims_dict[key]) for key in claims_dict if len(claims_dict[key]) == 5]
    print(fkt1)
    print(len(fkt1))
    print(fleiss_kappa(fkt1))