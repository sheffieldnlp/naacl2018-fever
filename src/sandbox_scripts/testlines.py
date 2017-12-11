from retrieval.fever_doc_db import FeverDocDB

db = FeverDocDB("data/fever/fever.db")
print(db.get_doc_lines("United_States"))