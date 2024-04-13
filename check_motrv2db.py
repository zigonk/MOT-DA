import json

DET_DB_PATH = 'data/Dataset/mot/det_db/det_db_motrv2.json'

det_db = json.load(open(DET_DB_PATH, 'r'))
for key in det_db.keys():
    print(key)