# todo realize and wrap to AppAction
import os

from tqdm import tqdm



from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.mongoext.MongoClientEXT_f import MongoClientExt


def delete_existing(db_client: MongoClientExt):

    records = FileRecord.collection().find({})

    for x in tqdm(records, desc='delete existing files'):
        file_rec = FileRecord(x['_id'])
        path = file_rec.full_path
        if not os.path.exists(path):
            print(f"deleting {path}")
            file_rec.delete_rec()


