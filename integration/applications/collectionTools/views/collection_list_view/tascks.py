from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from SLM.files_db.components.CollectionRecordScheme import FileTypeRouter

from SLM.destr_worck.bg_worcker import BGTask
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.iterable.bach_builder import BatchBuilder


class add_folder_to_db_task(BGTask):
    def __init__(self, path):
        super().__init__()
        self.token = "add_folder_to_db_task"
        self.cancel_token = True
        self.path = path

    def task_function(self, ):

        FileRecord.add_file_records_from_folder( self.path)
        yield "done"

class index_db_task(BGTask):
    def __init__(self):
        super().__init__()
        self.token = "add_folder_to_db_task"
        self.cancel_token = True


    def task_function(self):

        file_list = FileRecord.find({})
        bach_b = BatchBuilder(file_list, 16)

        def index_file(file: FileRecord):
            indexer = FileTypeRouter.instance().get_type_by_name(file.file_type).get_att("base_indexer",True)
            if indexer is not None:
                rec_data = file.get_record_data()
                indexer.index(rec_data)
                if indexer.shared_data.get("item_indexed", False):
                    return rec_data
            return None

        for butch in tqdm(bach_b.bach.values()):
            changed_files = []
            with ThreadPoolExecutor() as ex:
                futures = ex.map(index_file, butch)

            for future in futures:
                res = future
                if res is not None:
                    changed_files.append(res)
            FileRecord.update_bulk(changed_files)
        yield "done"
