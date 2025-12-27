import os
import sys

from SLM import Allocator

os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'
os.environ['APPDATA'] = r"D:\data\ImageDataManager"
os.environ['MONGODB_NAME'] = "files_db"
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from tqdm import tqdm

from SLM.files_db.components.CollectionRecordScheme import FileTypeRouter

from SLM.files_db.components.File_record_wraper import get_file_record_by_folder, FileRecord
from SLM.iterable.bach_builder import BatchBuilder


def index_folder_one_thread(path: str):
    path = path

    files = get_file_record_by_folder(path, recurse=True)

    file_list = list(files)

    def index_file(file: FileRecord):
        logger.info(f"start index:{file.full_path}")
        indexer = FileTypeRouter.instance().get_type_by_name(file.file_type).get_att("base_indexer", True)
        if indexer is not None:
            if not os.path.exists(file.full_path):
                return None
            rec_data = file.get_record_data()
            indexer.index(rec_data)
            if indexer.shared_data.get("item_indexed", False):
                return rec_data
        return None

    for butch in tqdm(file_list):
        res = index_file(butch)
        if res is not None:
            FileRecord.update_bulk([res])


def index_folder(query=None, worckers=4):
    if query is None:
        query = {}
    file_list = FileRecord.find(query)
    bach_b = BatchBuilder(file_list, 16)

    def index_file(file):
        if file is None:
            return
        indexer = FileTypeRouter.instance().get_type_by_name(file.file_type).get_att("base_indexer", True)
        if indexer is not None:
            # if file no exist on disck skip
            if not os.path.exists(file.full_path):
                return None
            rec_data = file.get_record_data()
            indexer.index(rec_data)
            if indexer.shared_data.get("item_indexed", False):
                return rec_data
        return None

    for butch in tqdm(bach_b.bach.values()):
        changed_files = []
        with ThreadPoolExecutor(worckers) as ex:
            futures = ex.map(index_file, butch)

        for future in futures:
            res = future
            if res is not None:
                changed_files.append(res)
        FileRecord.update_bulk(changed_files)


if __name__ == '__main__':
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    Allocator.init_modules()
    index_folder_one_thread(r'E:\rawimagedb\repository\nsfv repo\drawn\presort')
