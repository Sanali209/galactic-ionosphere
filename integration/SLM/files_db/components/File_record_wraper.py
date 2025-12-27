import fnmatch
import os
import re
from concurrent.futures import ThreadPoolExecutor

from SLM.appGlue.core import Allocator
from SLM.files_data_cache.imagedatacache import ImageDataCacheManager
from SLM.files_data_cache.thumbnail import ImageThumbCache
from SLM.iterable.bach_builder import BatchBuilder
from SLM.mongoext.MongoClientEXT_f import MongoClientExt
from SLM.vision.imagetotext.ImageToLabel import ImageToLabel

from tqdm import tqdm

from SLM.FuncModule import sanitize_file_path

from SLM.files_db.components.collectionItem import CollectionRecord
from SLM.appGlue.iotools.pathtools import get_files
from SLM.appGlue.progress_visualize import ProgressManager
from SLM.mongoext.wraper import MongoRecordWrapper, FieldPropInfo
from loguru import logger


class FileRecord(CollectionRecord):
    itemType: str = FieldPropInfo('item_type', str, 'FileRecord')
    """override itemType of CollectionRecord"""

    file_type: str = FieldPropInfo('file_type', str, None)

    @staticmethod
    def find_by_query(query_string: str):
        """
        Find records by a custom query string.
        e.g., "tags=tag1,tag2; name=file.txt"
        e.g., "tags REGEX ^character"
        """
        query_parts = query_string.split(';')
        mongo_query = {}
        for part in query_parts:
            part = part.strip()
            if not part:
                continue

            if ' REGEX ' in part:
                key, pattern = part.split(' REGEX ', 1)
                key = key.strip()
                pattern = pattern.strip().strip('"')
                mongo_query[key] = {'$regex': pattern, '$options': 'i'}  # Case-insensitive
            elif '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key == 'tags':
                    tags = [t.strip() for t in value.split(',')]
                    mongo_query[key] = {'$all': tags}
                else:
                    mongo_query[key] = value
        
        return FileRecord.find(mongo_query)

    @staticmethod
    def kwargs_to_md(kwargs):
        kw_str = ""
        for key, value in kwargs.items():
            kw_str += f"{key}={value},"
        kw_md5 = hash(kw_str)
        return kw_md5

    def get_ai_expertise(self, expertise_type, expertise_name, **kwargs):
        ver = None
        if expertise_type == "image-text":
            ver = ImageToLabel().get_backend_version(expertise_name)
        list_expertise = self.ai_expertise
        if list_expertise is None:
            list_expertise = []
            self.ai_expertise = list_expertise
        for expertise in list_expertise:
            if expertise['name'] == expertise_name and expertise['type'] == expertise_type:
                if expertise['version'] == ver:
                    if expertise['kwargs'] == FileRecord.kwargs_to_md(kwargs):
                        return expertise
                else:
                    list_expertise.remove(expertise)
                    break
        # create new expertise
        new_expertise = {'name': expertise_name, 'type': expertise_type, 'version': ver, 'kwargs': "", "data": {}}

        if expertise_type == "image-text":
            kwargs_dat = FileRecord.kwargs_to_md(kwargs)
            val = ImageToLabel().get_label_from_path(self.full_path, expertise_name, **kwargs)
            new_expertise['data'] = val
            new_expertise['kwargs'] = kwargs_dat

        list_expertise.append(new_expertise)
        self.ai_expertise = list_expertise
        return new_expertise

    # region file functions
    def get_thumb(self, size=None):
        if size is None:
            size = "medium"
        return ImageThumbCache.instance().get_thumb(self.full_path, size)

    def refresh_thumb(self):
        ImageThumbCache.instance().refresh_thumb(self.full_path)

    def move_to_folder(self, new_folder):
        """
        Move file to new folder and update local_path in db
        :param new_folder:
        :return:
        """
        new_path = os.path.join(new_folder, self.name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        if self.full_path is not None:
            source_path = str(self.full_path)
            if source_path != new_path:
                os.rename(source_path, new_path)
                self.local_path = new_folder

    # endregion
    @staticmethod
    def add_file_record_from_path(path):

        path = sanitize_file_path(path)
        res = FileRecord.find_one({'local_path': os.path.dirname(path),
                                   'name': os.path.basename(path)})
        return res

    @staticmethod
    def get_record_by_path(path):
        #replace in path to "\\"
        path = sanitize_file_path(path)
        query = {'local_path': os.path.dirname(path),
                 'name': os.path.basename(path)}
        res = FileRecord.find_one(query)
        return res

    @staticmethod
    def add_file_records_from_folder(folder_path, black_list=None):
        if black_list is None:
            black_list = ['*.db*', '*.json', '*.ini', '*.jsonl']
        records = []
        files = get_files(folder_path, ['*'])
        flen = len(files)
        prog_man = ProgressManager.instance()
        prog_man.max_progress = flen
        prog_man.set_description('add files to db')
        for file in tqdm(files, desc='add file records'):
            prog_man.step(f'add file {file}')
            in_black_list = False
            for black in black_list:
                if fnmatch.fnmatch(file, black):
                    in_black_list = True
                    continue
            if in_black_list:
                continue
            # check is file exist in db (local_path and name) if exist
            query = {'local_path': os.path.dirname(file),
                     'name': os.path.basename(file)}  # return record with this local_path and name
            # need indexes for this query
            record = FileRecord.find_one(query)
            if record is None:
                rec_date = FileRecord.create_record_data(file_path=file)
                records.append(rec_date)
        if len(records) > 0:
            FileRecord.collection().insert_many(records)

    @staticmethod
    def delete_all_file_records():
        FileRecord.collection().delete_many({'item_type': 'FileRecord'})
        # todo send message to message system  for handle dependant systems

    def delete(self):
        """delete record from db and file from disk"""
        if self.full_path is not None:
            try:
                if os.path.exists(self.full_path):
                    os.remove(self.full_path)
            except Exception as e:
                logger.error(f"Error deleting file {self.full_path} - {e}")

        self.delete_rec()

    def exists(self):
        """Check if file exist in db and on disk"""
        if self.full_path is None:
            return False
        if not os.path.exists(self.full_path):
            logger.warning(f"File {self.full_path} not found on disk")
            return False
        return True

    @classmethod
    def create_record_data(cls, **kwargs):
        from SLM.files_db.components.CollectionRecordScheme import FileTypeRouter
        """
        Create file record in db not use this method directly
        :param kwargs:
        :key file_path: file full path
        :return:
        """
        file_path = kwargs["file_path"]
        data = super().create_record_data(**kwargs)
        record = {
            'name': os.path.basename(file_path),
            'local_path': os.path.dirname(file_path),
            'extension': os.path.splitext(file_path)[1],

        }
        f_router = FileTypeRouter.instance()
        type_ = f_router.get_type_by_path(file_path)
        record['file_type'] = type_.name

        try:
            record['size'] = os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error getting file size {file_path} - {e}")
            record['size'] = 0
            # todo check file corrupted and not add to db add too special collection in db
            record['file_corrupted'] = True
        record['file_content_md5'] = ImageDataCacheManager().instance().path_to_md5(file_path)
        data.update(record)
        return data

    @property
    def full_path(self):
        data = self.get_record_data()
        if data is None or len(data.keys()) == 0:
            return None

        return os.path.join(data['local_path'], data['name'])

    @full_path.setter
    def full_path(self, value):
        data = {'local_path': os.path.dirname(value), 'name': os.path.basename(value)}
        self.set_record_data(data)


def get_file_record_by_folder(folder_path, recurse=False, filters=None):
    # replace / to \
    folder_path = str(folder_path).replace('/', '\\')


    # todo bug: if similar folder ends
    if recurse:
        query = {'local_path': {'$regex': '^' + re.escape(folder_path)}}
    else:
        query = {'local_path': folder_path}

    if filters is not None:
        for key, value in filters.items():
            query[key] = value

    return FileRecord.find(query)


def refind_exist_files(path,add_exist=True):
    files = get_files(path, ['*'])

    def find_file(file):
        content_md5 = ImageDataCacheManager.instance().path_to_md5(file)
        if content_md5 is None:
            logger.error(f"Error getting md5 for file {file}")
            return None
        rec = FileRecord.find_one({'file_content_md5': content_md5})
        if rec is None:
            logger.warning(f"File not found in db {file}")
            if add_exist:
                # todo: implement filter
                rec_date = FileRecord.create_record_data(file_path=file)
                FileRecord.collection().insert_one(rec_date)
            return None
        rec.full_path = file
        return rec

    bach_b = BatchBuilder(files, 16)
    for bach in tqdm(bach_b.bach.values(), desc='find files in db'):
        with ThreadPoolExecutor(4) as ex:
            ex.map(find_file, bach)


def remove_files_record_by_mach_pattern(path, regex_mach_pathern):
    query = {'local_path': {"$regex": '^' + re.escape(path)},
             'name': {'$regex': regex_mach_pathern}}
    records = FileRecord.find(query)
    for record in tqdm(records):
        record.delete_rec()
