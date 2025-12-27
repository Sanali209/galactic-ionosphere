import json
import os.path
from typing import Any

from loguru import logger
from tqdm import tqdm

from SLM.appGlue.DAL.datalist2 import MongoDataModel, DataViewCursor, MongoDataQuery, DataListModelBase
from SLM.appGlue.core import Allocator
from SLM.files_data_cache.imagedatacache import ImageDataCacheManager
from SLM.files_db.annotation_tool.annotation_export import DataSetExporterManager
from SLM.files_db.components.File_record_wraper import FileRecord, get_file_record_by_folder
from SLM.appGlue.iotools.pathtools import get_files
from SLM.mongoext.MongoClientEXT_f import MongoClientExt

from SLM.mongoext.wraper import MongoRecordWrapper, FieldPropInfo

all_jobTypes = ["binary/image", "multiclass/image", "multilabel/image", "image_object_detection", "image_segmentation",
                "image_to_text"]


# todo for improve performance need rewrite annotate list mechanism

class AnnotationRecord(MongoRecordWrapper):
    value: Any = FieldPropInfo(name="value", field_type=Any, default=None)
    parent_id: str = FieldPropInfo(name="parent_id", field_type=str, default=None)
    file_id: str = FieldPropInfo(name="file_id", field_type=str, default=None)

    @property
    def parent(self):
        parent_id = self.get_field_val('parent_id')
        return self.none_or(parent_id, AnnotationJob(parent_id))

    @parent.setter
    def parent(self, val):
        self.set_field_val('parent_id', val._id)

    @property
    def file(self) -> FileRecord:
        file_id = self.get_field_val('file_id')
        return self.none_or(file_id, FileRecord(file_id))

    @file.setter
    def file(self, val):
        self.set_field_val("file_id", val._id)


class AnnJobDataQuery(MongoDataQuery):
    def __init__(self, data_model: MongoDataModel, an_job_id):
        super().__init__(data_model)
        self.aggregation_pipline = {}
        self.job_id = an_job_id

    def get_by_query(self, skip=0, limit=100, sort="default", sort_alg=None):
        try:
            records = AnnotationRecord.find({"parent_id": self.job_id, "value": None})
            records = list(records)
            if skip != 0:
                records = records[skip:]
            if limit != 0:
                records = records[:limit]
            return records
        except Exception as e:
            logger.error(f"Error getting records: {e}")
            return []

    def count_all(self):
        return AnnotationRecord.collection().count_documents({"parent_id": self.job_id, "value": None})


class AJDataModel(DataListModelBase):
    def __init__(self, job_id):
        super().__init__()
        self.dataQuery = AnnJobDataQuery(self, job_id)

    def append(self, item):
        result = self._collection.insert_one(item)
        super().append(item)
        return result.inserted_id

    def remove(self, item):
        query = {"_id": item["_id"]}
        result = self._collection.delete_one(query)
        if result.deleted_count > 0:
            super().remove(item)

    def clear(self):
        result = self._collection.delete_many({})
        if result.deleted_count > 0:
            super().clear()


class AnnotationJob(MongoRecordWrapper):
    not_annotated = FieldPropInfo("not_annotated", list, [])
    name: str = FieldPropInfo("name", str, None)
    type: str = FieldPropInfo("type", str, "multiclass/image")
    choices: object = FieldPropInfo("choices", object, None)

    def __str__(self):
        return self.name + " " + self.type

    @staticmethod
    def get_by_name(job_name):

        query = {'name': job_name}
        record = AnnotationJob.find_one(query)
        if record is None:
            job_data = Allocator.config.def_annotation_jobs.dict.get(job_name, None)
            job: AnnotationJob = AnnotationJob.get_or_create(name=job_name)
            job.type = job_data['type']
            job.choices = job_data['chooses']
            record = job
        return record

    def __init__(self, oid):
        super().__init__(oid)
        self.job_data = AJDataModel(self._id)
        self.coll_view = DataViewCursor(self.job_data)
        self.coll_view.items_per_page = 0
        #self.coll_view.all_items_count()
        # need transform for quering

    def remove_annotation_dublicates(self):
        """
        remove dublicates from job
        :return:
        """
        records = self.get_all_annotated()
        for record in tqdm(records):
            query = {'parent_id': self._id, 'file_id': record.file._id}
            result = AnnotationRecord.find(query)
            if len(result) > 1:
                for i in range(1, len(result)):
                    print("delete")
                    result[i].delete()

    def remove_annotation_dublicates2(self):
        """
        remove dublicates from job
        :return:
        """
        query = {'parent_id': self._id}
        result = AnnotationRecord.find(query)
        progress = tqdm(result)
        path_dict = {}
        for record in progress:
            progress.set_description(f"processing {record.file.full_path}")
            if record.file.full_path in path_dict:
                print("delete")
                record.delete_rec()
            else:
                path_dict[record.file.full_path] = 1

    def remove_broken_annotations(self):
        """
        remove broken annotations
        :return:
        """
        records = self.get_all_annotated()
        for record in tqdm(records):
            if record.file is None or record.file.full_path is None:
                record.delete_rec()
                continue
            if not os.path.exists(record.file.full_path):
                record.delete_rec()

    def mark_not_annotated(self, file: FileRecord):
        record = AnnotationRecord.get_or_create(parent_id=self._id, file_id=file._id)

    def mark_not_annotated_in_directory(self, path):
        f_records = get_file_record_by_folder(path, recurse=True)
        for record in tqdm(f_records):
            self.mark_not_annotated(record)

    def file_exist_in_annotation(self, file_id):
        query = {'parent_id': self._id
            , 'file_id': file_id}
        result = AnnotationRecord.find_one(query)
        if result is not None:
            return True
        return result is not None

    def move_next_annotation_item(self):
        self.coll_view.move_next()
        #return self.coll_view.get_current_item()

    def move_prev_annotation_item(self):
        self.coll_view.move_previous()
        #return self.coll_view.get_current_item()

    def annotate(self, value, override_exist=True):
        current_item: AnnotationRecord = self.coll_view.get_current_item()
        if current_item is None:
            return
        if current_item.value is None or override_exist:
            current_item.value = value

    def annotate_file(self, file: FileRecord, value, override_annotation=False):
        record = self.get_annotation_record(file)
        if record is None:
            return
        if record.value is None or override_annotation:
            record.value = value

    def remove_annotation_record(self, file: FileRecord):
        query = {'parent_id': self._id
            , 'file_id': file._id}
        AnnotationRecord.collection().delete_one(query)

    def get_annotation_record(self, file: FileRecord):
        query = {'parent_id': self._id
            , 'file_id': file._id}
        record = AnnotationRecord.find_one(query)
        return record

    def count_annotated_items(self, value=None):
        query = {'parent_id': self._id, 'value': value}
        return AnnotationRecord.collection().count_documents(query)

    def export_to_dataset(self, path, _format):
        exporter = DataSetExporterManager().get_exporter_by_name(_format)
        if exporter is None:
            raise Exception("Exporter not found")
        exporter.ExportToDataset(path, self)

    def get_all_annotated(self) -> list[AnnotationRecord]:
        """
        Get list of all annotation records
        :return:
        """
        query = {'parent_id': self._id, 'value': {'$ne': None}}
        result = AnnotationRecord.find(query)
        return list(result)

    def get_all_not_annotated(self) -> list[AnnotationRecord]:
        """
        Get list of all annotation records
        :return:
        """
        query = {'parent_id': self._id, 'value': None}
        result = AnnotationRecord.find(query)
        return list(result)

    def get_ann_records_by_label(self, label) -> list[AnnotationRecord]:
        query = {'parent_id': self._id, 'value': label}
        result = AnnotationRecord.find(query)
        return list(result)

    def clear_not_annotated_list(self):
        records = AnnotationRecord.find({'parent_id': self._id, 'value': None})
        for record in tqdm(records):
            record.delete_rec()

    def rename_annotation_label(self, old_name, new_name):
        """
        rename annotation label
        :param param:
        :param param1:
        :return:
        """
        if not isinstance(self.choices, list):
            logger.error("AnnotationJob.choices is not list")
            return
        current_choices: list = self.choices
        if current_choices is None:
            return
        if old_name not in current_choices:
            return
        if old_name in current_choices:
            current_choices.remove(old_name)
            current_choices.append(new_name)
        self.choices = current_choices
        records = AnnotationRecord.find({'parent_id': self._id, 'value': old_name})
        for record in tqdm(records):
            record.value = new_name

    def add_annotation_choices(self, new_choises):
        """
        add new choices to annotation job
        :param new_choises:
        :return:
        """
        if not isinstance(self.choices, list):
            logger.error("AnnotationJob.choices is not list")
            return
        current_choices: list = self.choices
        if current_choices is None:
            return
        for choice in new_choises:
            if choice not in current_choices:
                current_choices.append(choice)
        self.choices = current_choices

    def clear_job(self):
        """
        clear job
        :return:
        """
        records = AnnotationRecord.find({'parent_id': self._id})
        for record in tqdm(records):
            record.delete_rec()


class AnotationMultilableJob(AnnotationJob):
    type: str = FieldPropInfo("type", str, "multilabel/image")

    def set_choises(self, choises):
        """
        set choises for multilabel job
        :param choises:
        :return:
        """
        if not isinstance(choises, list):
            logger.error("AnnotationJob.choices is not list")
            return
        self.choices = choises


class SLMAnnotationClient:

    def get_all_jobs(self, records_filter=None) -> list[AnnotationJob]:
        if records_filter is None:
            records_filter = {}
        result = AnnotationJob.find(records_filter)
        jobs = []
        for record in result:
            jobs.append(record)
        return jobs

    def restore_from_json(self, path):
        with open(path, 'r') as file:
            json_data = json.load(file)
        for job in tqdm(json_data['jobs']):
            job_name = job['name']
            job_data = job['data']
            annotation_job = AnnotationJob.get_by_name(job_name)
            if annotation_job is None:
                continue
            remove_not_ann = []
            for record in tqdm(job_data):

                md_5 = record[0]
                label = record[1]
                query = {'file_content_md5': md_5}
                file_record = FileRecord.find_one(query)
                if file_record is None:
                    continue
                # todo need batch update remove from not annotated is slow in step transferring from and to database
                if annotation_job.file_exist_in_annotation(file_record._id):
                    continue
                data = {"file_id": file_record._id, "parent_id": annotation_job._id,
                        "value": label}
                AnnotationRecord.collection().insert_one(data)
                remove_not_ann.append(file_record._id)
            data = annotation_job.get_record_data()
            _list = data.get("not_annotated", [])
            _list = set(_list)
            for item in tqdm(remove_not_ann):
                if item in _list:
                    _list.remove(item)
            data["not_annotated"] = list(_list)
            annotation_job.set_record_data(data)

    def save_to_json(self, path):
        jobs = self.get_all_jobs()
        data = {'jobs': []}
        for job in jobs:
            job_data = {}
            for record in tqdm(AnnotationRecord.find({'parent_id': job._id})):
                file: FileRecord = record.file
                md_5 = file.file_content_md5
                if md_5 is None:
                    continue
                if record.value is None:
                    continue

                job_data[md_5] = record.value
            job_data_list = list([(k, v) for k, v in job_data.items()])
            data['jobs'].append({'name': job.name, 'data': job_data_list})
        with open(path, 'w') as file_:
            json.dump(data, file_)


def annotate_folder(path, job, label):
    files = get_files(path, ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tiff', '*.tif'])
    anotated = 0
    for file in tqdm(files):
        query = {'local_path': os.path.dirname(file),
                 'name': os.path.basename(file)}
        file = FileRecord.find_one(query)
        if file is None:
            continue
        job.annotate_file(file, label)
        anotated += 1
    logger.info(f"Annotated {anotated} files")
