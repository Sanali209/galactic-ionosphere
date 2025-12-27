import json
import os
from collections import OrderedDict

from pymongo import UpdateOne
from tqdm import tqdm

from SLM.files_db.components.File_record_wraper import FileRecord

from SLM.mongoext.wraper import MongoRecordWrapper, FieldPropInfo


class TagRecord(MongoRecordWrapper):
    name: str = FieldPropInfo('name', str, '')
    fullName: str = FieldPropInfo('fullName', str, '')
    autotag: bool = FieldPropInfo('autotag', bool, False)

    remap_to_tags: str = FieldPropInfo('remap_to', str, None)
    ''' remap_to_tags is a string with tags separated by ";".
     each tag is a full name of tag, on execute remap_tag()
     it will be split by ";" and each tag will be added to file marked with this tag'''

    def remap_tag(self):
        if self.remap_to_tags is None:
            return
        files = FileRecord.find({'tags': self.fullName})
        tags_full_names = self.remap_to_tags.split(';')
        for tag_full_name in tags_full_names:
            add_tag = TagRecord.get_or_create(fullName=tag_full_name)
            for file in files:
                add_tag.add_to_file_rec(file)

    @classmethod
    def get_tags_of_file(cls, file: 'FileRecord'):
        tags_names = file.list_get('tags')
        tags = []
        for tag_ref in tags_names:
            tags.append(TagRecord.get_record_by_name(tag_ref))
        return tags

    @classmethod
    def get_record_by_name(cls, name: str):
        return cls.find_one({'fullName': name})

    @classmethod
    def get_all_tags(cls, root_tags=False):
        if root_tags:
            return cls.find({'parent_tag': None}, [('fullName', 1)])
        return cls.find({}, [('fullName', 1)])

    @classmethod
    def create_record_data(cls, **kwargs):
        full_name = kwargs.get("fullName", None)
        if full_name is None:
            pass
        data = super().create_record_data(fullName=full_name)
        name = os.path.basename(full_name)
        tag_dir = os.path.dirname(full_name)
        data["name"] = name
        if tag_dir != '':
            parent_tag = cls._get_or_create(fullName=tag_dir)
            data["parent_tag"] = parent_tag._id
        else:
            data["parent_tag"] = None

        return data

    def is_file_tagged(self, file: 'FileRecord') -> bool:
        """
        Check if file is tagged with this tag
        :param file:
        :return:
        """
        return self.fullName in file.list_get('tags')

    @classmethod
    def get_or_create(cls, full_name: str = "", **kwargs) -> 'TagRecord':
        """
        overridden method to search for tag by fullName or create new tag if not found
        :param full_name:
        :param kwargs:
        :return:
        """
        if full_name != "":
            kwargs["fullName"] = full_name
        return cls._get_or_create(**kwargs)

    @property
    def parent_tag(self) -> 'TagRecord':
        """
        contains parent tag _id
        :return:
        """
        data = self.get_field_val('parent_tag')
        return TagRecord(data)

    @parent_tag.setter
    def parent_tag(self, value: 'TagRecord'):
        self.set_field_val('parent_tag', value._id)

    def child_tags(self):
        return list(TagRecord.find({'parent_tag': self._id}))

    def tagged_files(self):
        return list(FileRecord.find({'tags': self.fullName}))

    def add_to_file_rec(self, file: FileRecord):
        file.metadata_dirty = True
        file.list_append('tags', self.fullName, no_dupes=True)

    def remove_from_file_rec(self, file: FileRecord):
        file.metadata_dirty = True
        file.list_remove('tags', self.fullName)

    def delete(self):
        child_tags = TagRecord.find({'parent_tag': self._id})
        for tag in child_tags:
            tag.delete()
        tag_name = self.fullName  # for optimization (avoiding multiple db queries on get_record_data())
        # remove tag from all files
        filter_q = {'tags': tag_name}
        update_q = {'$pull': {'tags': tag_name}}
        FileRecord.collection().update_many(filter_q, update_q)
        self.delete_rec()

    def rename(self, new_name):
        """
        Rename tag  and update all files with this tag
        :param new_name:
        :return:
        """
        new_parent_path = os.path.dirname(new_name)
        parent_tag = TagRecord.get_or_create(fullName=new_parent_path)
        self.parent_tag = parent_tag
        old_name = self.fullName
        self.fullName = new_name
        self.name = os.path.basename(new_name)
        filter_query = {'tags': old_name}
        records = FileRecord.collection().find(filter_query)
        bulk_op = []
        for file_record in records:
            tag_list = file_record['tags']
            tag_list.remove(old_name)
            tag_list.append(new_name)
            file_id = file_record['_id']
            bulk_op.append(UpdateOne({'_id': file_id}, {'$set': {'tags': tag_list}}))
        if len(bulk_op) > 0:
            FileRecord.collection().bulk_write(bulk_op)

        child_tags = TagRecord.find({'parent_tag': self._id})
        for tag in child_tags:
            child_tag_new_full_name = tag.fullName.replace(old_name, new_name)
            tag.rename(child_tag_new_full_name)

    # override equal method

    @classmethod
    def get_tags_report(cls):
        tags = cls.get_all_tags()
        report = {}
        for tag in tqdm(tags):
            report[tag.fullName] = {"files": len(tag.tagged_files()),
                                    "child_tags": len(tag.child_tags()),
                                    "autotag": tag.autotag}
        #sort dictionary by key
        report = OrderedDict(sorted(report.items()))
        # save report to json file
        json_save_path = os.path.join(os.path.dirname(__file__), "tags_report.json")
        with open(json_save_path, 'w') as f:
            json.dump(report, f)
        # open file with report
        os.startfile(json_save_path)
