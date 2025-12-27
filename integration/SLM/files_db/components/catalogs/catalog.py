import os

from pymongo import UpdateOne

from SLM.appGlue.DesignPaterns import allocator
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.mongoext.MongoClientEXT_f import DBInitializer, MongoClientExt
from SLM.mongoext.wraper import MongoRecordWrapper, FieldPropInfo


class CatalogRecord(MongoRecordWrapper):
    name: str = FieldPropInfo('name', str, '')
    fullName: str = FieldPropInfo('fullName', str, '')

    @classmethod
    def get_catalogs_of_file(cls, file: 'FileRecord'):
        tags_names = file.list_get('catalogs')
        tags = []
        for tag_ref in tags_names:
            tags.append(CatalogRecord.get_record_by_name(tag_ref))
        return tags

    @classmethod
    def get_record_by_name(cls, name: str):
        return cls.find_one({'fullName': name})

    @classmethod
    def get_all_catalogs(cls, root_tags=False):
        if root_tags:
            return cls.find({'parentCatalog': None}, [('fullName', 1)])
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
            data["parentCatalog"] = parent_tag._id
        else:
            data["parentCatalog"] = None

        return data

    @classmethod
    def get_or_create(cls, full_name: str = "", **kwargs):
        if full_name != "":
            kwargs["fullName"] = full_name
        return cls._get_or_create(**kwargs)

    @property
    def parentCatalog(self) -> 'CatalogRecord':
        """
        contains parent tag _id
        :return:
        """
        data = self.get_field_val('parent_tag')
        return CatalogRecord(data)

    @parentCatalog.setter
    def parentCatalog(self, value: 'CatalogRecord'):
        self.set_field_val('parentCatalog', value._id)

    def add_to_file_rec(self, file):
        file.metadata_dirty = True
        file.list_append('catalogs', self.fullName, no_dupes=True)

    def remove_from_file_rec(self, file: FileRecord):
        file.metadata_dirty = True
        file.list_remove('catalogs', self.fullName)

    def delete(self):
        child_tags = CatalogRecord.find({'parentCatalog': self._id})
        for tag in child_tags:
            tag.delete()
        tag_name = self.fullName  # for optimization (avoiding multiple db queries on get_record_data())
        # remove tag from all files
        filter_q = {'catalogs': tag_name}
        update_q = {'$pull': {'catalogs': tag_name}}
        FileRecord.collection().update_many(filter_q, update_q)
        self.delete_rec()

    def rename(self, new_name):
        """
        Rename tag  and update all files with this tag
        :param new_name:
        :return:
        """
        new_parent_path = os.path.dirname(new_name)
        parent_tag = CatalogRecord.get_or_create(fullName=new_parent_path)
        self.parentCatalog = parent_tag
        old_name = self.fullName
        self.fullName = new_name
        self.name = os.path.basename(new_name)
        filter_query = {'catalogs': old_name}
        records = FileRecord.collection().find(filter_query)
        bulk_op = []
        for file_record in records:
            tag_list = file_record['catalogs']
            tag_list.remove(old_name)
            tag_list.append(new_name)
            file_id = file_record['_id']
            bulk_op.append(UpdateOne({'_id': file_id}, {'$set': {'catalogs': tag_list}}))
        if len(bulk_op) > 0:
            FileRecord.collection().bulk_write(bulk_op)

        child_tags = CatalogRecord.find({'parentCatalog': self._id})
        for tag in child_tags:
            child_tag_new_full_name = tag.fullName.replace(old_name, new_name)
            tag.rename(child_tag_new_full_name)

    # override equal method


def init(config):
    service = allocator.Allocator.get_instance(MongoClientExt)
    service.register_collection("catalogs_records", CatalogRecord)
    CatalogRecord.create_index("fullName", {'fullName': 1})


allocator.Allocator.add_initializer(init)
