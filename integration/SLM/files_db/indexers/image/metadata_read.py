from SLM.files_db.indexers.image.content_md5 import files_db_indexer
from SLM.files_db.components.fs_tag import TagRecord

from SLM.chains.chains_main import DictFieldMergeChainFunction, DictFormatterChainFunction
from SLM.indexerpyiplain.idexpyiplain import ItemIndexer
from SLM.metadata.MDManager.mdmanager import MDManager

# todo :implement force _reindex metadata
# todo: implement import tags from file

collect_descr_coollect_mapping = {"SLMMeta:collected_description":
    [
        'Ducky:Comment', 'EXIF:ImageDescription', 'IPTC:By-lineTitle',
        "IPTC:Caption-Abstract", 'MakerNoteUnknownText', "IPTC:DocumentNotes", 'IPTC:Headline',
        "IPTC:Writer - Editor", 'XMP:About', "XMP:Caption", 'XMP:ArtworkCopyrightNotice',
        "XMP:ArtworkPhysicalDescription", 'EXIF:UserComment',
        "XMP:ArtworkSourceInvURL", "XMP:ArtworkTitle", "XMP:AuthorsPosition", "XMP:CopyrightOwnerName",
        "XMP:Country", "XMP:City", "XMP:Credit", "XMP:DerivedFromManageTo", "XMP:Label",
        'XMP:ManifestReferenceFilePath', "XMP:SupplementalCategories", "XMP:TextLayerName",
        "XMP:TextLayerText", "XMP:TextLayersLayerName", "XMP:TextLayersLayerText"
    ],
    "SLMMeta:collected_objects": ["IPTC:ObjectName", "XMP:PersonInImage", "XMP:Personality"],
    "SLMMeta:collected_authors": ['XMP:Creator', "XMP:Author", "XMP:Artist",
                                  'XMP:ArtworkCreator', "XMP:ArtworkOrObjectAOCreatorID"],
}

fields_mapping = {
    'SLMMeta:modified_date': ['XMP:ModifyDate', 'EXIF:ModifyDate'],
    'SLMMeta:rating': ['XMP:Rating', "EXIF:Rating"],
    'SLMMeta:tags': ['XMP:Subject', "XMP:Keywords", "IPTC:Keywords"],
    'SLMMeta:categories': ["IPTC:SupplementalCategories", "XMP:SupplementalCategories"],
    'SLMMeta:autor': ['XMP:Creator', "XMP:Author", "XMP:Artist", 'IPTC:By-lineTitle', "IPTC:Caption-Abstract"],
    "SLMMeta:title": ['XMP:Title', "XMP:Label", "XMP:Headline", 'EXIF:ImageDescription', 'IPTC:Headline'],
    'SLMMeta:description': ["XMP:Notes", 'XMP:Description', "XMP:Caption", 'EXIF:ImageDescription'],
    'SLMMeta:notes': ["XMP:UserComment", 'EXIF:UserComment', 'File:Comment', "IPTC:DocumentNotes", "PNG:Note"],
    'SLMMeta:MIMETYPE': ['File:MIMEType'],
    'SLM:meta_error': ['MDM:metadata_read_error']
}


class Image_MetadataRead(files_db_indexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fieldName = "MetadataRead"
        # todo use globall setings
        self.embed_all_metadata = False


    def add_metadata(self, item, key, value):
        if not self.embed_all_metadata:
            return
        metadata_collection = item.get('metadata', {})
        metadata_collection[key] = value
        item['metadata'] = metadata_collection

    def index(self, parent_indexer: ItemIndexer, item, need_index):
        from SLM.files_db.components.File_record_wraper import FileRecord
        file_item = FileRecord(item["_id"])
        metadata = MDManager(file_item.full_path)
        metadata.Read()

        def keyword_validator(x):
            if isinstance(x, list):
                # sometimes we have not string values in list
                return [str(i) for i in x]
            if isinstance(x, str):
                return x.split(',')

            if isinstance(x, int):
                return [str(x)]

            return x

        fields_validators = {
            'SLMMeta:modified_date': lambda x: x,
            'SLMMeta:rating': lambda x: x,
            'SLMMeta:tags': keyword_validator,
            'SLMMeta:autor': lambda x: x,
            'SLMMeta:description': lambda x: x,
        }

        collect_descr = DictFieldMergeChainFunction()
        collect_descr.map = collect_descr_coollect_mapping

        formate_meta = DictFormatterChainFunction()
        formate_meta.copy_source = True
        formate_meta.mapping = fields_mapping
        formate_meta.validators = fields_validators

        collect_descr | formate_meta

        result = collect_descr.run(**metadata.metadata)

        tags_field = result.get('SLMMeta:tags', [])
        if 'SLMMeta:tags' in result:
            del result['SLMMeta:tags']
        taglist = item.get("tags", [])
        # process tags
        for tag in tags_field:
            new_tag_parts = []
            tag_parts = tag.split('|')
            for tag_part in tag_parts:
                tag_part = tag_part.strip()
                new_tag_parts.append(tag_part)
            tag = '/'.join(new_tag_parts)
            tag = tag.replace("//", "/")
            tag_record = TagRecord.get_or_create(tag)

            taglist.append(tag_record.fullName)
        taglist = list(set(taglist))
        item["tags"] = taglist

        for key, value in result.items():
            if key.startswith('SLMMeta:'):
                nkey = key.replace('SLMMeta:', '')

                item[nkey] = value
            else:
                self.add_metadata(item, key, value)

        parent_indexer.shared_data["item_indexed"] = True

        self.mark_as_indexed(item,parent_indexer)
