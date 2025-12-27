import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.appGlue.iotools.pathtools import get_files
from SLM.files_data_cache.imageToLabel import ImageToTextCache

from SLM.files_db.annotation_tool.annotation_export import DataSetExporterImageMultiClass_dirs

from SLM.files_db.components.relations.relation import RelationRecord
from SLM.files_db.files_db_module import IndexerConfig
from SLM.files_db.object_recognition.object_recognition import Detection
from SLM.groupcontext import group
from SLM.iterable.bach_builder import BatchBuilder
from SLM.metadata.MDManager.mdmanager import MDManager
import unittest
from SLM.files_db.components.File_record_wraper import FileRecord, get_file_record_by_folder, refind_exist_files, \
    remove_files_record_by_mach_pattern

from SLM.appGlue.DAL.datalist2 import MongoDataModel, DataViewCursor
from SLM.files_db.annotation_tool.annotation import AnnotationJob, annotate_folder, SLMAnnotationClient
from SLM.files_db.files_functions.index_folder import index_folder, index_folder_one_thread

from SLM.files_db.components.fs_tag import TagRecord

# https://colab.research.google.com/drive/15GeudBTXGnl6ok1efZ9ogF5JOu9PzTb0#scrollTo=JXaSEk35oqFH
# fine tune single label transformer
config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"
Allocator.init_modules()


class TestSLMFSDB(unittest.TestCase):

    def test_add_folder(self):
        # added to maintain panel
        FileRecord.add_file_records_from_folder(r"E:\rawimagedb\repository")

    def test_index_one_folder(self):
        # added to maintain panel
        path = r'E:\rawimagedb\repository'
        index_folder_one_thread(path)

    def test_remove_faces_files(self):
        res = Detection.find({})
        for rec in tqdm(res):
            rec.delete_rec()
        #records = get_file_record_by_folder(TestSLMFSDB.path, recurse=True)
        records = FileRecord.find({})
        for record in tqdm(records):
            record.list_remove("indexed_by", "face_detection")
        relations = RelationRecord.find({'type': "similar_face_search"})
        for relation in tqdm(relations):
            relation.delete_rec()

    def test_reset_index(self):
        files_list = FileRecord.find({})
        for file in tqdm(files_list):
            file.list_remove("indexed_by", "face_detection")

    def test_remove_1_0_tags(self):
        tags = TagRecord.find({})
        for tag in tqdm(tags):
            tag: TagRecord
            count = len(tag.child_tags())
            if count > 0:
                continue
            count = len(tag.tagged_files())
            if count > 10:
                continue

            print(f"remove tag {tag.fullName}")
            tag.delete()

    def test_clear_metadata(self):
        def clear_metadata(images_path):
            files = get_files(images_path, ['*.jpg', '*.png', '*.jpeg'])
            metamanager = MDManager(None)
            for file in tqdm(files):
                exist_dir_list = ['APP0:all', 'APP14:all', 'FlashPix:all', 'ICC_Profile:all', 'EXIF:all',
                                  "IPTC:all", 'XMP:all']

                metamanager.backend.del_tag(file, "all")

        clear_metadata(r'E:\rawimagedb\repository\safe repo')

    def test_rem_dupli_tags(self):
        tags = TagRecord.find({}, {"fullName": 1})
        delete_list = []
        for ind in tqdm(range(len(tags))):
            tag = tags[ind]
            tag: TagRecord
            if ind + 1 < len(tags):
                tag2 = tags[ind + 1]
                tag2: TagRecord
                if tag.fullName == tag2.fullName:
                    delete_list.append(tag2)

        for tag in tqdm(delete_list):
            tag.delete_rec()

    def test_fix_tags_parenting(self):
        # reindex all tags parenting by using thei full name thei containg parenting data as / separator:
        # sample: parent/child/grandchild/other level
        tags = TagRecord.find({})
        for tag in tqdm(tags):
            tag: TagRecord
            full_name = tag.fullName or ""
            if "/" in full_name:
                parts = [p for p in full_name.split("/") if p]
                parent_name = "/".join(parts[:-1])
                parent_tag = TagRecord.get_or_create(parent_name) if parent_name else None
                tag.set_field_val("parent_tag", parent_tag._id if parent_tag else None)
            else:
                tag.set_field_val("parent_tag", None)
            # remove legacy field
            tag.set_field_val("parentTag", None)


    def test_remove_1_0_tags_from_nam(self):
        tags = TagRecord.find_one({"name": "from_name"}).child_tags()

        def check_tag(tag):
            count = len(tag.child_tags())
            if count > 0:
                return
            count = len(tag.tagged_files())
            if count > 2:
                return
            tag.delete_rec()

        bach_b = BatchBuilder(tags, 16)
        for butch in tqdm(bach_b.bach.values()):
            with ThreadPoolExecutor(max_workers=16) as executor:
                executor.map(check_tag, butch)

    def test_remove_files_by_ext(self):
        path = r"D:"
        query = {'local_path': {"$regex": '^' + re.escape(path)}}
        records = FileRecord.find(query)
        for record in tqdm(records):
            record: FileRecord
            record.delete_rec()

    def test_remove_files_by_ext2(self):
        ext = ".ini"
        query = {'name': {"$regex": '.*' + re.escape(ext) + '$'}}
        records = FileRecord.find(query)
        for record in tqdm(records):
            record: FileRecord
            record.delete_rec()

    def test_redetect_undetected(self):
        # reset flag face_detection in files without detection
        path = r"E:\rawimagedb\repository\nsfv repo\drawn\presort"
        query = {'local_path': {"$regex": '^' + re.escape(path)}}
        files = FileRecord.find(query)
        for file in tqdm(files):
            file: FileRecord
            detection_query = {"parent_image_id": file._id}
            detections = Detection.find(detection_query)
            if len(detections) == 0:
                file.list_remove("indexed_by", "face_detection")

    def test_delete_faces(self):
        files_list = FileRecord.find({})
        for file in tqdm(files_list):
            file.list_remove("indexed_by", "face_detection")

    def test_index_folder(self):
        path = r"E:\rawimagedb\repository\nsfv repo\drawn\presort"
        config: IndexerConfig = Allocator.config.Indexer
        config.detectFaces = True
        query = {'local_path': {"$regex": '^' + re.escape(path)}}
        # todo known bug freez if antivirus in active mode totalsecure 360
        index_folder(query, 6)

    def test_remove_anotation_dubs(self):
        jobs = AnnotationJob.find({"name": "rating_competition"})
        for job in tqdm(jobs):
            job: AnnotationJob
            job.remove_annotation_dublicates2()

    def test_reove_brocken_annotatiob(self):
        jobs = AnnotationJob.find({})
        for job in tqdm(jobs):
            job: AnnotationJob
            job.remove_broken_annotations()

    def test_remove_folder_from_anotating(self):
        path = r"D:\image db\nsfw onli db"
        job_name = "rating"
        job = AnnotationJob.get_by_name(job_name)
        items = get_file_record_by_folder(path, recurse=True)
        for item in tqdm(items):
            item: FileRecord
            if (item.full_path is None) or not os.path.exists(item.full_path):
                continue
            job.remove_annotation_record(item)

    def test_remove_choose_from_annotation(self):
        job_name = "NSFWFilter"
        job: AnnotationJob = AnnotationJob.get_by_name(job_name)
        choices: list = job.choices
        choices.remove("ero")
        choices.remove("porn")
        choices.remove("explicit")
        job.choices = choices

    def test_add_choose_to_annotation(self):
        job_name = "NSFWFilter"
        job: AnnotationJob = AnnotationJob.get_by_name(job_name)
        new_choises = ["sensitive", "explicit", 'questionable']
        job.add_annotation_choices(new_choises)

    def test_rename_choose_in_annotation(self):
        job_name = "NSFWFilter"
        job: AnnotationJob = AnnotationJob.get_by_name(job_name)
        rename = ('safe', 'general')
        job.rename_annotation_label(rename[0], rename[1])

    def test_clear_annotation_job(self):
        job_name = "NSFWFilter"
        job: AnnotationJob = AnnotationJob.get_by_name(job_name)
        job.clear_job()

    def test_tagging(self):
        path = r"D:\image db\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"
        f_record = FileRecord.get_record_by_path(path)
        print(f_record)
        tag = TagRecord.get_or_create(full_name="cat/test_tag/tag")
        print(tag)
        tag.add_to_file_rec(f_record)

        tags = TagRecord.get_tags_of_file(f_record)

        print(tags)

    def test_read_rating(self):
        path = r"E:\rawimagedb\repository\nsfv repo\furi\Yiff Artworks"
        filelist = get_files(path, ['*.jpg', '*.png'])
        job = AnnotationJob.get_by_name("rating")
        for file in tqdm(filelist):
            file: FileRecord = FileRecord.get_record_by_path(file)

            metadata_manager = MDManager(file.full_path)
            metadata_manager.Read()
            rating = metadata_manager.metadata.get('XMP:Rating', None)
            if rating == 4 or rating == 5:
                job.annotate_file(file, job.choices[2])
                print(job.choices[2])
            elif rating == 3:
                job.annotate_file(file, job.choices[1])
                print(job.choices[1])
            elif rating == 2 or rating == 1:
                job.annotate_file(file, job.choices[0])
                print(job.choices[0])

    def test_refind_exist_files(self):
        path = r"E:\rawimagedb\repository"
        refind_exist_files(path)

    def test_annotate_folder(self):
        path = r"F:\rawimagedb\repository\safe repo\presort\text page"
        job = AnnotationJob.get_by_name("image genres")
        label = job.choices[5]
        print(label)
        annotate_folder(path, job, label)

    def test_change_drive_letter(self):
        path = r"F:\rawimagedb"
        files = get_file_record_by_folder(path, recurse=True)
        for file in tqdm(files):
            file: FileRecord
            file.local_path = file.local_path.replace("F:", "E:")

    def test_move_folder(self):
        source_folder = r"E:\rawimagedb\repository\nsfv repo\drawn\_site rip"
        dest_folder = r"E:\rawimagedb\repository\nsfv repo\drawn\comix\bv"
        items = get_file_record_by_folder(source_folder, recurse=True)
        for item in tqdm(items):
            item: FileRecord
            if (item.full_path is None) or not os.path.exists(item.full_path):
                continue
            if item.full_path.startswith(source_folder):
                file_path_dif = item.local_path.replace(source_folder, "")
                dest_m_folder = dest_folder + file_path_dif
                if not os.path.exists(dest_m_folder):
                    os.makedirs(dest_m_folder)
                try:
                    item.move_to_folder(dest_m_folder)
                except:
                    continue

    def test_move_by_annotation_label(self):
        source_folder = r"E:\rawimagedb\repository\nsfv repo"
        dest_folder = r"X:\rawdb\sketch\bw"
        job = AnnotationJob.get_by_name("sketch binary")
        label = job.choices[1]
        items = job.get_ann_records_by_label(label)
        for item in tqdm(items):
            item: FileRecord = item.file
            if (item.full_path is None) or not os.path.exists(item.full_path):
                continue
            if item.full_path.startswith(source_folder):
                file_path_dif = item.local_path.replace(source_folder, "")
                dest_m_folder = dest_folder + file_path_dif
                if not os.path.exists(dest_m_folder):
                    os.makedirs(dest_m_folder)
                try:
                    item.move_to_folder(dest_m_folder)
                except:
                    continue

    def test_export_base(self):
        exporter = DataSetExportfrImageMultiClass_dirs()
        ajob = AnnotationJob.get_by_name("rating")
        exporter.ExportToDataset(r"D:\image_export\rating", ajob)

    def test_prediction(self):
        path = r"E:\rawimagedb\repository\nsfv repo\drawn\drawn xxx autors"
        files = get_file_record_by_folder(path, recurse=True)
        path = files[0].full_path
        annotat_label = ImageToTextCache.instance().get_by_path(path, "multiclass_NSFW_HF")
        print(annotat_label)

    def test_remove_exist_file_records(self):
        # added to maintain panel
        path = r"E:\rawimagedb\repository"
        #files = get_file_record_by_folder(path, recurse=True)
        files = FileRecord.find({})
        for file in tqdm(files):
            if file.full_path is None or not os.path.exists(file.full_path):
                print(file.full_path)
                file.delete()

    def test_tags_report(self):
        TagRecord.get_tags_report()

    def test_import_boru_data(self):
        image_path = r"E:\rawimagedb\repository\nsfv repo\rule34"
        json_path = r"E:\rawimagedb\repository\nsfv repo\rule34\results.jsonl"
        tags_prefix = "imported/boru/"
        rows = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))

        data_list = rows

        data_dict = {}
        for data_item in tqdm(data_list):
            data_dict[data_item["topic_id"]] = data_item

        files = get_file_record_by_folder(image_path, recurse=True)
        for file in tqdm(files):
            file: FileRecord
            imported = file.get_field_val("boru_imported")
            if imported:
                continue
            all_tags = []
            if file.full_path is None or not os.path.exists(file.full_path):
                continue
            mach = re.search(r"_([^_]+)\.\w+$", file.full_path)
            if mach is None:
                continue
            post_is = mach.group(1)
            topic_data = data_dict.get(post_is)
            if topic_data is not None:

                fields_data = topic_data.get("fields")
                source_url = fields_data.get("stat_source_url")
                file.source = source_url

                tags_general = fields_data.get("tags_general")
                if tags_general is not None:
                    tags_general = ["general/" + tag for tag in tags_general]
                    all_tags.extend(tags_general)
                tags_character = fields_data.get("tags_character")
                if tags_character is not None:
                    tags_character = ["character/" + tag for tag in tags_character]
                    all_tags.extend(tags_character)
                tags_artist = fields_data.get("tags_artist")
                if tags_artist is not None:
                    tags_artist = ["artist/" + tag for tag in tags_artist]
                    all_tags.extend(tags_artist)
                tags_copyright = fields_data.get("tags_copyright")
                if tags_copyright is not None:
                    tags_copyright = ["copyright/" + tag for tag in tags_copyright]
                    all_tags.extend(tags_copyright)

                tags_metadata = fields_data.get("tags_metadata")
                if tags_metadata is not None:
                    tags_metadata = ["metadata/" + tag for tag in tags_metadata]
                    all_tags.extend(tags_metadata)
                cont_rating = fields_data.get("stat_rating_raw")
                if cont_rating is not None:
                    all_tags.append("metadata/rating/" + str(cont_rating))
                for tag in all_tags:
                    tag = tags_prefix + tag
                    tag_record = TagRecord.get_or_create(tag)
                    tag_record.add_to_file_rec(file)
                file.set_field_val("boru_imported", True)

    def test_backup_ann(self):
        path = r"G:\Мой диск\imdb\js.json"
        client = SLMAnnotationClient()
        client.save_to_json(path)

    def test_delete_files_on_folder(self):
        path = r"E:\rawimagedb\repository\nsfv repo\3d\authors 3d\Pat\Sleepless Nights 03"
        files = get_file_record_by_folder(path, recurse=True)
        for file in tqdm(files):
            file.delete()

    def test_refresh_thumb(self):
        path = r"D:\imgdb"
        file = FileRecord.find({"local_path": {"$regex": '^' + re.escape(path)}})
        for file in tqdm(file):
            file: FileRecord
            file.refresh_thumb()

    def test_remove_fileRecords_duplicates(self):
        query = {}
        records = FileRecord.find(query)
        file_paths = {}
        for record in tqdm(records):
            record: FileRecord
            if record.full_path is None or not os.path.exists(record.full_path):
                record.delete_rec()
                print(f"Deleted record with missing file: {record.full_path}")
                continue
            if record.full_path in file_paths:
                file_paths[record.full_path].append(record)
            else:
                file_paths[record.full_path] = [record]

        for path, recs in tqdm(file_paths.items()):
            if len(recs) > 1:
                for rec in recs[1:]:
                    print(f"Deleting duplicate record: {rec.full_path}")
                    rec.delete_rec()

    def test_import_and_write_metadata(self):
        import diskcache
        from loguru import logger

        image_path = r"E:\rawimagedb\repository\nsfv repo\rule34"
        json_path = r"E:\rawimagedb\repository\nsfv repo\rule34\results.jsonl"
        cache_path = os.path.join(os.path.dirname(json_path), "processed_cache")
        processed_files_cache = diskcache.Index(cache_path)

        logger.add("metadata_import.log", rotation="500 MB")

        rows = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))

        data_dict = {data_item["topic_id"]: data_item for data_item in rows}
        files = get_file_record_by_folder(image_path, recurse=True)

        for file in tqdm(files):
            file: FileRecord
            if file.full_path is None or not os.path.exists(file.full_path):
                logger.warning(f"File not found: {file.full_path}")
                continue

            if file.full_path in processed_files_cache:
                logger.info(f"Skipping already processed file: {file.full_path}")
                continue

            mach = re.search(r"_([^_]+)\.\w+$", file.full_path)
            if mach is None:
                continue

            post_id = mach.group(1)
            topic_data = data_dict.get(post_id)

            if topic_data:
                all_tags = []
                fields_data = topic_data.get("fields", {})

                tag_types = {
                    "tags_general": "general/",
                    "tags_character": "character/",
                    "tags_artist": "artist/",
                    "tags_copyright": "copyright/",
                    "tags_metadata": "metadata/"
                }

                for tag_type, prefix in tag_types.items():
                    tags = fields_data.get(tag_type)
                    if tags:
                        all_tags.extend([prefix + tag for tag in tags])

                cont_rating = fields_data.get("stat_rating_raw")
                if cont_rating is not None:
                    all_tags.append("metadata/rating/" + str(cont_rating))

                if all_tags:
                    try:
                        md_manager = MDManager(file.full_path)
                        md_manager.metadata['XMP:Subject'] = all_tags
                        md_manager.Save()
                        processed_files_cache[file.full_path] = True
                        logger.success(f"Successfully wrote metadata for: {file.full_path}")
                    except Exception as e:
                        logger.error(f"Failed to write metadata for {file.full_path}: {e}")
