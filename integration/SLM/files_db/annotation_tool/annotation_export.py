import os
import uuid
from collections import defaultdict
from typing import List

import loguru
from tqdm import tqdm

from SLM.appGlue.core import Service, Allocator
from SLM.appGlue.iotools.pathtools import copy_file_ifExist
from PIL import Image

from SLM.files_data_cache.pool import PILPool


# colab notebook for training
# https://colab.research.google.com/drive/15GeudBTXGnl6ok1efZ9ogF5JOu9PzTb0#scrollTo=e0WwGPyPpQsJ

def load_md5_list(md5_file_path):
    md5_list = []
    if os.path.exists(md5_file_path):
        with open(md5_file_path, "r") as file:
            md5_list = {line.strip() for line in file}
    return md5_list


def save_md5_list(md5_list, md5_file_path):
    # if os.path.exists(md5_file_path):
    if os.path.exists(md5_file_path):
        os.remove(md5_file_path)
    with open(md5_file_path, "w") as file:
        for md5 in md5_list:
            if md5 is None:
                continue
            file.write(md5 + "\n")


def get_last_batch_number(dataset_path):
    batch_folders = [f for f in os.listdir(dataset_path) if
                     os.path.isdir(os.path.join(dataset_path, f)) and f.isdigit()]
    batch_numbers = sorted(int(f) for f in batch_folders)
    return batch_numbers[-1] if batch_numbers else 0


def create_batch(bach_size, distribution, labeled_images):
    batch = []
    for cat, items in labeled_images.items():
        copy_count = int(bach_size * distribution[cat])
        cop_max = min(copy_count, len(items))
        copy_items = items[:cop_max]
        for item in copy_items:
            batch.append(item)
            labeled_images[cat].remove(item)
    return batch


class DataSetExporter:

    def __init__(self, name: str):
        self.name = name

    def is_job_type_supported(self, job_type: str):
        return False

    def ExportToDataset(self, dataset_path: str, annotation_job):
        pass



class DataSetExporterManager(Service):
    exporters: List[DataSetExporter] = []

    def register(self, exporter: DataSetExporter):
        self.exporters.append(exporter)

    def get_exporter_by_name(self, name: str):
        for exporter in self.exporters:
            if exporter.name == name:
                return exporter
        return None

    def get_all_exporters(self):
        return self.exporters

    def get_all_supported_job_types(self, job_type: str):
        types = []
        for exporter in self.exporters:
            if exporter.is_job_type_supported(job_type):
                types.append(exporter)
        return types

Allocator.res.register( DataSetExporterManager())


class DataSetExporterImageMultiClass_dirs(DataSetExporter):
    def __init__(self):
        super().__init__("ImageMultiClass_dirs")
        self.category_limit = 100000

    def is_job_type_supported(self, job_type: str):
        return job_type == "multiclass/images"

    def ExportToDataset(self, dataset_path: str, annotation_job: 'AnnotationJob'):
        from SLM.files_db.annotation_tool.annotation import AnnotationRecord
        result = annotation_job.get_all_annotated()
        category_count = defaultdict(int)
        for item in tqdm(result):
            item: AnnotationRecord
            dir = os.path.join(dataset_path, item.value)
            if not os.path.exists(dir):
                os.makedirs(dir)
            imageItem = item.file
            if imageItem is None or imageItem.full_path is None:
                continue
            if category_count[item.value] > self.category_limit:
                continue
            try:
                pil_image = PILPool.get_pil_image(imageItem.full_path).copy()
                try:
                    pil_image.verify()
                    category_count[item.value] += 1
                    image_name = str(uuid.uuid4()) + ".jpg"
                    new_path = os.path.join(dir, image_name)

                except Exception as e:
                    loguru.logger.error(e)
                    continue

                pil_image = PILPool.get_pil_image(imageItem.full_path).copy()
                pil_image.thumbnail((256, 256))
                pil_image.save(new_path)
            except Exception as e:
                print(e)
                continue






class DataSetExporterImageMultiClass_dirs_cum(DataSetExporter):
    def __init__(self):
        super().__init__("ImageMultiClass_dirs")
        self.bach_size = 100000
        self.clamp_cat = 10000

    def is_job_type_supported(self, job_type: str):
        return job_type == "multiclass/images"

    def ExportToDataset(self, dataset_path: str, annotation_job: 'AnnotationJob', ignore_cat=None):
        if ignore_cat is None:
            ignore_cat = []
        from SLM.files_db.annotation_tool.annotation import AnnotationRecord
        if not os.path.exists(dataset_path):
            return
        # Load parent MD5 list
        md5_list = load_md5_list(dataset_path + "\\" + "data.json")

        # Calculate the last batch folder number
        last_batch_number = get_last_batch_number(dataset_path)

        # Get all annotated records
        result = annotation_job.get_all_annotated()
        labeled_images = defaultdict(list)

        for item in tqdm(result):
            item: AnnotationRecord
            if item.file is None or item.file.full_path is None:
                item.delete_rec()
                continue

            # Skip if the image is exist on disk
            if not os.path.exists(item.file.full_path):
                item.delete_rec()
                continue

            if item.file.file_content_md5 in md5_list:
                continue
            if item.value in ignore_cat:
                continue
            if len(labeled_images[item.value]) > self.clamp_cat:
                continue
            labeled_images[item.value].append(item.file)
            md5_list.append(item.file.file_content_md5)
        all_images_count = sum(len(v) for v in labeled_images.values())
        distribution = {}
        for cat, items in labeled_images.items():
            distribution[cat] = all_images_count / len(items)
        # Create batch folders
        while True:
            batch = create_batch(self.bach_size, distribution, labeled_images)
            if len(batch) == 0:
                break

            batch_folder = os.path.join(dataset_path, str(last_batch_number + 1))
            os.makedirs(batch_folder, exist_ok=True)

            for image in tqdm(batch):
                image_name = os.path.basename(image.name)
                label = annotation_job.get_annotation_record(image).value
                new_path = os.path.join(batch_folder, label, image_name)
                folder_path = os.path.join(batch_folder, label)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    # check file trunck

                try:
                    pil_image = PILPool.get_pil_image(image.full_path).copy()
                    pil_image.verify()
                    pil_image = PILPool.get_pil_image(image.full_path).copy()
                    pil_image.thumbnail((256, 256))
                    pil_image.save(new_path)
                except Exception as e:
                    loguru.logger.error(e)

            last_batch_number += 1

        # Save parent MD5 list
        save_md5_list(md5_list, dataset_path + "\\" + "data.json")


class DataSetExporterImageMultiClass_anomali(DataSetExporter):
    def __init__(self):
        super().__init__("ImageMultiClass_dirs")

    def is_job_type_supported(self, job_type: str):
        return job_type == "multiclass/images"

    def ExportToDataset(self, dataset_path: str, annotation_job: 'AnnotationJob', annotator):

        from SLM.files_db.annotation_tool.annotation import AnnotationRecord
        if not os.path.exists(dataset_path):
            return
        # add 1 item to each class
        for label in annotation_job.choices:
            dir = os.path.join(dataset_path, label)
            if not os.path.exists(dir):
                os.makedirs(dir)
            item = annotation_job.get_file_records_by_label(label)[0]
            imageItem = item.file
            if imageItem is None:
                continue
            try:
                new_path = os.path.join(dir, imageItem.name)

                copy_file_ifExist(imageItem.full_path, new_path)
            except Exception as e:
                print(e)
                continue
        # Get all annotated records
        result = annotation_job.get_all_annotated()
        for item in tqdm(result):
            item: AnnotationRecord
            if item.file is None or item.file.full_path is None:
                continue

            # Skip if the image is exist on disk
            if not os.path.exists(item.file.full_path):
                continue

            if not annotator.is_satisfied_by(item.value, item.file):
                target_folder = os.path.join(dataset_path, item.value)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                image_name = os.path.basename(item.file.name)
                new_path = os.path.join(target_folder, image_name)
                try:
                    copy_file_ifExist(item.file.full_path, new_path)
                except Exception as e:
                    loguru.logger.error(e)

DataSetExporterManager().register(DataSetExporterImageMultiClass_dirs_cum())

