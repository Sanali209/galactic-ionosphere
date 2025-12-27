import json
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from SLM.files_data_cache.pool import PILPool

os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'
os.environ['APPDATA'] = r"D:\data\ImageDataManager"
os.environ['MONGODB_NAME'] = "files_db"
from SLM.iterable.bach_builder import BatchBuilder

from SLM.files_db.components.File_record_wraper import get_file_record_by_folder, FileRecord
from applications.grpcTascks.client import ATask, TaskClient
from PIL import Image


def pil_image_to_str(image: Image):
    import io
    import base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_str = base64.b64encode(img_byte_arr).decode()
    return img_str


class Get_image_label(ATask):
    def __init__(self):
        super().__init__("vision_comics_bw", [])
        self.content_md5 = []
        self.set_task_complete_callback(self.on_complete_t)
        self.pipline_name = "multiclass_sketch_bf"

    def on_complete_t(self, task):
        from SLM.files_data_cache.imageToLabel import ImageToTextCache
        print(f"Task {task.task_id} completed with result: {task.result}")
        result = json.loads(task.result)
        counter = 0
        for res in result:
            ImageToTextCache.instance().set_by_md5(self.content_md5[counter], self.pipline_name, ["1.0", res])
            counter += 1

    def add_image_from_path(self, path):
        from SLM.files_data_cache.imagedatacache import ImageDataCacheManager
        from SLM.files_data_cache.imageToLabel import ImageToTextCache
        md_5 = ImageDataCacheManager.instance().path_to_md5(path)
        if self.content_md5 is None:
            return False
        exist_res = ImageToTextCache.instance().get_by_md5(md_5, self.pipline_name)
        if exist_res is not None:
            return False
        image = PILPool.get_pil_image(path)
        image.thumbnail((256, 256))
        image_base64 = pil_image_to_str(image)
        self.content_md5.append(ImageDataCacheManager.instance().path_to_md5(path))
        self.args.append(image_base64)


def execute_task(args1):
    task = Get_image_label()
    for arg in args1:
        task.add_image_from_path(arg)
    if len(task.args) > 0:
        client.run_task(task)


# Использование
if __name__ == '__main__':
    client = TaskClient()
    executor = ThreadPoolExecutor(max_workers=1)

    # Запускаем поток для получения уведомлений
    path = r"E:\rawimagedb\repository\nsfv repo\drawn\drawn xxx autors"
    files = get_file_record_by_folder(path, recurse=True)
    bachBuilder = BatchBuilder(files, 8)
    args = []
    for bach in tqdm(bachBuilder.bach.values()):
        args.clear()
        args.extend([f.full_path for f in bach])
        execute_task(args)
        #executor.submit(execute_task, args)

    client.wait_for_complete_all_tasks()
