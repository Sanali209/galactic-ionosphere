import os
import unittest
os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'
os.environ['APPDATA'] = r"D:\data\ImageDataManager"
os.environ['MONGODB_NAME'] = "test_files_db"
from SLM.files_db.vector_db_ext.vector_db_ext import SearchScopeMongoDb


from SLM.vector_db.vector_db import VectorDB, ResultGroup
from SLM.files_db.components.File_record_wraper import FileRecord


class TestVectorDB(unittest.TestCase):
    def testSearch(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        record = FileRecord.get_record_by_path(r"Y:\rawimagedb\repository\safe repo\asorted images\3\37025611193_d421e70426_b.jpg")
        table = VectorDB.get_pref("FileRecordMobileNetV3Small")
        search_ind = SearchScopeMongoDb(
            {"local_path": r"Y:\rawimagedb\repository\safe repo\asorted images\3"}, table)
        result:ResultGroup = search_ind.search(record.get_record_data(), 9)
        counter = 0
        for res in result.results:
            record = FileRecord(res.data_item["_id"])
            print(record.full_path)
            #draw image by matplotlib with sub
            path = record.full_path
            img = mpimg.imread(path)
            sub = plt.subplot(3, 3, counter + 1)
            sub.imshow(img)
            counter += 1
        plt.show()
