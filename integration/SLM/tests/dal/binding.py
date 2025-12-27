import os
import unittest

from SLM.appGlue.DAL.binding.bind import PropInfo, PropUser

os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'
os.environ['APPDATA'] = r"D:\Sanali209\Python\SLM\tests\dal"


class persbindings(PropUser):
    test_prop: str = PropInfo(True)


class PersistSettings:
    def __init__(self):
        self.prop = persbindings()


class TestSLMFSDB(unittest.TestCase):
    def test_persist_binds(self):
        pers_obj = PersistSettings()
        print(pers_obj.prop.test_prop)
        pers_obj.prop.test_prop = "test"
        print(pers_obj.prop.test_prop)
