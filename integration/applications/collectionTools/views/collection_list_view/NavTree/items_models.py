import fnmatch
import os
import string

from tqdm import tqdm

from SLM.appGlue.DesignPaterns import allocator
from fs.osfs import OSFS
from fs.mountfs import MountFS

from SLM.files_db.components.collectionItem import CollectionRecord


class FsRoot:
    def __init__(self):
        allocator.Allocator.register(FsRoot, self)
        self.name = "fs"
        drives = [f"{letter}:" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:")]
        self.root_fs = MountFS()  # create root fs
        for drive in drives:
            try:
                fs = OSFS(drive)
                self.root_fs.mount(drive, fs)
            except Exception as e:
                print(e)

    def get_subFolders_info(self):
        folders = self.root_fs.listdir("/")
        return [FsFolderInfo(folder + "/") for folder in folders]


class FsFolderInfo:
    def __init__(self, path):
        self.path = path
        fs = allocator.Allocator.get_instance(FsRoot).root_fs
        self.name = fs.getdetails(path).name
        if self.name == "":
            self.name = path

    def get_subFolders_info(self):
        fs: OSFS = allocator.Allocator.get_instance(FsRoot).root_fs
        try:
            folders = fs.listdir(self.path)
        except:
            # todo prevent expand unacesible fold3ers
            return []
        ret_list = []
        for folder in tqdm(folders):
            full_path = os.path.join(self.path, folder)
            # check if folder
            if fs.isdir(full_path):
                #info = fs.getdetails(full_path)
                ret_list.append(FsFolderInfo(full_path + "/"))
        return ret_list

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name



class RecordTypeRoot:
    def __init__(self):
        self.name = "Record Types"
        self.record_types = [RecordTypeModel(x) for x in CollectionRecord.itemTypeMap.keys()]


class RecordTypeModel:
    def __init__(self, name):
        self.name = name
