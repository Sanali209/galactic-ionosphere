import os
import re
import shutil
import subprocess

import psutil
from diskcache import Cache
from tqdm import tqdm


class PathManagerSys:
    dir_paths = []
    dir_black_list = []
    file_black_list = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tif', '*.tiff']

    def addDirPath(self, path):
        self.dir_paths.append(path)

    def addDirBlackList(self, path):
        self.dir_black_list.append(path)

    def addFileBlackList(self, path):
        self.file_black_list.append(path)

    def getFiles(self):
        files = []
        for dir_path in self.dir_paths:
            files.extend(get_files(dir_path, self.extensions))
        for dir_black in self.dir_black_list:
            files = [file for file in files if dir_black not in file]
        for file_black in self.file_black_list:
            files = [file for file in files if file_black != file]
        return files

    def IsFileInBlackList(self, filepath):
        for dir_black in self.dir_black_list:
            if dir_black in filepath:
                return True
        for file_black in self.file_black_list:
            if file_black == filepath:
                return True
        return False

    def ImportSettingsFromData(self, data):
        self.dir_paths = data.get('dir_paths', [])
        self.dir_black_list = data.get('dir_black_list', [])
        self.file_black_list = data.get('file_black_list', [])
        self.extensions = data.get('extensions', self.extensions)

    def ExportSettingsToData(self):
        data = {'dir_paths': self.dir_paths, 'dir_black_list': self.dir_black_list,
                'file_black_list': self.file_black_list, 'extensions': self.extensions}
        return data


class PathListTask:
    def __init__(self, path, key, on_get_list: callable = None):
        self.list_cache = Cache(path)
        self.key = key
        self.on_get_list = on_get_list

    def get_list(self):
        cached_list = self.list_cache.get(self.key, default=None)
        if cached_list is None:
            if self.on_get_list is None:
                raise Exception("on_get_list is None")
            cached_list: list = self.on_get_list()
            self.list_cache.set(self.key, cached_list)
        return cached_list.copy()

    def remove_path(self, path_str):
        cached_list = self.list_cache.get(self.key, default=None)
        if cached_list is None:
            return
        if path_str in cached_list:
            cached_list.remove(path_str)
        self.list_cache.set(self.key, cached_list)

    def clear(self):
        self.list_cache.set(self.key, None)


# recursive file search with given  list
def get_files(path, exts=None, file_ignore_masck="", sub_dirs=True):
    """
    :param path: ProjectSettingsPath to search
    :param exts: list of extensions to search ["*.jpg", "*.png"] need include "*" in mask
       :param file_ignore_masck:  of files to ignore "Thumbs.db"
       @param sub_dirs:
    """
    if exts is None:
        exts = ["*"]
    import os
    import fnmatch
    matches = []
    if sub_dirs:
        for root, dirnames, filenames in tqdm(os.walk(path)):
            for file in filenames:
                if fnmatch.fnmatch(file, file_ignore_masck):
                    continue
                for ext in exts:
                    # no case match
                    if fnmatch.fnmatch(file, ext):
                        matches.append(os.path.join(root, file))
    else:
        for file in tqdm(os.listdir(path)):
            # if directory skip
            if os.path.isdir(os.path.join(path, file)):
                continue
            if fnmatch.fnmatch(file, file_ignore_masck):
                continue
            for ext in exts:
                # no case match
                if fnmatch.fnmatch(file, ext):
                    matches.append(os.path.join(path, file))
    return matches


def get_sub_dirs(path):
    sub_pats = []
    for dir in os.listdir(path):
        sub_path = os.path.join(path, dir)
        if os.path.isdir(sub_path):
            try:
                os.listdir(sub_path)
                sub_pats.append(sub_path)
            except:
                pass
    return sub_pats


def get_drive_letters():
    drive_letters = [drive.device for drive in psutil.disk_partitions()]
    # check permissions
    for drive_letter in drive_letters:
        try:
            os.listdir(drive_letter)
        except:
            drive_letters.remove(drive_letter)
    return drive_letters


def faindNewName(target):
    path, file = os.path.split(target)
    name, ext = os.path.splitext(file)
    # find number in name
    num = re.findall(r'\d+', name)
    if len(num) == 0:
        num = 1
    else:
        num = int(num[0])
        name = name.replace(str(num), '')

    while os.path.exists(target):
        num += 1
        target = os.path.join(path, name + str(num) + ext)
    return target


def move_file_ifExist(source, target):
    if not os.path.exists(source):
        return
    if os.path.exists(target):
        target = faindNewName(target)
    shutil.move(source, target)
    return target


def copy_file_ifExist(source, target):
    if os.path.exists(target):
        target = faindNewName(target)
    shutil.copy(source, target)

def open_in_explorer(path):
    """
    .. todo::

       Validate all post fields

    .. todolist::
    """

    subprocess.Popen(f'explorer /select,"{path}"')