"""TODO: add documentation AND REFACTOR"""
import base64
import fnmatch
import os
import subprocess
import requests


class list_ext:
    @staticmethod
    def first_or_default(list_inst, default=None):
        return list_inst[0] if len(list_inst) > 0 else default


def unarchAll(folder):
    import os
    import patoolib
    from SLM.appGlue.iotools.pathtools import get_files
    # get all archives files in a folder and extract them
    # pip install patool
    # source: https://wummel.github.io/patool/
    for file in get_files(folder, ['*.zip', '*.rar', '*.7z']):
        root = os.path.dirname(file)
        patoolib.extract_archive(file, outdir=root)


def sanitize_file_path(file_path):
    # deltet ' or " on start and end of path
    file_path = file_path.strip('\'"')
    return os.path.abspath(file_path)


# function get files and directories in directory and filter by filter delegate
def getZipingFiles_and_folders(curZipDir, exclude="*.zip"):
    ZipedFiles = []
    for curentFile in os.listdir(curZipDir):
        zip = True
        # split notZipMasck to list of masks by ";" and check curentFile by each mask
        for mask in exclude.split(";"):
            if fnmatch.fnmatch(curentFile, mask):
                zip = False
                break
        if zip: ZipedFiles.append(curentFile)

    return ZipedFiles


# function get directories in directory not recursive
def getDirectories(rootSerchPath):
    Directories = []
    for file in os.listdir(rootSerchPath):
        if os.path.isdir(os.path.join(rootSerchPath, file)):
            Directories.append(os.path.join(rootSerchPath, file))

    # if return list empty return rootSerchPath
    if not Directories:
        Directories.append(rootSerchPath)
    return Directories


def human_readable_size(size, precision=2):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1
        size = size / 1024.0
    return "%.*f%s" % (precision, size, suffixes[suffixIndex])


def deleteFile_to_recycle_bin(file):
    import send2trash
    if os.path.exists(file):
        send2trash.send2trash(file)


def getFolders(directory, include_masc="*"):
    Files = []
    for root, dirs, files in os.walk(directory):
        for file in dirs:
            if fnmatch.fnmatch(file, include_masc):
                Files.append(os.path.join(root, file))
    return Files


# flatten directory. gaet all files in directory and subdirectories and move to start_time directory
def flaten_directory(path):
    import os
    import shutil
    for root, dirs, files in os.walk(path):
        for file in files:
            # get source file ProjectSettingsPath
            source_file_path = os.path.join(root, file)
            # get target file ProjectSettingsPath
            target_file_path = os.path.join(path, file)
            # create target ProjectSettingsPath if not exist
            if not os.path.exists(os.path.dirname(target_file_path)):
                os.makedirs(os.path.dirname(target_file_path))
            # copi onli if source and target file is not same
            if not source_file_path == target_file_path:
                shutil.copyfile(source_file_path, target_file_path)
                # delete source file
                os.remove(source_file_path)


def documenting_example(inimagepath, outhimagepath):
    """
    convert one imageView format to another need coresponding file extension

    This is an H1
    =============

    This is an H2
    -------------

    #. Step 1.

        * Item 1.

        * Item 2.

    #. Step 2.

    create link to `google <https://www.google.com/>`_.



    .. code-block:: python

        pygments_style = 'sphinx'

    .. note::
       This is note PathLabelVal. Use a note for information you want the user to
       pay particular attention to.


    :param inimagepath: input imageView ProjectSettingsPath
    :param outhimagepath: output imageView ProjectSettingsPath
    :return: None

      """
    return None
