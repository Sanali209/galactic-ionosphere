import os
import shutil

from tqdm import tqdm

from SLM.appGlue.iotools.pathtools import get_files
from SLM.vision.dubFileHelper import DuplicateFindHelper

if __name__ == '__main__':
    dubfinder = DuplicateFindHelper()
    folder = r'F:\rawimagedb\repository\safe repo\asorted images\0\unrated'
    filelist = get_files(folder, exts=['*.jpg', '*.png', '*.jpeg'])
    sorted = dubfinder.sort_images_by_futures_base(filelist)
    sorted_folder_path = os.path.join(folder, 'sorted')
    if not os.path.exists(sorted_folder_path):
        os.mkdir(sorted_folder_path)

    counter = 0
    def formateCounter(counter):
        # diget format 00001
        return str(counter).zfill(5)

    for file in tqdm(sorted):
        counter += 1
        formated = formateCounter(counter)
        filename = os.path.basename(file)
        newfilename = formated + '_' + filename
        newfilepath = os.path.join(sorted_folder_path, newfilename)
        #copy file to destination
        shutil.copy(file, newfilepath)