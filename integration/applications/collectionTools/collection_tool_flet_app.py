import os

from SLM.flet.flet_ext import FletGlueApp
from applications.collectionTools.views.collection_list_view.file_list_exp import FileListEditorView
from loguru import logger

logger.add("file_list_exp.log")

os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'

app = FletGlueApp()
app.start_view = FileListEditorView()

app.run()
