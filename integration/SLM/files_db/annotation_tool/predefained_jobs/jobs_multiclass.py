import os

from SLM import Allocator

os.environ['MONGODB_NAME'] = "files_db"
from SLM.files_db.annotation_tool.annotation import AnnotationJob

job_map = {
    "NSFWFilter": ["general", "sensitive", "explicit"],
    "worlds": ["mortal kombat", "the witcher", "nier automata", "nier replicanta", "overwatch", "mass effect",
               "fallout", "skyrim", "zelda", "bioshock", "tomb raider", "resident evil", "horizon zero dawn",
               "final fantasy", "world of warcraft", "war hammer", "star wars", "dragon ball"],
    "famous people": ["jessica rabbit", "lara croft"],
    "abstract": ["abstract", "pattern", "other"],
    "sketch binary": ["sketch color", "sketch bw", "other"],
    "comiks binary": ["comix color", "comix bw", "comix cower", "image set", "other"],
    "image genres": ["3d render", "photo", "combined", "drawing", "pixel art", "text", "other"],
    'image composition genres': ['fantasy', 'sci-fi', 'horror', 'cyberpunk', 'steampunk', 'other'],
    'image type': ['anime', 'manga', 'comic', 'cartoon', '3d sculpt', 'game screen', 'other'],
    "image_source": ["game screen", "fanart", "meme", "other"],
    'NSFWFilterCensure': ['bloor', 'black bar', 'other'],
    'rating': ['low', 'normal', 'high'],
    'image content': ['environment', 'person', 'beauty', 'object', 'other'],
    'NSFWGenres': ['traditional', 'bdsm', 'futanari', 'tentackles'],
    'char count': ['1male', '1female', '2male', '2female', '1male1female', '1futa',
                   '1teen', '2female1male', '2male1female', '2futa', '2teen', '3female', '3male']
}


def create_empty_jobs_multiclass():
    job_type = "multiclass/image"
    for job_name, choices in job_map.items():
        job: AnnotationJob = AnnotationJob.get_or_create(name=job_name)
        job.type = job_type
        job.choices = choices


config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"
Allocator.init_modules()
create_empty_jobs_multiclass()
