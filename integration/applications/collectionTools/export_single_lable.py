from SLM.appGlue.images.image_anotation.imageanotation import DataSetExporterImageMultiClass_dirs, AnnotationManager
from SLM.appGlue.images.imagesidecard import SideCardDB

# todo integrate to annotation pipline

export_to_path = r'G:\My Drive\rawdb\db export\rating'

image_db = SideCardDB()

# job_name = 'image genres'

job_name = 'rating'


exporter = DataSetExporterImageMultiClass_dirs()

jobManager: AnnotationManager = AnnotationManager()

job = jobManager.set_current_job_by_name(job_name)

exporter.ExportToDataset(export_to_path, job)
