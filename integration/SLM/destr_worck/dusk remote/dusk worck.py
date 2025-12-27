from PIL import Image
from dask.distributed import Client
from tqdm import tqdm

from SLM.appGlue.iotools.pathtools import get_files
from SLM.files_data_cache.pool import PILPool

from SLM.iterable.bach_builder import BatchBuilder
from SLM.vision.imagetotext.ImageToLabel import  multiclass_comix_bf

dusk_client = Client("tcp://4.tcp.eu.ngrok.io:16404")
backend = None
def _BuildIndex_dusk( image_paths: list[str]):
    from SLM.files_data_cache.thumbnail import ImageThumbCache
    from SLM.files_data_cache.imagedatacache import ImageDataCacheManager
    # dushbord on http://localhost:8787/

    bach_b = BatchBuilder(image_paths, 16)
    def get_embedding(pil_image,md_5):
        global backend
        if backend is None:
            # Initialize backend once per worker process
            #backend = multiclass_comix_bf()
            pass
        label = '1'# backend.get_label_from_pil_image(pil_image)
        print(label)
        return md_5,  label

    for buch in tqdm(bach_b.bach.values()):
        pil_images = [PILPool.get_pil_image(ImageThumbCache.instance().get_thumb(path)) for path in buch]
        md_5 = [ImageDataCacheManager.instance().path_to_md5(path) for path in buch]
        futures = dusk_client.map(
            get_embedding, pil_images, md_5)
        dusk_client.gather(futures)
        for future in futures:
            res = future.result()
            print(res)

if __name__ == "__main__":
    files = get_files(r"E:\rawimagedb\repository\safe repo\presort\combined",["*.jpg"])
    _BuildIndex_dusk( files)