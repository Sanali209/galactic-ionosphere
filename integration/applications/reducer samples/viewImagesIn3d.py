import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from vedo import Mesh, Text3D, show, settings, Plane

from SLM import Allocator
from SLM.appGlue.iotools.pathtools import get_files
from SLM.files_data_cache.tensor import Embeddings_cache
from SLM.vision.imagetotensor.backends.BLIP import CNN_Encoder_BLIP
from SLM.vision.imagetotensor.backends.DINO import CNN_Encoder_DINO

from SLM.vision.imagetotensor.backends.clip_vit_dirml import CNN_Encoder_CLIP_DML
from SLM.vision.imagetotensor.backends.inceptionV3 import CNN_Encoder_InceptionV3
from SLM.vision.imagetotensor.backends.inception_resnet_v2 import CNN_Encoder_InceptionResNetV2
from SLM.vision.imagetotensor.backends.mobile_net_v3 import CNN_encoder_ModileNetv3_Small
from SLM.vision.imagetotensor.backends.resnet import CNN_Encoder_ResNet50
from SLM.vision.imagetotensor.backends.resnetinceptionfacenet512 import CNN_Encoder_FaceNet
from SLM.vision.imagetotensor.custom.custom_emb import CNN_Encoder_custom
from SLM.vision.imagetotensor.custom_mobile_net.custom_mobv2_emb import CNN_Encoder_mv2_custom
from SLM.vision.vector_fuse import VectorReducer

config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"
Allocator.init_modules()

images_dir = r"E:\rawimagedb\repository\nsfv repo\drawn\presort\buties\Chunks"  # Укажи свой путь к папке с изображениями

emb_cache = Embeddings_cache([CNN_Encoder_FaceNet.format, CNN_Encoder_BLIP.format, CNN_Encoder_DINO.format,
                              CNN_Encoder_InceptionResNetV2.format, CNN_Encoder_InceptionV3.format,
                              CNN_encoder_ModileNetv3_Small.format, CNN_Encoder_ResNet50.format,
                              CNN_Encoder_CLIP_DML.format, CNN_Encoder_custom.format,
                              CNN_Encoder_mv2_custom.format])

# Настройки vedo
settings.default_font = "Courier"
#settings.use_depth_peeling = True


# === Загрузка изображений и эмбеддингов ===
def load_images_and_embeddings():
    embeddings = []
    image_paths = []

    for fname in get_files(images_dir, ["*.jpg", "*.jpeg", "*.png"]):
        vector = emb_cache.get_by_path(fname, CNN_Encoder_CLIP_DML.format)
        if vector is not None:
            embeddings.append(vector)
            image_paths.append(fname)

    return np.array(embeddings), image_paths


# === Главный блок ===
if __name__ == "__main__":
    embeddings, image_paths = load_images_and_embeddings()

    reducer = VectorReducer(target_dim=3, method="umap")
    reducer.load("models/reduction3dim")
    #reducer.fit("clip", embeddings)

    vectors_3d = np.array([reducer.transform("clip", e) for e in tqdm(embeddings)])

    actors = []
    for img_path, position in tqdm(zip(image_paths, vectors_3d)):
        plane = Plane(pos=position, s=(0.15, 0.15), c='white')
        plane.texture(img_path)
        label = Text3D(os.path.basename(img_path), pos=position + (0, 0, 0.05), s=0.02, c='black')
        actors += [plane, label]

    show(actors, __doc__, axes=1, viewup='z', bg='white')
