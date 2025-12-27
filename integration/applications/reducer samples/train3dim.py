
import numpy as np
from tqdm import tqdm

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

images_dir = r"E:\rawimagedb\repository\nsfv repo\drawn\presort\buties"
images_pTHS = get_files(images_dir, ["*.jpg", "*.jpeg", "*.png"])
emb_cache = Embeddings_cache([CNN_Encoder_FaceNet.format, CNN_Encoder_BLIP.format, CNN_Encoder_DINO.format,
                              CNN_Encoder_InceptionResNetV2.format, CNN_Encoder_InceptionV3.format,
                              CNN_encoder_ModileNetv3_Small.format, CNN_Encoder_ResNet50.format,
                              CNN_Encoder_CLIP_DML.format, CNN_Encoder_custom.format,
                              CNN_Encoder_mv2_custom.format])
clip_vectors = []
for img_path in tqdm(images_pTHS):
    vector = emb_cache.get_by_path(img_path, CNN_Encoder_CLIP_DML.format)
    if vector is not None:
        clip_vectors.append(vector)

clip_vectors = np.array(clip_vectors)

mobil_net_vectors = []
for img_path in tqdm(images_pTHS):
    vector = emb_cache.get_by_path(img_path, CNN_encoder_ModileNetv3_Small.format)
    if vector is not None:
        mobil_net_vectors.append(vector)
mobil_net_vectors = np.array(mobil_net_vectors)
dino_vectors = []
for img_path in tqdm(images_pTHS):
    vector = emb_cache.get_by_path(img_path, CNN_Encoder_DINO.format)
    if vector is not None:
        dino_vectors.append(vector)
dino_vectors = np.array(dino_vectors)


# 2. Создаём редуктор
reducer = VectorReducer(target_dim=3, method="umap")

# 3. Обучаем модель проекции для каждого типа эмбеддингов
reducer.fit("clip", clip_vectors)
reducer.fit("dino", dino_vectors)
reducer.fit("mobilenet", mobil_net_vectors)

# 4. Сохраняем модели (опционально)
reducer.save("models/reduction3dim")
