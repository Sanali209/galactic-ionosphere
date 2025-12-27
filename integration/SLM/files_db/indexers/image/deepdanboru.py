from timm.data import create_transform, resolve_data_config

from SLM.files_db.components.fs_tag import TagRecord
from SLM.files_db.indexers.image.content_md5 import files_db_indexer
from SLM.indexerpyiplain.idexpyiplain import ItemIndexer
from SLM.vision.imagetotext.backends.deep_dan_boru import DeepDanbury
from torch import Tensor, nn
import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.nn import functional as F

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


class DeepDunBoru(files_db_indexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fieldName = "deepdanbooru"

    def index(self, parent_indexer: ItemIndexer, item, need_index):
        from SLM.files_db.components.File_record_wraper import FileRecord
        file_item = FileRecord(item["_id"])

        dd_backend = DeepDanbury()
        resalt = dd_backend.get_label_from_path(file_item.full_path)
        labels = resalt.split(",")
        #remove witespace
        labels = [label.strip() for label in labels]
        item_tags = item.get("tags", [])
        for label in labels:
            label_clear = label.strip()
            label_clear = "auto/deepdanboru/" + label_clear
            tag = TagRecord.get_or_create(fullName=label_clear)
            item_tags.append(tag.fullName)
        item_tags = list(set(item_tags))
        item["tags"] = item_tags

        parent_indexer.shared_data["item_indexed"] = True
        self.mark_as_indexed(item, parent_indexer)


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
        repo_id: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
        probs: Tensor,
        labels: LabelData,
        gen_threshold: float,
        char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


class SmilingWolfTagger(files_db_indexer):
    repo_id = "SmilingWolf/wd-vit-tagger-v3"
    model: nn.Module = timm.create_model("hf-hub:" + "SmilingWolf/wd-vit-tagger-v3").eval()
    state_dict = timm.models.load_state_dict_from_hf("SmilingWolf/wd-vit-tagger-v3")
    res = model.load_state_dict(state_dict)
    labels: LabelData = load_labels_hf("SmilingWolf/wd-vit-tagger-v3")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fieldName = "SmilingWolfTagger"

    def get_ratings(self,ratings: dict[str, int]) -> str:
        """
        get ratings by bigest value
        :param ratings: dict with ratings
        :return: string with ratings
        """
        rating_val = None
        rating_max_val = 0
        for key, value in ratings.items():
            if value > rating_max_val:
                rating_max_val = value
                rating_val = key
        return rating_val

    def index(self, parent_indexer: ItemIndexer, item, need_index):
        from SLM.files_db.components.File_record_wraper import FileRecord
        file_item = FileRecord(item["_id"])
        img_input: Image.Image = Image.open(file_item.full_path)
        # ensure image is RGB
        img_input = pil_ensure_rgb(img_input)
        # pad to square with white background
        img_input = pil_pad_square(img_input)
        # run the model's input transform to convert to tensor and rescale
        inputs: Tensor = SmilingWolfTagger.transform(img_input).unsqueeze(0)
        # NCHW image RGB to BGR
        inputs = inputs[:, [2, 1, 0]]

        with torch.inference_mode():
            # move model to GPU, if available
            if torch_device.type != "cpu":
                model = SmilingWolfTagger.model.to(torch_device)
                inputs = inputs.to(torch_device)
            else:
                model = SmilingWolfTagger.model
            # run the model
            outputs = model.forward(inputs)
            # apply the final activation function (timm doesn't support doing this internally)
            outputs = F.sigmoid(outputs)
            # move inputs, outputs, and model back to to cpu if we were on GPU
            if torch_device.type != "cpu":
                inputs = inputs.to("cpu")
                outputs = outputs.to("cpu")
                model = model.to("cpu")

        caption, taglist, ratings, character, general = get_tags(
            probs=outputs.squeeze(0),
            labels=SmilingWolfTagger.labels,
            gen_threshold=0.35,
            char_threshold=0.75,
        )

        item_tags = item.get("tags", [])
        taglist = taglist.strip().split(",")
        for label in taglist:
            label_clear = label.strip()
            label_clear = "auto/wd_tag/" + label_clear
            tag = TagRecord.get_or_create(fullName=label_clear)
            item_tags.append(tag.fullName)
        for character in character.keys():
            character_clear = character.strip()
            character_clear = "auto/wd_character/" + character_clear
            tag = TagRecord.get_or_create(fullName=character_clear)
            item_tags.append(tag.fullName)
        rating = self.get_ratings(ratings)
        if rating is not None:
            rating_clear = "auto/wd_rating/" + str(rating)
            tag = TagRecord.get_or_create(fullName=rating_clear)
            item_tags.append(tag.fullName)
        item_tags = list(set(item_tags))
        item["tags"] = item_tags

        parent_indexer.shared_data["item_indexed"] = True
        self.mark_as_indexed(item, parent_indexer)
