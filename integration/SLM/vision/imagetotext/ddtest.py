import os
import re
from functools import lru_cache
from typing import List, Mapping, Tuple

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download


def _yield_tags_from_txt_file(txt_file: str):
    with open(txt_file, 'r') as f:
        for line in f:
            if line:
                yield line.strip()


@lru_cache()
def get_deepdanbooru_tags() -> List[str]:
    tags_file = hf_hub_download('chinoll/deepdanbooru', 'tags.txt')
    return list(_yield_tags_from_txt_file(tags_file))


@lru_cache()
def get_deepdanbooru_onnx() -> ort.InferenceSession:
    onnx_file = hf_hub_download('chinoll/deepdanbooru', 'deepdanbooru.onnx')
    return ort.InferenceSession(onnx_file)


def image_preprocess(image: Image.Image) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')

    o_width, o_height = image.size
    scale = 512.0 / max(o_width, o_height)
    f_width, f_height = map(lambda x: int(x * scale), (o_width, o_height))
    image = image.resize((f_width, f_height))

    data = np.asarray(image).astype(np.float32) / 255  # H x W x C
    height_pad_left = (512 - f_height) // 2
    height_pad_right = 512 - f_height - height_pad_left
    width_pad_left = (512 - f_width) // 2
    width_pad_right = 512 - f_width - width_pad_left
    data = np.pad(data, ((height_pad_left, height_pad_right), (width_pad_left, width_pad_right), (0, 0)),
                  mode='constant', constant_values=0.0)

    assert data.shape == (512, 512, 3), f'Shape (512, 512, 3) expected, but {data.shape!r} found.'
    return data.reshape((1, 512, 512, 3))  # B x H x W x C


RE_SPECIAL = re.compile(r'([\\()])')


def image_to_deepdanbooru_tags(image: Image.Image, threshold: float,
                               use_spaces: bool, use_escape: bool, include_ranks: bool, score_descend: bool) \
        -> Tuple[str, Mapping[str, float]]:
    tags = get_deepdanbooru_tags()
    session = get_deepdanbooru_onnx()
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    result = session.run(output_names, {input_name: image_preprocess(image)})[0]
    filtered_tags = {
        tag: float(score) for tag, score in zip(tags, result[0])
        if score >= threshold
    }

    text_items = []
    tags_pairs = filtered_tags.items()
    if score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
    for tag, score in tags_pairs:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace('_', ' ')
        if use_escape:
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{score:.3f})"
        text_items.append(tag_outformat)
    output_text = ', '.join(text_items)

    return output_text, filtered_tags


if __name__ == '__main__':
    path = r"E:\rawimagedb\repository\nsfv repo\furi\furi autors\Kadath\Gallery\1aa343642caf37afd1c0ec361cc5d50a.jpg"
    image = Image.open(path)
    res = image_to_deepdanbooru_tags(image, 0.5, False, False, False, True)
    print(res[0])

