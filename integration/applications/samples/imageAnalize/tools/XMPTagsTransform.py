from tqdm import tqdm


from SLM.NLPSimple.NLPPipline import NLPPipline, NLPTextReplaceByRegexDict
from SLM.appGlue.iotools.pathtools import get_files
from SLM.groupcontext import group
from SLM.metadata.MDManager.mdmanager import MDManager
from samples.imageAnalize.tools.Tag_transform_dict import replace_dict

# create paiplain
NLP_pipline = NLPPipline()

with group():
    replacer = NLPTextReplaceByRegexDict()
    replacer.dictionary = replace_dict
    NLP_pipline.operations.append(replacer)


def start_from(string, prefix: list):
    for pref in prefix:
        if string.startswith(pref):
            return True
    return False


def replace(string, prefix: list):
    for pref in prefix:
        if string.startswith(pref):
            return string[len(pref):]
    return string


pref_list = ["auto|FromDescription|ImageCaprioner_TO8|tags|", "deepdanboru|"]
transformed_tag_prefix = "auto|transformed|"

work_directory = r'F:\rawimagedb\repository\safe repo\presort'
file_paths = get_files(work_directory, ['*.jpg', '*.png'])
images_dict = {}


def run():
    for file_path in tqdm(file_paths):
        xmpmeta = MDManager(file_path)
        xmpmeta.Read()
        xmp_tags = xmpmeta.metadata.get('XMP:Subject', [])
        if isinstance(xmp_tags, str):
            xmp_tags = xmp_tags.split(",")
        new_tags = []
        for xmp_tag in xmp_tags:
            print(xmp_tag)
            if not start_from(xmp_tag, pref_list):
                new_tags.append(xmp_tag)
                continue

            tag_without_prefix = replace(xmp_tag, pref_list).strip()
            tag_without_prefix_start = tag_without_prefix
            print(tag_without_prefix_start)
            NLP_pipline.text = tag_without_prefix
            NLP_pipline.run()
            transformed_text = NLP_pipline.text
            print(transformed_text)
            if transformed_text == tag_without_prefix_start:
                new_tags.append(xmp_tag)
                continue
            else:
                trtags = transformed_text.split(",")
                # delete empty tags
                trtags = [tag for tag in trtags if (tag != "" and tag.startswith("auto|transformed|"))]
                # delete duplicates
                trtags = list(set(trtags))
                new_tags.extend(trtags)
        print(new_tags)
        xmpmeta.Clear()
        xmpmeta.metadata['XMP:Subject'] = new_tags
        xmpmeta.Save()

run()
def next_image_path(path):
    if path not in file_paths:
        return file_paths[0]
    file_paths.remove(path)
    return file_paths[0]


def editTag(path):
    next_image = next_image_path(path)
    xmpmeta = MDManager(next_image)
    xmpmeta.Read()
    xmp_tags = xmpmeta.metadata.get('XMP:Subject', [])
    if isinstance(xmp_tags, str):
        xmp_tags = xmp_tags.split(",")
    # remove tags start with auto|transformed|
    xmp_tags = [tag for tag in xmp_tags if not tag.startswith(transformed_tag_prefix)]
    # remmove tags start with manual|
    xmp_tags = [tag for tag in xmp_tags if not tag.startswith("manual|")]
    xmptagstext = ",".join(xmp_tags)

    return xmptagstext, next_image, next_image


import gradio as gr

with gr.Blocks() as gui:
    gui.title = "edit tags"
    curent_path = gr.Textbox(label="curent_path")
    metadata = gr.Textbox(label="unprocesed_tag")
    #image = gr.Image(label="image", elem_id="image_up")

    next = gr.Button(value="next")
    next.click(fn=editTag, inputs=[curent_path], outputs=[metadata,  curent_path])

gui.launch()
