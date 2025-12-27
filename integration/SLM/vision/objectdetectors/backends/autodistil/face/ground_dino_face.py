import xmlrpc

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from supervision import Detections

from SLM.appGlue.helpers import get_colab_xmlrpc_ngrock_url, image_to_zipped_base64
from SLM.files_data_cache.pool import PILPool
from SLM.vision.objectdetectors.object_detect import object_detector_backend
from xmlrpc.client import ServerProxy
from PIL import Image


class object_detector_groundDino(object_detector_backend):
    """
    This is a wrapper for the GroundingDINO model from AutoDistill.
    return a list of dictionaries with the following keys:
    - version: the version of the model
    - label: the label of the detected object (e.g. "face", "person")
    - region_format: the format of the region (e.g. "abs_xywh")
    - score: the score of the detected object
    - region: the region of the detected object in the format [x, y, width, height]
    - image: the cropped image of the detected object in PIL format
    - image_path: the path of the image
    """
    name = "object_detector_groundDino"
    version = "1.0.0"

    def __init__(self):
        self.ontology = CaptionOntology(
            {
                "face": "face",
                "person": "person"
            }
        )
        self.model = GroundingDINO(ontology=self.ontology)

    def detect(self, image: Image):
        return []

    def detect_by_path(self, image_path: str) -> any:

        try:
            results = []
            result: Detections = self.model.predict(image_path)

            for img_obj in result:
                res_dict = {"version": self.version, "label": "face", "region_format": "abs_xywh",
                            'score': float(img_obj[2]),
                            'region': [img_obj[0][0], img_obj[0][1], img_obj[0][2] - img_obj[0][0],
                                       img_obj[0][3] - img_obj[0][1]]}
                res_dict['region'] = [int(x) for x in res_dict['region']]

                #convert numpay array to regular array
                label = self.ontology.classes()[img_obj[3]]
                res_dict['label'] = label
                image_size_array = img_obj[0]
                image_size_array = image_size_array.tolist()

                face_image = PILPool.get_pil_image(image_path, copy=False).crop(image_size_array)
                res_dict['image'] = face_image
                results.append(res_dict)
            return results
        except Exception as e:
            print(e)
            return []


class object_detector_groundDinoXMLRPC(object_detector_backend):
    name = "object_detector_groundDino"
    version = "1.0.0"
    url = get_colab_xmlrpc_ngrock_url()

    def __init__(self):
        super().__init__()

    def detect(self, image: Image):
        return []

    def detect_by_path(self, image_path: str) -> any:

        try:
            with xmlrpc.client.ServerProxy(self.url) as proxy:
                ziped_image = image_to_zipped_base64(image_path)
                results = proxy.detect_face(ziped_image)

            for img_obj in results:
                #convert numpay array to regular array
                image_size_array = (img_obj['region'][0], img_obj['region'][1],
                                    img_obj['region'][0] + img_obj['region'][2],
                                    img_obj['region'][1] + img_obj['region'][3])
                face_image = PILPool.get_pil_image(image_path, copy=False).crop(image_size_array)
                img_obj['image'] = face_image
            return results
        except Exception as e:
            print(e)
            return []



