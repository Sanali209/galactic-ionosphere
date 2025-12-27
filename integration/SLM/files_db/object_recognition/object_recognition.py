# todo implement object detection subsystem
import os
from enum import Enum

from SLM.appGlue.core import Allocator
from SLM.files_data_cache.pool import PILPool
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.collectionItem import CollectionRecord
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.mongoext.MongoClientEXT_f import MongoClientExt
from SLM.mongoext.wraper import FieldPropInfo, MongoRecordWrapper
from PIL import Image


class RegionFormat(Enum):
    """
    Enum for representing different formats of bounding box regions.

    Formats:
    - abs_xywh: Absolute coordinates [x, y, width, height]
    - abs_xyxy: Absolute coordinates [x1, y1, x2, y2]
    - norm_xywh: Normalized [x, y, width, height] (values 0-1)
    - norm_xyxy: Normalized [x1, y1, x2, y2] (values 0-1)
    - proc_xywh: Percentage [x, y, width, height] (values 0-100)
    - proc_xyxy: Percentage [x1, y1, x2, y2] (values 0-100)
    """
    abs_xywh = 'abs_xywh'
    abs_xyxy = 'abs_xyxy'
    norm_xywh = 'norm_xywh'
    norm_xyxy = 'norm_xyxy'
    proc_xywh = 'proc_xywh'
    proc_xyxy = 'proc_xyxy'


class DetectionObjectClass(MongoRecordWrapper):
    """
    Represents a class of detected objects in the object recognition system.

    Attributes:
        name (str): The name of the detection class
    """
    name: str = FieldPropInfo('name', str, None)

    def get_recognized_objects(self) -> list['Recognized_object']:
        """
        Retrieves all recognized objects that belong to this class.

        Returns:
            list[Recognized_object]: List of recognized objects with this class
        """
        return Recognized_object.find({'obj_class_id': self._id})

    @classmethod
    def get(cls, name):
        """
        Get a detection object class by name.

        Args:
            name (str): The name of the detection class to find

        Returns:
            DetectionObjectClass: The class object if found, None otherwise
        """
        return cls.find_one({'name': name})


class Detection(CollectionRecord):
    """
    Detection record representing a single detected object in an image.

    This class stores information about detected objects including their position,
    classification, confidence score, and references to parent objects.

    Attributes:
        itemType (str): Override itemType of CollectionRecord, always 'Detection'
        obj_name (str): Name of the detected object
        object_class (str): Name of the object class
        rect_region (list[int]): Region of object in image [x,y,w,h] or [x1,y1,x2,y2]
        region_format (str): Format of region, uses RegionFormat enum values
        backend (str): Name of detection backend that identified the object
        obj_image_path (str): Path to cropped image of just the object
        score (float): Confidence score of detection (0.0 to 1.0)
        is_wrong (bool): Flag indicating if the detection has been marked as incorrect
        parent_obj_id (ObjectId): Reference to parent Recognized_object
        parent_image_id (ObjectId): Reference to parent FileRecord containing source image
    """
    itemType: str = FieldPropInfo('item_type', str, 'Detection')
    obj_name: str = FieldPropInfo('obj_name', str, None)
    object_class: str = FieldPropInfo('object_class', str, None)
    rect_region: list = FieldPropInfo('rect_region', list[int], [])
    region_format: str = FieldPropInfo('region_format', str, "abs_xywh")
    backend: str = FieldPropInfo('backend', str, None)
    obj_image_path: str = FieldPropInfo('obj_image_path', str, None)
    score: float = FieldPropInfo('score', float, 0.0)
    is_wrong: bool = FieldPropInfo('is_wrong', bool, False)
    parent_obj_id: object = FieldPropInfo('parent_obj_id', str, None)
    parent_image_id: object = FieldPropInfo('parent_image_id', str, None)

    def get_rect(self, format_=RegionFormat.abs_xywh.value):
        """
        Get the bounding box rectangle in the specified format.

        Note: Currently only supports conversion between abs_xywh and abs_xyxy
        when the current format is abs_xywh.

        Args:
            format_ (str): Target format from RegionFormat enum

        Returns:
            tuple: Bounding box coordinates in the requested format
        """
        # not full implementated only for abs_xywh and abs_xyxy if current format is abs_xywh
        reg = []
        cur_format = RegionFormat(self.region_format)
        if cur_format is None:
            self.region_format = RegionFormat.abs_xywh.value
            cur_format = RegionFormat.abs_xywh
        if cur_format == RegionFormat.abs_xywh:
            if format_ == RegionFormat.abs_xywh.value:
                reg = self.rect_region
            if format_ == RegionFormat.abs_xyxy.value:
                reg = [self.rect_region[0], self.rect_region[1], self.rect_region[0] + self.rect_region[2],
                       self.rect_region[1] + self.rect_region[3]]
                # return as tuple
        return tuple(reg)

    def set_rect(self, rect, format_=RegionFormat.abs_xywh.value, edit_name=False):
        """
        Set the region of the detected object.

        Updates the bounding box and regenerates the object crop image.

        Args:
            rect (list): Bounding box coordinates in specified format
            format_ (str): Format of the input rect, from RegionFormat enum
            edit_name (bool): Whether to change the name of the cropped object image
                            to invalidate cached values
        """
        rect = [int(x) for x in rect]
        if self.region_format is None:
            self.region_format = RegionFormat.abs_xywh.value

        cur_format = RegionFormat(self.region_format)

        if cur_format == RegionFormat.abs_xywh:
            set_region = []
            if format_ == RegionFormat.abs_xywh.value:
                set_region = rect
            if format_ == RegionFormat.abs_xyxy.value:
                set_region = [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]
            # check if region changed
            if set_region != self.rect_region:
                self.rect_region = set_region
                self.update_rect_image(edit_name)

    @property
    def parent_file(self):
        """
        Get the parent file record containing the source image.

        Returns:
            FileRecord: The parent file record or None if not set
        """
        if self.parent_image_id is None:
            return None
        return FileRecord(self.parent_image_id)

    @property
    def parent_obj(self):
        """
        Get the parent recognized object.

        Returns:
            Recognized_object: The parent recognized object or None if not set
        """
        if self.parent_obj_id is None:
            return None
        return Recognized_object.find_one({'_id': self.parent_obj_id})

    def set_ref_to(self, ref_id):
        """
        Create a relation from a reference object to this detection.

        Args:
            ref_id: ID of the object to create relation from
        """
        # reference manage by ref class
        RelationRecord.get_or_create(from_=CollectionRecord(ref_id), to_=self, type="detection")

    def delete(self):
        """
        Delete this detection, its cropped image file, and all relations.
        """
        # delete image with object
        if self.obj_image_path is not None:
            os.remove(self.obj_image_path)
        # delete all relations and delete record
        RelationRecord.delete_all_relations(self)
        self.delete_rec()

    def set_recognized_object(self, rec_obj):
        """
        Set the parent recognized object for this detection.

        Args:
            rec_obj (Recognized_object): The recognized object to set as parent
        """
        self.parent_obj_id = rec_obj._id

    def update_rect_image(self, edit_name=False):
        """
        Update the cropped image of the detected object.

        Extracts the region from the parent image and saves it as a separate file.

        Args:
            edit_name (bool): Whether to change the filename of the cropped image
        """
        if self.parent_image_id is None:
            return
        image = FileRecord.find_one({'_id': self.parent_image_id})
        if image is None:
            return
        image_path = self.parent_file.full_path
        save_im = self.obj_image_path
        rect = self.get_rect("abs_xyxy")
        pimage: Image = PILPool.get_pil_image(image_path)
        res = pimage.crop(rect)
        if edit_name:
            file_folder = os.path.dirname(save_im)
            file_name = os.path.basename(save_im).split('.')[0]
            self.obj_image_path = os.path.join(file_folder, f"e{file_name}.jpg")
        res.save(self.obj_image_path)

    def get_class(self):
        """
        Get the detection object class for this detection.

        Returns:
            DetectionObjectClass: The object class
        """
        return DetectionObjectClass.find_one({'name': self.object_class})

    def set_class(self, obj_class):
        """
        Set the object class for this detection.

        Args:
            obj_class (DetectionObjectClass): The object class to set
        """
        self.object_class = obj_class.name

    def get_parent_fileRecord(self):
        """
        Get the parent file record.

        Returns:
            FileRecord: The parent file record
        """
        return FileRecord.find_one({'_id': self.parent_image_id})

    def set_parent_fileRecord(self, parent: FileRecord):
        """
        Set the parent file record.

        Args:
            parent (FileRecord): The parent file record to set
        """
        self.parent_image_id = parent._id

    def delete_all_childs_detections(self, parent_obj):
        """
        Delete all child detections of a parent object.

        Args:
            parent_obj (Recognized_object): The parent object
        """
        detections = Detection.find({'parent_obj_id': parent_obj._id})
        for detection in detections:
            detection.delete()

    @classmethod
    def del_detection(cls, parent_obj):
        """
        Schedule deletion of all detections associated with a parent file.

        Args:
            parent_obj (FileRecord): The parent file record
        """
        if not isinstance(parent_obj, FileRecord):
            return
        Detection.collection().delete_many({'parent_image_id': parent_obj._id})


class Recognized_object(CollectionRecord):
    """
    Represents a recognized object entity that can have multiple detections.

    This class serves as a grouping mechanism for multiple detections of the
    same object across different images or videos.

    Attributes:
        itemType (str): Override itemType of CollectionRecord, always 'Recognized_object'
        name (str): Name of the recognized object
        obj_class (ObjectId): Reference to DetectionObjectClass record
    """
    itemType: str = FieldPropInfo('item_type', str, 'Recognized_object')
    name: str = FieldPropInfo('name', str, None)
    obj_class: object = FieldPropInfo('obj_class_id', object, None)

    def get_detections(self):
        """
        Get all detections of this recognized object.

        Returns:
            list[Detection]: List of detections associated with this object
        """
        return Detection.find({'parent_obj_id': self._id})

    def set_obj_class(self, obj_class):
        """
        Set the object class for this recognized object.

        Args:
            obj_class (DetectionObjectClass): The object class to set
        """
        self.obj_class = obj_class._id

    def set_detection(self, detection):
        """
        Associate a detection with this recognized object.

        Args:
            detection (Detection): The detection to associate
        """
        detection.parent_obj_id = self._id

    def GetAll(self, class_=None):
        if class_ is None:
            return self.find({})
        return self.find({'obj_class_id': class_._id})
