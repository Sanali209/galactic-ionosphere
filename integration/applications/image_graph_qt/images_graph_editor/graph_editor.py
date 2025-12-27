import json
import math
import os
import re
from enum import Enum

import loguru
from PySide6.QtWidgets import (QApplication, QInputDialog, QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView,
                               QGraphicsEllipseItem,
                               QGraphicsLineItem, QPushButton, QColorDialog, QGraphicsItem, QComboBox, QHBoxLayout,
                               QMenu, QGraphicsRectItem, QGraphicsSceneContextMenuEvent, QFileDialog, QLineEdit)
from PySide6.QtCore import QRectF, QPointF, Qt, QObject
from PySide6.QtGui import QPainter, QDragEnterEvent, QDropEvent, QContextMenuEvent, QAction, QPen
from bson import ObjectId
from tqdm import tqdm

from SLM.appGlue.core import Allocator, ContextMode
from SLM.files_data_cache.tensor import Embeddings_cache
from SLM.files_db.components.File_record_wraper import FileRecord, get_file_record_by_folder
from SLM.files_db.components.fs_tag import TagRecord
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget
from SLM.pySide6Ext.widgets.graph_editor import Node, GraphScene, GraphView, Edge, NodeGroup
from SLM.pySide6Ext.widgets.tools import WidgetBuilder
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
from SLM.vision.vector_fuse import VectorReducer, EmbeddingFusion


# notes
# arrange add possibility arrange only selected nodes

class BaseVars:
    default_node_size = 128
    reducer_models_path = r"D:\Sanali209\Python\applications\reducer samples\models"


class BaseMode(ContextMode):
    def __init__(self):
        super().__init__()
        self._panning = False
        self._last_mouse_pos = None

    def on_drag_enter(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def on_drag_drop(self, event, scene):
        scene_pos = scene.views()[0].mapToScene(event.position().toPoint())
        last_pos = scene_pos
        add_node_list = []
        if event.mimeData().hasUrls():
            for url in tqdm(event.mimeData().urls()):
                file_path = url.toLocalFile()
                file_record = FileRecord.get_record_by_path(file_path)
                last_pos = QPointF(last_pos.x() + 512, last_pos.y())

                if file_record is not None:
                    if not scene.is_node_data_context_exists(file_record):
                        node = FileRecordNode(last_pos.x(), last_pos.y(), 128, file_record)
                        add_node_list.append(node)
                    else:
                        #set node to last pos
                        node = scene.get_node_by_data_context(file_record)
                        if node is not None:
                            node.setPos(last_pos)
            event.acceptProposedAction()
            if len(add_node_list) > 0:
                scene.addItems(add_node_list)
            return
        if event.mimeData().hasText():
            try:
                json_data = json.loads(event.mimeData().text())
                for record_dat in tqdm(json_data):
                    if record_dat["type"] == 'FileRecord':
                        id = record_dat["id"]
                        file_record = FileRecord.find_one({"_id": ObjectId(id)})
                        last_pos = QPointF(last_pos.x() + 512, last_pos.y())

                        if file_record is not None:
                            if not scene.is_node_data_context_exists(file_record):
                                node = FileRecordNode(last_pos.x(), last_pos.y(), 128, file_record)
                                add_node_list.append(node)
                            else:
                                # set node to last pos
                                node = scene.get_node_by_data_context(file_record)
                                if node is not None:
                                    node.setPos(last_pos)
                    if record_dat["type"] == 'TagRecord':
                        tag_name = record_dat["name"]
                        tag_record = TagRecord.find_one({"_id": ObjectId(tag_name)})
                        tag_node = TagNode(last_pos.x(), last_pos.y(), 128, tag_record)

                        add_node_list.append(tag_node)
            except Exception as e:
                loguru.logger.error(f"Error parsing json data {e}")
        if len(add_node_list) > 0:
            scene.addItems(add_node_list)
        event.acceptProposedAction()

    def mousePressEvent(self, scene, event):
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            item = scene.itemAt(event.scenePos(), scene.views()[0].transform())
            if isinstance(item, Node):
                item.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
                if scene.last_node and scene.last_node != item:
                    # Create an edge between the last node and the current node

                    if isinstance(item, PinNode):
                        edge = RelationRecordEdge(scene.last_node, item, None)
                        scene.addItem(edge)

                    relation = RelationRecord.get_or_create(from_=scene.last_node.file_record, to_=item.file_record,
                                                            type="similar_search")
                    if relation is not None and not RelationRecordEdge.is_relation_exist(scene, relation):
                        edge = RelationRecordEdge(scene.last_node, item, relation)
                        scene.addItem(edge)
                        scene.last_node = None
                    else:
                        relation_edge = RelationRecordEdge.get_relation_edge(scene, relation)
                        relation.set_field_val("sub_type", "manual")
                        if relation_edge is not None:
                            relation_edge.colorize("manual")

                else:
                    scene.last_node = item

    def mouseMoveEvent(self, scene, event):
        pass

    def mouseReleaseEvent(self, scene, event):
        # Other mouse release handling
        pass

    def keyPressEvent(self, scene, event):
        if event.key() == Qt.Key_Delete:
            scene.delete_selected_items()
        # on key pres d navigate scene to right
        elif event.key() == Qt.Key.Key_D:
            view = scene.views()[0]
            current_pos = view.horizontalScrollBar().value()
            new_pos = current_pos + 100
            view.horizontalScrollBar().setValue(new_pos)
        # on key press a navigate scene to left
        elif event.key() == Qt.Key.Key_A:
            view = scene.views()[0]
            current_pos = view.horizontalScrollBar().value()
            new_pos = current_pos - 100
            view.horizontalScrollBar().setValue(new_pos)
        # on key press w navigate scene to up
        elif event.key() == Qt.Key.Key_W:
            view = scene.views()[0]
            current_pos = view.verticalScrollBar().value()
            new_pos = current_pos - 100
            view.verticalScrollBar().setValue(new_pos)
        # on key press s navigate scene to down
        elif event.key() == Qt.Key.Key_S:
            view = scene.views()[0]
            current_pos = view.verticalScrollBar().value()
            new_pos = current_pos + 100
            view.verticalScrollBar().setValue(new_pos)




        # on key press 1 connect selected nodes to 1 pinNode
        elif event.key() == Qt.Key.Key_1:
            selected_nodes = scene.get_selected_nodes()
            if len(selected_nodes) > 0:
                all_pins = scene.get_nodes_by_data_context_type(str)
                if len(all_pins) > 0:
                    pin_node = all_pins[0]
                    for node in selected_nodes:
                        if isinstance(node, FileRecordNode):
                            edge = RelationRecordEdge(node, pin_node, None)
                            scene.addItem(edge)
        elif event.key() == Qt.Key.Key_2:
            selected_nodes = scene.get_selected_nodes()
            if len(selected_nodes) > 0:
                all_pins = scene.get_nodes_by_data_context_type(str)
                if len(all_pins) > 1:
                    pin_node = all_pins[1]
                    for node in selected_nodes:
                        if isinstance(node, FileRecordNode):
                            edge = RelationRecordEdge(node, pin_node, None)
                            scene.addItem(edge)
        elif event.key() == Qt.Key.Key_3:
            selected_nodes = scene.get_selected_nodes()
            if len(selected_nodes) > 0:
                all_pins = scene.get_nodes_by_data_context_type(str)
                if len(all_pins) > 2:
                    pin_node = all_pins[2]
                    for node in selected_nodes:
                        if isinstance(node, FileRecordNode):
                            edge = RelationRecordEdge(node, pin_node, None)
                            scene.addItem(edge)
        elif event.key() == Qt.Key.Key_4:
            selected_nodes = scene.get_selected_nodes()
            if len(selected_nodes) > 0:
                all_pins = scene.get_nodes_by_data_context_type(str)
                if len(all_pins) > 3:
                    pin_node = all_pins[3]
                    for node in selected_nodes:
                        if isinstance(node, FileRecordNode):
                            edge = RelationRecordEdge(node, pin_node, None)
                            scene.addItem(edge)
        elif event.key() == Qt.Key.Key_5:
            selected_nodes = scene.get_selected_nodes()
            if len(selected_nodes) > 0:
                all_pins = scene.get_nodes_by_data_context_type(str)
                if len(all_pins) > 4:
                    pin_node = all_pins[4]
                    for node in selected_nodes:
                        if isinstance(node, FileRecordNode):
                            edge = RelationRecordEdge(node, pin_node, None)
                            scene.addItem(edge)

    def keyReleaseEvent(self, scene, event):
        if event.key() == Qt.Key.Key_Shift:
            pass


class AddEdgeMode(BaseMode):
    def __init__(self):
        super().__init__()
        self.saved_mouse_press = None

    def activate(self):
        self.saved_mouse_press = RelationRecordEdge.mousePressEvent
        RelationRecordEdge.mousePressEvent = lambda edge, event: AddEdgeMode.edge_mousePressEvent(self, edge, event)

    def deactivate(self):
        RelationRecordEdge.mousePressEvent = self.saved_mouse_press

    @staticmethod
    def edge_mousePressEvent(self, edge, event):
        relation = edge.relation_record
        relation.set_field_val("sub_type", "manual")
        edge.colorize("manual")

    def mousePressEvent(self, scene, event):
        item = scene.itemAt(event.scenePos(), scene.views()[0].transform())

        if not isinstance(item, Node):
            return

        item.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        if scene.last_node and scene.last_node != item:
            # If Shift is pressed, just update the last node without creating an edge
            if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                scene.last_node = item
            else:
                # Create an edge between the last node and the current node
                relation = RelationRecord.get_or_create(from_=scene.last_node.file_record, to_=item.file_record,
                                                        type="similar_search")
                if relation is not None and not RelationRecordEdge.is_relation_exist(scene, relation):
                    edge = RelationRecordEdge(scene.last_node, item, relation)
                    scene.addItem(edge)
                    scene.last_node = item
                else:
                    relation_edge = RelationRecordEdge.get_relation_edge(scene, relation)
                    relation.set_field_val("sub_type", "manual")
                    if relation_edge is not None:
                        relation_edge.colorize("manual")
        else:
            scene.last_node = item


class ImageSearchRelSubType(Enum):
    wrong = "wrong"
    similar = "similar"
    not_similar = "near_dub"
    similar_style = "similar_style"
    manual = "manual"
    some_person = "some_person"
    some_image_set = "some_image_set"
    other = "other"
    hiden = "hiden"
    none = "none"


class ImageSearchRelFilter:
    def __init__(self):
        self.threshold_min = 0.0
        self.threshold_max = 1.0
        self.file_reit_min = 0.0
        self.file_reit_max = 1.0
        self.working_folder = r"E:\rawimagedb\repository\nsfv repo\drawn\presort\buties"
        self.subtypes = [ImageSearchRelSubType.none.value]
        #for rel_type in ImagaeSearchRelSubType:
        #self.subtypes.append(rel_type.value)
        #self.subtypes.remove(ImagaeSearchRelSubType.wrong.value)

    def get_filter(self):
        return {"distance": {"$gt": self.threshold_min,
                             "$lt": self.threshold_max},
                "type": "similar_search",
                "sub_type": {"$in": [x for x in self.subtypes]},
                }

    def get_pipline_filter(self):
        return {"relations.distance": {"$gt": self.threshold_min, "$lt": self.threshold_max},
                "relations.type": "similar_search",
                "relations.sub_type": {"$in": [x for x in self.subtypes]},
                }

    def get_pipline(self):
        pipeline = [
            # Stage 1: Match file records by local_path regex
            {
                "$match": {
                    "local_path": {"$regex": f"^{re.escape(self.working_folder)}"}
                }
            },
            # Stage 2: Lookup relations collection
            {
                "$lookup": {
                    "from": "relation_records",
                    "localField": "_id",
                    "foreignField": "from_id",
                    "as": "relations"
                }
            },
            # Stage 3: Filter relations with rel_filter
            {
                "$unwind": {
                    "path": "$relations",
                    "preserveNullAndEmptyArrays": False  # Optional
                }
            },
            # Stage 4: Match relations by distance
            {
                "$match": self.get_pipline_filter()
            },
            # Stage 4: Group by _id
            {
                "$group": {
                    "_id": "$_id",
                    "relations": {"$push": "$relations"}
                }
            },
            # filter relations list not empty
            {
                "$match": {
                    "relations": {"$ne": []}
                }
            },
            # sort by name
            {
                "$sort": {"name": 1}
            }
        ]
        return pipeline

    def is_subtype_in_filter(self, subtype: ImageSearchRelSubType):
        return subtype in self.subtypes


class TagNode(Node):
    def __init__(self, x, y, radius, TagRec: TagRecord):
        if TagRec is None:
            pass
        super().__init__(x, y, radius, TagRec.fullName)
        self.name = TagRec.fullName
        self.data_context = TagRec

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.name:
            painter.setPen(Qt.GlobalColor.black)
            painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, self.name)

    @classmethod
    def is_record_exist(cls, scene, record):
        all_nodes = scene.get_nodes_by_data_context_type(TagRecord)
        for node in all_nodes:
            if node.data_context == record:
                return True
        return False

    def on_add(self, parent_scene):
        parent_scene: GraphScene
        """Add tag node to the scene."""
        all_nodes = parent_scene.get_nodes_by_data_context_type(FileRecord)
        tagRec: TagRecord = self.data_context
        try:
            for file_node in all_nodes:
                if tagRec.is_file_tagged(file_node.data_context):
                    edge = TagFileEdge(self, file_node)
                    parent_scene.addItem(edge)


        except Exception as e:
            loguru.logger.error(f"Error adding node {self.name} to scene: {e}")


class PinNode(Node):
    def __init__(self, x, y, radius, name: str):
        super().__init__(x, y, radius, name)
        self.name = name

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.name:
            painter.setPen(Qt.GlobalColor.black)
            painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, self.name)

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create context menu for pins."""
        menu = QMenu()
        delete_action = QAction("Delete Pin", menu)
        delete_action.triggered.connect(lambda: self.delete())
        menu.addAction(delete_action)

        select_child = QAction("Select child nodes", menu)
        select_child.triggered.connect(lambda: self.select_child_nodes())
        menu.addAction(select_child)

        menu.exec(event.screenPos())

    def select_child_nodes(self):
        """Select all nodes connected to this pin."""
        scene = self.scene()
        if scene is not None:
            for edge in scene.edges:
                if edge.node1 == self or edge.node2 == self:
                    if isinstance(edge.node1, FileRecordNode):
                        edge.node1.setSelected(True)
                    if isinstance(edge.node2, FileRecordNode):
                        edge.node2.setSelected(True)


class FileRecordNode(Node):
    def __init__(self, x, y, radius, file_record: FileRecord):
        super().__init__(x, y, radius, file_record)
        self.file_record = file_record
        self.set_image(file_record.full_path)

    @staticmethod
    def is_record_exist(scene, record):
        all_nodes = scene.get_nodes_by_data_context_type(FileRecord)
        for node in all_nodes:
            if node.file_record == record:
                return True
        return False

    @staticmethod
    def get_node_by_record(scene, record):
        all_nodes = scene.nodes
        for node in all_nodes:
            if node.file_record == record:
                return node
        return None

    def on_add(self, parent_scene):
        parent_scene: GraphScene
        all_nodes = parent_scene.nodes
        try:
            forward_query = {"from_id": self.file_record._id,
                             "to_id": {"$in": [x.file_record._id for x in all_nodes if isinstance(x, FileRecordNode)]},
                             "type": "similar_search"}
            backward_query = {"to_id": self.file_record._id, "from_id": {
                "$in": [x.file_record._id for x in all_nodes if isinstance(x, FileRecordNode)]},
                              "type": "similar_search"}
            forward_rels = RelationRecord.find(forward_query)
            backward_rels = RelationRecord.find(backward_query)
            for relation in forward_rels:
                file_record = FileRecord(relation.to_id)
                target_file_record = parent_scene.get_node_by_data_context(file_record)
                edge = RelationRecordEdge(self, target_file_record, relation)
                parent_scene.addItem(edge)
            for relation in backward_rels:
                file_record = FileRecord(relation.from_id)
                target_file_record = parent_scene.get_node_by_data_context(file_record)
                edge = RelationRecordEdge(target_file_record, self, relation)
                parent_scene.addItem(edge)
            # set tags relations
            tag_nodes_all = parent_scene.get_nodes_by_data_context_type(TagRecord)
            for node in tag_nodes_all:
                tag_record: TagRecord = node.data_context
                if tag_record.is_file_tagged(self.file_record):
                    edge = TagFileEdge(node, self)
                    parent_scene.addItem(edge)
        except Exception as e:
            loguru.logger.error(f"Error adding node {self.file_record.name} to scene: {e}")

    def on_delete(self, parent_scene):
        all_edges = parent_scene.edges.copy()  # Create a copy to avoid modification during iteration
        for edge in all_edges:
            if edge.node1 == self or edge.node2 == self:
                edge.delete()

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create context menu for nodes, edges, and groups."""
        menu = QMenu()
        tags_of_file = QAction("Show tags of file", menu)
        tags_of_file.triggered.connect(lambda: self.show_tags())
        menu.addAction(tags_of_file)

        delete_action = QAction("Create group", menu)
        delete_action.triggered.connect(lambda: self.create_group_from_sel())

        menu.addAction(delete_action)

        hide_node = QAction("Hide node", menu)
        hide_node.triggered.connect(lambda: self.hide_node())
        menu.addAction(hide_node)

        create_pin = QAction("Create pin", menu)
        create_pin.triggered.connect(lambda: self.create_pin())
        menu.addAction(create_pin)

        zoom_menu = menu.addMenu("Zoom")
        scale_up_action = QAction("Scale 2.0x", zoom_menu)
        scale_up_action.triggered.connect(lambda: self.scale_nodes(2.0))
        zoom_menu.addAction(scale_up_action)

        scale_down_action = QAction("Scale 0.5x", zoom_menu)
        scale_down_action.triggered.connect(lambda: self.scale_nodes(0.5))
        zoom_menu.addAction(scale_down_action)

        pins_nodes = self.scene().get_nodes_by_data_context_type(str)
        if len(pins_nodes) > 0:
            pin_menu = menu.addMenu("Pins")

            def add_pin_edge(pin):
                edge = RelationRecordEdge(self, pin, None)
                self.scene().addItem(edge)

            for pin_node in pins_nodes:
                pin_action = QAction(pin_node.data_context, pin_menu)
                pin_action.triggered.connect(lambda checked, x=pin_node: add_pin_edge(x))
                pin_menu.addAction(pin_action)

        menu.exec(event.screenPos())

    def show_tags(self):

        """Show tags of file as nodes in scene."""

        place_x = self.x() + 512
        place_y = self.y() + 512
        scene: GraphScene = self.scene()
        if scene is not None:
            selected_file_nodes = scene.get_selected_nodes(FileRecordNode)
            for node in selected_file_nodes:
                tags = TagRecord.get_tags_of_file(node.data_context)
                if tags:
                    for tag in tags:
                        if not TagNode.is_record_exist(scene, tag):
                            tag_node = TagNode(place_x, place_y, 128, tag)
                            scene.addItem(tag_node)  # on add call automatically on_add method of TagNode
                            place_x += 256  # Increment x position for next tag node

            else:
                loguru.logger.info(f"No tags found for file {self.file_record.name}")

    def scale_nodes(self, factor):
        scene = self.scene()
        selected_nodes = scene.get_selected_nodes()

        if not selected_nodes:
            return

        # Calculate the bounding box of the selected nodes
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for node in selected_nodes:
            pos = node.pos()
            min_x = min(min_x, pos.x())
            min_y = min(min_y, pos.y())
            max_x = max(max_x, pos.x())
            max_y = max(max_y, pos.y())

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_point = QPointF(center_x, center_y)

        for node in selected_nodes:
            old_pos = node.pos()
            relative_pos = old_pos - center_point
            new_relative_pos = relative_pos * factor
            new_pos = center_point + new_relative_pos
            node.setPos(new_pos)

    def hide_node(self):
        app = Allocator.get_instance(Edit_graph_app)
        app.hidden_records.append(self.file_record._id)
        # remove self frome scene
        self.scene().removeItem(self)

    def create_group_from_sel(self):
        scene = self.scene()
        selected_nodes = scene.get_selected_nodes()
        if len(selected_nodes) > 1:
            group = NodeGroup(selected_nodes)
            scene.addItem(group)
            scene.groups.append(group)

    def create_pin(self):
        scene = self.scene()
        # Запрашиваем имя для пина у пользователя
        pin_name, ok = QInputDialog.getText(scene.views()[0],
                                            "Create Pin",
                                            "Enter pin name:",
                                            QLineEdit.Normal,
                                            self.file_record.name)

        # Если пользователь нажал "OK" и ввел непустое имя
        if ok and pin_name:
            pin = PinNode(self.x() + 512, self.y(), 128, pin_name)
            scene.addItem(pin)
            all_pins = scene.get_nodes_by_data_context_type(str)
            legend_label_str = ""
            counter = 0
            for pin_node in all_pins:
                counter += 1
                legend_label_str += "press" + str(counter) + pin_node.name + "\n"
            scene.views()[0].legend_label.setText(legend_label_str)
            scene.views()[0].legend_label.adjustSize()


class RelationRecordEdge(Edge):
    def __init__(self, node1, node2, relation_record):
        super().__init__(node1, node2)
        self.relation_record: RelationRecord = relation_record

    @staticmethod
    def is_relation_exist(scene, relation):
        all_edges = scene.edges
        for edge in all_edges:
            if edge.relation_record == relation:
                return True
        return False

    def on_add(self, parent_scene):
        try:
            sub_type = self.relation_record.get_field_val("sub_type")
            self.colorize(sub_type)
        except Exception as e:
            loguru.logger.error(f"Error setting color for edge {self}: {e}")
            self.set_color(Qt.GlobalColor.black)

    def colorize(self, subtype):
        if subtype == ImageSearchRelSubType.wrong.value:
            self.set_color(Qt.GlobalColor.red)
        elif subtype == ImageSearchRelSubType.none.value:
            self.set_color(Qt.GlobalColor.yellow)
        elif subtype == ImageSearchRelSubType.similar_style.value:
            self.set_color(Qt.GlobalColor.blue)
        elif subtype == ImageSearchRelSubType.similar.value:
            self.set_color(Qt.GlobalColor.green)
        elif subtype == ImageSearchRelSubType.manual.value:
            self.set_color(Qt.GlobalColor.darkGreen)
        else:
            self.set_color(Qt.GlobalColor.black)

    def delete_from_db(self):
        self.scene().removeItem(self)
        rel_record = self.relation_record
        if rel_record is not None:
            # remove from db
            rel_record.delete_rec()
            # remove from scene

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create a context menu for nodes, edges, and groups."""
        menu = QMenu()
        delete_action = QAction("Delete Edge", menu)
        delete_action.triggered.connect(lambda: self.delete_from_db())
        menu.addAction(delete_action)

        open_in_exp = QAction("Open in explorer", menu)
        open_in_exp.triggered.connect(lambda: self.file_record.open_in_explorer())
        menu.addAction(open_in_exp)
        sub_menu = menu.addMenu("set edge type")
        for i in ImageSearchRelSubType:
            sub_menu.addAction(i.value, lambda x=i: self.set_rel_sub_type(x))

        menu.exec(event.screenPos())

    def open_in_explorer(self):
        path = self.file_record.full_path
        if path is not None:
            os.system(f'explorer /select,"{path}"')

    def set_rel_sub_type(self, subtype):
        selected_edges = self.scene().get_selected_edges()
        for edge in selected_edges:
            if isinstance(edge, RelationRecordEdge):
                edge.set_sub_type(subtype)

    def set_sub_type(self, subtype):
        """Set the subtype of the relation and update the edge color."""
        self.relation_record.set_field_val("sub_type", subtype.value)
        self.colorize(subtype.value)

    def mousePressEvent(self, event):
        pass

    @classmethod
    def get_relation_edge(cls, scene, relation):
        all_edges = scene.edges
        for edge in all_edges:
            if edge.relation_record == relation:
                return edge
        return None

    def delete(self):
        self.scene().removeItem(self)


class TagFileEdge(Edge):
    def __init__(self, TagNode, Filenode):
        super().__init__(TagNode, Filenode)
        self.TagRecord: TagRecord = TagNode.data_context
        self.FileRecord: FileRecord = Filenode.data_context

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create a context menu for nodes, edges, and groups."""
        menu = QMenu()
        delete_action = QAction("Delete Edge from db", menu)

        menu.exec(event.screenPos())

    def deleteEdge_from_db(self):
        self.scene().removeItem(self)
        tagRec: TagRecord = self.TagRecord
        fileRec: FileRecord = self.FileRecord
        if tagRec is not None and fileRec is not None:
            TagRecord.remove_from_file_rec(fileRec)

    def delete(self):
        self.scene().removeItem(self)


class Edit_graph_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        Allocator.res.register(self)
        self.hidden_records = []
        self.main_view = MainView()
        menu_bar = self._main_window.menuBar()
        menu = menu_bar.addMenu("File")
        WidgetBuilder.add_menu_item(menu, "Save Graph", lambda: self.save_graph())
        WidgetBuilder.add_menu_item(menu, "Load Graph", lambda: self.load_graph())
        menu.addSeparator()
        WidgetBuilder.add_menu_item(menu, "load folder", lambda x: self.load_folder())
        WidgetBuilder.add_menu_item(menu, "Load folder recursive", lambda x: self.load_folder_recursive())
        WidgetBuilder.add_menu_item(menu, "Load nodes with references (current folder)", 
                                    lambda x: self.load_nodes_with_references(recursive=False))
        WidgetBuilder.add_menu_item(menu, "Load nodes with references (recursive)", 
                                    lambda x: self.load_nodes_with_references(recursive=True))
        WidgetBuilder.add_menu_item(menu, "Create ref on selected",
                                    lambda x: self.create_ref_on_selected())
        WidgetBuilder.add_menu_item(menu, "Move grouped to folder", lambda x: self.move_grouped_to_folder())
        WidgetBuilder.add_menu_item(menu, "Pull by relation", lambda x: self.pull_by_relation())

        WidgetBuilder.add_menu_item(menu, "set tag", lambda x: self.set_tag())

        menu = menu_bar.addMenu("Scene")
        WidgetBuilder.add_menu_item(menu, "Arrange by folder", lambda x: self.arrange_by_folder())
        WidgetBuilder.add_menu_item(menu, "place related near", lambda x: self.place_similar_k_nodes_near)
        WidgetBuilder.add_menu_item(menu, "clear wrong relations", lambda x: self.main_view.clear_wrong_relations())
        WidgetBuilder.add_menu_item(menu, "clear scene", lambda x: self.main_view.scene.clear_scene())

    def save_graph(self):
        file_path, _ = QFileDialog.getSaveFileName(self.main_view, "Save Graph", "", "JSON Files (*.json)")
        if not file_path:
            return

        scene = self.main_view.scene
        graph_data = {"nodes": [], "edges": []}

        for node in scene.nodes:
            node_data = {
                "id": str(node.file_record._id) if isinstance(node, FileRecordNode) else node.name,
                "type": "FileRecordNode" if isinstance(node, FileRecordNode) else "PinNode",
                "pos": {"x": node.x(), "y": node.y()},
            }
            if isinstance(node, PinNode):
                node_data["name"] = node.name
            graph_data["nodes"].append(node_data)

        for edge in scene.edges:
            if isinstance(edge, RelationRecordEdge) and edge.relation_record:
                edge_data = {
                    "from": str(edge.node1.file_record._id) if isinstance(edge.node1,
                                                                          FileRecordNode) else edge.node1.name,
                    "to": str(edge.node2.file_record._id) if isinstance(edge.node2,
                                                                        FileRecordNode) else edge.node2.name,
                    "relation_id": str(edge.relation_record._id)
                }
                graph_data["edges"].append(edge_data)
            else:
                edge_data = {"from": str(edge.node1.file_record._id) if isinstance(edge.node1,
                                                                                   FileRecordNode) else edge.node1.name,
                             "to": str(edge.node2.file_record._id) if isinstance(edge.node2,
                                                                                 FileRecordNode) else edge.node2.name,
                             "relation_id": None}
                graph_data["edges"].append(edge_data)

        with open(file_path, "w") as f:
            json.dump(graph_data, f, indent=4)

    def load_graph(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_view, "Load Graph", "", "JSON Files (*.json)")
        if not file_path:
            return

        with open(file_path, "r") as f:
            graph_data = json.load(f)

        scene = self.main_view.scene
        scene.clear_scene()

        nodes_map = {}
        for node_data in graph_data["nodes"]:
            pos = QPointF(node_data["pos"]["x"], node_data["pos"]["y"])
            if node_data["type"] == "FileRecordNode":
                file_record = FileRecord.find_one({"_id": ObjectId(node_data["id"])})
                if file_record:
                    node = FileRecordNode(pos.x(), pos.y(), 128, file_record)
                    scene.addItem(node)
                    nodes_map[node_data["id"]] = node
            elif node_data["type"] == "PinNode":
                node = PinNode(pos.x(), pos.y(), 128, node_data["name"])
                scene.addItem(node)
                nodes_map[node_data["name"]] = node

        for edge_data in graph_data["edges"]:
            # no need load relation record relations they load automatically on add event ?
            # bath if edge_data["relation_id"] is none need restore relation edge
            from_node = nodes_map.get(edge_data["from"])
            to_node = nodes_map.get(edge_data["to"])
            if edge_data["relation_id"]:
                continue
            else:
                relation = None
            if from_node and to_node:
                edge = RelationRecordEdge(from_node, to_node, relation)
                scene.addItem(edge)

        all_pins = scene.get_nodes_by_data_context_type(str)
        legend_label_str = ""
        counter = 0
        for pin_node in all_pins:
            counter += 1
            legend_label_str += "press" + str(counter) + pin_node.name + "\n"
        scene.views()[0].legend_label.setText(legend_label_str)
        scene.views()[0].legend_label.adjustSize()

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self.main_view, "Select Folder")
        add_node_list = []
        pos_y = 0
        scene = self.main_view.scene
        if folder_path is not None:
            for file in tqdm(os.listdir(folder_path)):
                pos_y += 512
                file_path = os.path.join(folder_path, file)
                file_record = FileRecord.get_record_by_path(file_path)
                if file_record is not None:
                    node = scene.get_node_by_data_context(file_record)
                    if node is None:
                        node = FileRecordNode(0, pos_y, 128, file_record)
                    node.setPos(0, pos_y)
                    add_node_list.append(node)
        scene.addItems(add_node_list)

    def load_folder_recursive(self):
        """Load all files from folder and all subfolders."""
        folder_path = QFileDialog.getExistingDirectory(self.main_view, "Select Folder")
        if not folder_path:
            return
        
        add_node_list = []
        pos_y = 0
        scene = self.main_view.scene
        
        # Use get_file_record_by_folder with recurse=True
        file_records = get_file_record_by_folder(folder_path, recurse=True)
        
        for file_record in tqdm(file_records, desc="Loading files recursively"):
            pos_y += 512
            node = scene.get_node_by_data_context(file_record)
            if node is None:
                node = FileRecordNode(0, pos_y, 128, file_record)
            node.setPos(0, pos_y)
            add_node_list.append(node)
        
        scene.addItems(add_node_list)

    def load_nodes_with_references(self, recursive=False):
        """Load only files that have relations from selected folder."""
        folder_path = QFileDialog.getExistingDirectory(self.main_view, "Select Folder")
        if not folder_path:
            return
        
        add_node_list = []
        pos_y = 0
        scene = self.main_view.scene
        
        # Get all file records from folder
        file_records = get_file_record_by_folder(folder_path, recurse=recursive)
        
        for file_record in tqdm(file_records, desc="Loading nodes with references"):
            # Check if file has any relations
            has_relations = RelationRecord.find_one({
                "$or": [
                    {"from_id": file_record._id},
                    {"to_id": file_record._id}
                ]
            })
            
            if has_relations:
                pos_y += 512
                node = scene.get_node_by_data_context(file_record)
                if node is None:
                    node = FileRecordNode(0, pos_y, 128, file_record)
                node.setPos(0, pos_y)
                add_node_list.append(node)
        
        scene.addItems(add_node_list)

    def set_tag(self):
        pass

    def move_grouped_to_folder(self):
        scene = self.main_view.scene
        selected_groups = scene.get_selected_groups()
        main_view = Allocator.get_instance(MainView)
        folder_path = QFileDialog.getExistingDirectory(main_view, "Select Folder")
        for group in selected_groups:
            nodes = [node for node in group.nodes if isinstance(node, FileRecordNode)]
            for node in nodes:
                file_record: FileRecord = node.file_record
                file_record.move_to_folder(folder_path)

    def pull_by_relation(self):
        all_nodes = self.main_view.scene.get_nodes_by_data_context_type(FileRecord)
        all_file_records = [node.file_record for node in all_nodes if isinstance(node, FileRecordNode)]
        all_relations = RelationRecord.find({"$or": [{"from_id": {"$in": [x._id for x in all_file_records]}},
                                                     {"to_id": {"$in": [x._id for x in all_file_records]}}]})
        start_x = 0
        for relation in tqdm(all_relations):
            sub_type = relation.get_field_val("sub_type")
            if sub_type == ImageSearchRelSubType.wrong.value:
                continue
            #if sub_type == ImageSearchRelSubType.wrong.none:
            #   continue
            from_id = relation.get_field_val("from_id")
            to_id = relation.get_field_val("to_id")
            if from_id in self.hidden_records:
                continue
            if to_id in self.hidden_records:
                continue
            from_node = None
            to_node = None
            for node in all_nodes:
                if node.file_record._id == from_id:
                    from_node = node
                if node.file_record._id == to_id:
                    to_node = node
            if from_node is None:
                f_record = FileRecord.find_one({"_id": from_id})
                if f_record is None:
                    relation.delete_rec()
                    continue
                from_node = FileRecordNode(0, 0, 128, f_record)
                # set position
                from_node.setPos(start_x, 0)
                start_x += 512
                self.main_view.scene.addItem(from_node)

            if to_node is None:
                f_record = FileRecord.find_one({"_id": to_id})
                if f_record is None:
                    relation.delete_rec()
                    continue
                to_node = FileRecordNode(0, 0, 128, FileRecord.find_one({"_id": to_id}))
                # set position
                to_node.setPos(start_x, 512)
                start_x += 512
                self.main_view.scene.addItem(to_node)

    def arrange_by_folder(self):
        all_nodes = self.main_view.scene.nodes
        all_folders = {}
        init_x = 0
        init_y = 0
        for node in all_nodes:
            file_record: FileRecord = node.file_record
            folder_path = file_record.local_path
            if folder_path is not None:
                folder_path = folder_path.replace("\\", "/")
                folder_files = all_folders.get(folder_path, [])
                folder_files.append(node)
                all_folders[folder_path] = folder_files

        for folder_path in tqdm(all_folders.keys()):
            folder_files = all_folders[folder_path]
            for node in folder_files:
                node.setPos(init_x, init_y)
                init_x += 512
            init_x = 0
            init_y += 512
        self.main_view.scene.recalc_edges_position()

    def create_ref_on_selected(self):
        selected_nodes = self.main_view.scene.get_selected_nodes()
        if len(selected_nodes) < 2:
            return
        for i in range(len(selected_nodes) - 1):
            node1 = selected_nodes[i]
            node2 = selected_nodes[i + 1]
            if isinstance(node1, FileRecordNode) and isinstance(node2, FileRecordNode):
                relation = RelationRecord.get_or_create(from_=node1.file_record, to_=node2.file_record,
                                                        type="similar_search")
                relation.set_field_val("sub_type", "manual")
                # Check if the relation already exists in the scene
                if relation is not None and not RelationRecordEdge.is_relation_exist(self.main_view.scene, relation):
                    edge = RelationRecordEdge(node1, node2, relation)
                    self.main_view.scene.addItem(edge)

    def place_similar_k_nodes_near(self):
        #todo revrite this
        """Place similar nodes near the first selected node."""
        selected_nodes = self.main_view.scene.get_selected_nodes()
        first_node = selected_nodes[0] if selected_nodes else None
        if first_node is None:
            return
        init_x = first_node.x()
        init_y = first_node.y() + 512
        get_linked_records = RelationRecord.find({"from_id": first_node.file_record._id,
                                                  "type": "similar_search"})
        linked_nodes = []
        for rel in get_linked_records:
            to_id = rel.get_field_val("to_id")
            file_record = FileRecord.find_one({"_id": to_id})
            if file_record is not None:
                node = self.main_view.scene.get_node_by_data_context(file_record)
                if node is not None:
                    linked_nodes.append(node)
        for node in linked_nodes:
            if isinstance(node, FileRecordNode):
                node.setPos(init_x, init_y)
                init_x += 512


class MainView(PySide6GlueWidget):

    def __init__(self):
        self.emb_cache = Embeddings_cache([CNN_Encoder_FaceNet.format, CNN_Encoder_BLIP.format, CNN_Encoder_DINO.format,
                                           CNN_Encoder_InceptionResNetV2.format, CNN_Encoder_InceptionV3.format,
                                           CNN_encoder_ModileNetv3_Small.format, CNN_Encoder_ResNet50.format,
                                           CNN_Encoder_CLIP_DML.format, CNN_Encoder_custom.format,
                                           CNN_Encoder_mv2_custom.format])
        self.clip_2d_reducer = VectorReducer(target_dim=2, method="umap")
        self.clip_2d_reducer.load(r"G:\Мой диск\models\reduction2dim")
        self.reducer512 = VectorReducer(target_dim=512, method="umap")
        self.reducer512.load(r"G:\Мой диск\models\reduction512dim")

        self.rel_filter = ImageSearchRelFilter()
        self.vector_fuse = EmbeddingFusion(self.clip_2d_reducer)
        super().__init__()
        Allocator.res.register(self)
        self.scene = GraphScene()
        self.scene.all_arrange_algorithms["Arrange by clip"] = self.arrange_graph_by_clip
        self.scene.all_arrange_algorithms["Arrange by dino"] = self.arrange_graph_by_dino
        self.scene.all_arrange_algorithms["Arrange by mobilenet"] = self.arrange_graph_by_mobilenet
        self.scene.all_arrange_algorithms["Arrange by custom"] = self.arrange_graph_by_custom
        self.scene.all_arrange_algorithms["Arrange by combined"] = self.arrange_graph_by_combined
        self.scene.all_arrange_algorithms["Arrange by combined2"] = self.arrange_graph_by_combined2
        self.scene.mode_context.modes["Select"] = BaseMode()
        self.scene.mode_context.modes["Add Edge"] = AddEdgeMode()
        self.scene.mode_context.set_mode("Select")
        self.view = GraphView(self.scene)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

        # Create toolbar layout
        self.toolbar_layout = QHBoxLayout()

        # Mode selection combo
        self.mode_combo = QComboBox()

        self.mode_combo.addItems(self.scene.mode_context.modes.keys())
        self.mode_combo.currentTextChanged.connect(self.scene.mode_context.set_mode)

        self.toolbar_layout.addWidget(self.mode_combo)

        self.arrange_combo = QComboBox()
        self.arrange_combo.addItems(self.scene.all_arrange_algorithms.keys())
        self.arrange_combo.currentTextChanged.connect(lambda x: self.scene.set_arrange_algorithm(x))
        self.toolbar_layout.addWidget(self.arrange_combo)

        # Arrange button
        self.arrange_button = QPushButton("Arrange")
        self.arrange_button.clicked.connect(lambda: self.scene.arrange_graph())
        self.toolbar_layout.addWidget(self.arrange_button)

        # Zoom in/out buttons
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.view.zoom_in)
        self.toolbar_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.view.zoom_out)
        self.toolbar_layout.addWidget(self.zoom_out_button)

        self.fit_in_view_all = QPushButton("Fit in view")
        self.fit_in_view_all.clicked.connect(lambda: self.view.FitInViewAll())
        self.toolbar_layout.addWidget(self.fit_in_view_all)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.toolbar_layout)
        main_layout.addWidget(self.view)
        self.setLayout(main_layout)

    def arrange_graph_by_clip(self):
        node_list = self.scene.nodes

        scene_multipler = math.sqrt(len(node_list)) * 256
        for node in tqdm(node_list):
            if isinstance(node, FileRecordNode):
                clip_vector = self.emb_cache.get_by_path(node.file_record.full_path, CNN_Encoder_CLIP_DML.format)
                if clip_vector is not None:
                    pos = self.clip_2d_reducer.transform("clip", clip_vector).tolist()
                    pos = QPointF(pos[0] * scene_multipler, pos[1] * scene_multipler)
                    node.setPos(pos)
                    #update edges position
                    for edge in self.scene.edges:
                        edge.update_position()

    def arrange_graph_by_dino(self):
        node_list = self.scene.nodes

        scene_multipler = math.sqrt(len(node_list)) * 256
        for node in tqdm(node_list):
            if isinstance(node, FileRecordNode):
                dino_vector = self.emb_cache.get_by_path(node.file_record.full_path, CNN_Encoder_DINO.format)
                if dino_vector is not None:
                    pos = self.clip_2d_reducer.transform("dino", dino_vector).tolist()
                    pos = QPointF(pos[0] * scene_multipler, pos[1] * scene_multipler)
                    node.setPos(pos)
                    # update edges position
                    for edge in self.scene.edges:
                        edge.update_position()

    def arrange_graph_by_mobilenet(self):
        node_list = self.scene.nodes

        scene_multipler = math.sqrt(len(node_list)) * 256
        for node in tqdm(node_list):
            if isinstance(node, FileRecordNode):
                mobilenet_vector = self.emb_cache.get_by_path(node.file_record.full_path,
                                                              CNN_encoder_ModileNetv3_Small.format)
                if mobilenet_vector is not None:
                    pos = self.clip_2d_reducer.transform("mobilenet", mobilenet_vector).tolist()
                    pos = QPointF(pos[0] * scene_multipler, pos[1] * scene_multipler)
                    node.setPos(pos)
                    # update edges position
                    for edge in self.scene.edges:
                        edge.update_position()

    def arrange_graph_by_custom(self):
        node_list = self.scene.nodes

        scene_multipler = math.sqrt(len(node_list)) * 256
        for node in tqdm(node_list):
            if isinstance(node, FileRecordNode):
                custom_vector = self.emb_cache.get_by_path(node.file_record.full_path,
                                                              CNN_Encoder_custom.format)
                if custom_vector is not None:
                    pos = self.clip_2d_reducer.transform("custom", custom_vector).tolist()
                    pos = QPointF(pos[0] * scene_multipler, pos[1] * scene_multipler)
                    node.setPos(pos)
                    # update edges position
                    for edge in self.scene.edges:
                        edge.update_position()

    def arrange_graph_by_combined(self):
        node_list = self.scene.nodes

        scene_multiple = math.sqrt(len(node_list)) * 1024 * 8
        for node in tqdm(node_list):
            if isinstance(node, FileRecordNode):
                clip_vector = self.emb_cache.get_by_path(node.file_record.full_path, CNN_Encoder_CLIP_DML.format)
                dino_vector = self.emb_cache.get_by_path(node.file_record.full_path, CNN_Encoder_DINO.format)
                mobilenet_vector = self.emb_cache.get_by_path(node.file_record.full_path,
                                                              CNN_encoder_ModileNetv3_Small.format)
                if clip_vector is not None and dino_vector is not None and mobilenet_vector is not None:
                    self.vector_fuse.add_embedding("clip", clip_vector, 0.7)
                    self.vector_fuse.add_embedding("dino", dino_vector, 0.2)
                    self.vector_fuse.add_embedding("mobilenet", mobilenet_vector, 0.1)
                    pos = self.vector_fuse.fuse().tolist()
                    self.vector_fuse.clear()
                    pos = QPointF(pos[0] * scene_multiple, pos[1] * scene_multiple)
                    node.setPos(pos)
                    # update edges position
                    for edge in node.edges:
                        edge.update_position()

    def arrange_graph_by_combined2(self):
        node_list = self.scene.nodes

        scene_multiple = math.sqrt(len(node_list)) * 256
        fuse512 = EmbeddingFusion(self.reducer512)
        for node in tqdm(node_list):
            if isinstance(node, FileRecordNode):
                clip_vector = self.emb_cache.get_by_path(node.file_record.full_path, CNN_Encoder_CLIP_DML.format)
                dino_vector = self.emb_cache.get_by_path(node.file_record.full_path, CNN_Encoder_DINO.format)
                mobilenet_vector = self.emb_cache.get_by_path(node.file_record.full_path,
                                                              CNN_encoder_ModileNetv3_Small.format)
                if clip_vector is not None and dino_vector is not None and mobilenet_vector is not None:
                    fuse512.add_embedding("clip", clip_vector, 0.7)
                    fuse512.add_embedding("dino", dino_vector, 0.2)
                    fuse512.add_embedding("mobilenet", mobilenet_vector, 0.1)
                    comb_vector = fuse512.fuse()
                    fuse512.clear()
                    pos = self.clip_2d_reducer.transform("clip_dino_mobilenet", comb_vector).tolist()
                    pos = QPointF(pos[0] * scene_multiple, pos[1] * scene_multiple)
                    node.setPos(pos)
                    # update edges position
                    for edge in node.edges:
                        edge.update_position()

    def clear_wrong_relations(self):
        all_edges = self.scene.edges.copy()
        for edge in all_edges:
            if isinstance(edge, RelationRecordEdge) and edge.relation_record is not None:
                if edge.relation_record.get_field_val("sub_type") == ImageSearchRelSubType.wrong.value:
                    edge.delete()
        self.scene.recalc_edges_position()


if __name__ == "__main__":
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"

    QtApp = Edit_graph_app()
    QtApp.set_main_widget(QtApp.main_view)

    QtApp.run()
