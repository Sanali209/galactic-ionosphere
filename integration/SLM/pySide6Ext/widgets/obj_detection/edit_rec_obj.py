from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QCompleter, QLabel, QPushButton, QMessageBox
from SLM.pySide6Ext.widgets.tools import WidgetBuilder as wb

from SLM.files_db.object_recognition.object_recognition import DetectionObjectClass, Recognized_object
from SLM.pySide6Ext.dialogsEx import UniDialog

class CustomCompleter(QCompleter):
    def __init__(self, words, parent=None):
        super().__init__(words, parent)
        self.setCompletionMode(QCompleter.PopupCompletion)
        self.setFilterMode(Qt.MatchContains)
        self.setCaseSensitivity(Qt.CaseInsensitive)

class EditRecognizedObjectsDialog(UniDialog):
    """
    Dialog for editing recognized objects
    fields:
    dialog_result: bool - result of dialog
    current_class: DetectionObjectClass - current selected class
    current_recognized_object: Recognized_object - current selected recognized object
    """
    recognized_objects_combo_box: QComboBox = None

    def __init__(self, window_title, parent=None):
        super().__init__(window_title, parent)

        self.current_class = None
        self.current_recognized_object = None

        self.class_combo_box: QComboBox = (wb(QComboBox())
                                           .add_to_layout_with(self.layout, [QLabel("All Classes")])
                                           .build())
        self.class_completer = CustomCompleter(self.class_combo_box.model(), self)
        self.class_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.class_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.class_combo_box.setInsertPolicy(QComboBox.InsertPolicy.InsertAtBottom)
        self.class_combo_box.setCompleter(self.class_completer)
        self.class_combo_box.setEditable(True)

        all_classes = DetectionObjectClass.find({})
        for obj_class in all_classes:
            self.class_combo_box.addItem(obj_class.name)

        self.recognized_objects_combo_box = (
            wb(QComboBox()).add_to_layout_with(self.layout, [QLabel("Recognized Objects")]).build())
        self.recognized_objects_combo_box.setEditable(True)
        self.recognized_completer = CustomCompleter(self.recognized_objects_combo_box.model(), self)
        self.recognized_objects_combo_box.setCompleter(self.recognized_completer)

        self.class_combo_box.lineEdit().editingFinished.connect(self.add_new_class)
        self.class_combo_box.currentIndexChanged.connect(self.on_select_class)
        self.recognized_objects_combo_box.setInsertPolicy(QComboBox.InsertPolicy.InsertAtBottom)
        self.recognized_objects_combo_box.lineEdit().editingFinished.connect(self.add_new_recognized_object)
        self.recognized_objects_combo_box.currentIndexChanged.connect(self.on_select_recognized_object)

        self.class_combo_box.setCurrentIndex(0)
        self.on_select_class(0)

        self.remove_class_button = QPushButton("Remove Class")
        self.remove_class_button.clicked.connect(self.remove_selected_class)
        self.layout.addWidget(self.remove_class_button)

        self.remove_recognized_button = QPushButton("Remove Recognized Object")
        self.remove_recognized_button.clicked.connect(self.remove_selected_recognized_object)
        self.layout.addWidget(self.remove_recognized_button)

        self.ok_button.clicked.connect(self.accept)

    def add_new_class(self):
        new_class_name = self.class_combo_box.currentText()
        existing_classes = [self.class_combo_box.itemText(i) for i in range(self.class_combo_box.count())]

        if new_class_name == "":
            return

        if new_class_name not in existing_classes:
            if self.confirmation_dialog("Add New Class", f"Do you want to add '{new_class_name}'?"):
                DetectionObjectClass.new_record(name=new_class_name)
                self.class_combo_box.addItem(new_class_name)

        # Обновление списка после добавления нового класса
        existing_classes.append(new_class_name)
        self.class_combo_box.setCurrentIndex(existing_classes.index(new_class_name))
        self.on_select_class(self.class_combo_box.currentIndex())

    def on_select_class(self, index):
        class_name = self.class_combo_box.currentText()
        self.current_class = DetectionObjectClass.find_one({"name": class_name})
        recognized_objects = Recognized_object.find({"obj_class_id": self.current_class._id})
        self.recognized_objects_combo_box.clear()
        for obj in recognized_objects:
            self.recognized_objects_combo_box.addItem(obj.name)

    def add_new_recognized_object(self):
        new_recognized_name = self.recognized_objects_combo_box.currentText()

        # Получаем список текущих элементов в комбобоксе
        existing_objects = [self.recognized_objects_combo_box.itemText(i) for i in
                            range(self.recognized_objects_combo_box.count())]

        if new_recognized_name == "":
            return

        # Если нового имени нет в списке, добавляем его
        if new_recognized_name not in existing_objects:
            if self.confirmation_dialog("Add New Recognized Object", f"Do you want to add '{new_recognized_name}'?"):
                self.current_recognized_object = Recognized_object.new_record(name=new_recognized_name,
                                                                              obj_class=self.current_class._id)
                self.recognized_objects_combo_box.addItem(new_recognized_name)
                existing_objects.append(new_recognized_name)  # Обновляем локальный список

        # Устанавливаем индекс элемента, предварительно проверив его наличие
        try:
            index = existing_objects.index(new_recognized_name)
            self.recognized_objects_combo_box.setCurrentIndex(index)
        except ValueError:
            print(f"Error: '{new_recognized_name}' not found in list")

    def on_select_recognized_object(self, index):
        recognized_name = self.recognized_objects_combo_box.itemText(index)
        self.current_recognized_object = Recognized_object.find_one(
            {"obj_class_id": self.current_class._id, "name": recognized_name})
        print(recognized_name)

    def remove_selected_class(self):
        class_name = self.class_combo_box.currentText()
        if self.confirmation_dialog("Remove Class", f"Are you sure you want to remove class '{class_name}'?"):
            DetectionObjectClass.find_one({"name": class_name}).delete_rec()
            self.class_combo_box.removeItem(self.class_combo_box.currentIndex())

    def remove_selected_recognized_object(self):
        recognized_name = self.recognized_objects_combo_box.currentText()
        if self.confirmation_dialog("Remove Recognized Object",
                                    f"Are you sure you want to remove recognized object '{recognized_name}'?"):
            Recognized_object.delete_one({"name": recognized_name, "obj_class_id": self.current_class._id})
            self.recognized_objects_combo_box.removeItem(self.recognized_objects_combo_box.currentIndex())

    def confirmation_dialog(self, title, text):
        reply = QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No)
        return reply == QMessageBox.Yes

    def accept(self):
        self.dialog_result = True
        super().accept()
