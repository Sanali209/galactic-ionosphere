from PySide6.QtCore import Qt, QStringListModel, Signal
from PySide6.QtWidgets import QWidget, QCompleter, QVBoxLayout, QLabel, QLineEdit, QGridLayout, QHBoxLayout, QComboBox, \
    QPushButton

from SLM.pySide6Ext.pySide6Q import FlowLayout


class TagBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.edittags = []
        self.AllTags = []
        self.columns = 4
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.initUI()


    tag_changed_event = Signal(str, object)
    ''':arg 
            str: change type - 'add', 'remove'
            object: tag name
            '''

    def initUI(self):
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.tagLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.tagLayout)
        self.tagLabel = QLabel("Tags")
        self.tagLayout.addWidget(self.tagLabel)
        self.tagEdit = QLineEdit()
        self.tagEdit.setCompleter(self.completer)
        self.tagEdit.setPlaceholderText("Add Tag")
        self.tagLayout.addWidget(self.tagEdit)
        self.tagEdit.returnPressed.connect(self.tagEditReturnPressed)

        self.tagListLayout = FlowLayout()
        # set spacing 1 pixel
        self.tagListLayout.setSpacing(1)
        # set margin 1 pixel
        self.tagListLayout.setContentsMargins(1, 1, 1, 1)

        self.mainLayout.addLayout(self.tagListLayout)

    def set_completer(self, taglist: list):
        """
        Set the completer for the tag input field.
        :param taglist: List of tags to be used in the completer.
        """
        self.AllTags = taglist
        self.completer = QCompleter(self.AllTags)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.tagEdit.setCompleter(self.completer)
        self.UpdateUI()

    def UpdateUI(self):

        # clear tag layout
        for i in reversed(range(self.tagListLayout.count())):
            self.tagListLayout.itemAt(i).widget().setParent(None)

        for i in range(len(self.edittags)):
            tag = self.edittags[i]
            tagwidget = TagItemWidget(tag, self)
            self.tagListLayout.addWidget(tagwidget)

    def addTag(self, tag):
        self.edittags.append(tag)

        self.UpdateUI()
        self.tag_changed_event.emit('add', tag)

    def removeTag(self, tag):
        self.edittags.remove(tag)
        self.UpdateUI()
        self.tag_changed_event.emit('remove', tag)

    def tagEditReturnPressed(self):
        text = self.tagEdit.completer().currentCompletion()
        if self.tagEdit.completer().popup().isVisible():
            self.tagEdit.completer().popup().hide()
            return
        if self.tagEdit.text() == "":
            return

        tag = self.tagEdit.text()
        self.addTag(tag)
        self.tagEdit.setText("")


class TagItemWidget(QWidget):
    def __init__(self, tagname="", parent=None):
        super().__init__(parent)
        self.tagname = tagname
        self.initUI()

    # noinspection PyUnresolvedReferences
    def initUI(self):
        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)
        # set spacing 1 pixel
        self.mainLayout.setSpacing(1)
        # set margin 1 pixel
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.tagnameLabel = QLabel(self.tagname)
        self.mainLayout.addWidget(self.tagnameLabel)
        self.delButton = QPushButton("X")
        # set button width to 20 pixel
        self.delButton.setMaximumWidth(20)
        self.mainLayout.addWidget(self.delButton)
        self.delButton.clicked.connect(self.delButtonClicked)

    def delButtonClicked(self):
        self.parent().removeTag(self.tagname)
        self.deleteLater()


class TagInputh(QWidget):
    def __init__(self, parent=None):
        super(TagInputh, self).__init__(parent)
        self.tags = ['test', 'test2', 'test3']
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.tag_input = QLineEdit()
        self.layout.addWidget(self.tag_input)
        self.tag_input.setPlaceholderText("Enter tags separated by comma")
        self.tag_input.returnPressed.connect(self.updateComboBox)
        #self.tag_input.textChanged.connect(self.updateComboBox)
        self.comboLayout = QHBoxLayout()
        self.layout.addLayout(self.comboLayout)
        self.tag_combobox = QComboBox()
        self.comboLayout.addWidget(self.tag_combobox)
        self.add_button = QPushButton("Add Tag")
        self.comboLayout.addWidget(self.add_button)
        self.add_button.clicked.connect(self.addSelectedTag)
        self.model = QStringListModel()
        self.completer = QCompleter(self.model)
        self.tag_combobox.setModel(self.model)
        self.tag_combobox.setCompleter(self.completer)
        self.model.setStringList(self.tags)

    def updateComboBox(self):
        text = self.tag_input.text()
        newtags = set(tag.strip() for tag in text.split(',') if tag.strip())
        self.tags = list(set(self.tags + list(newtags)))
        self.model.setStringList(self.tags)

    def addSelectedTag(self):
        selected_tag = self.tag_combobox.currentText()
        current_text = self.tag_input.text()
        if current_text:
            current_text += ', '
        current_text += selected_tag
        self.tag_input.setText(current_text)

    def set_tags(self, tags):
        self.tags = tags
        self.model.setStringList(self.tags)

    def set_text(self, text):
        self.tag_input.setText(text)
