from PySide6.QtWidgets import QPushButton, QLineEdit

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget, FlowLayout
from SLM.pySide6Ext.binding import binding_load

binding_load()


class test_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        menu_bar = self._main_window.menuBar()
        menu = menu_bar.addMenu("File")


class binding_class(PropUser):
    test_text: str = PropInfo()

    def __init__(self):
        super().__init__()
        self.test_text = "Hello World"


class testWidget(PySide6GlueWidget):

    def __init__(self):
        self.prop = binding_class()
        super().__init__()

    def define_gui(self):
        test_button = QPushButton("Test")
        test_text_input = QLineEdit()
        self.prop.dispatcher.test_text.bind(bind_target=test_text_input)
        horizontal_layout = FlowLayout()

        self.setLayout(horizontal_layout)
        horizontal_layout.addWidget(test_text_input)
        horizontal_layout.addWidget(test_button)


QtApp = test_app()
QtApp.set_main_widget(testWidget())

QtApp.run()
