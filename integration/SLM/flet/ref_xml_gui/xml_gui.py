from SLM.flet.ref_xml_gui.controls import update_parsers
from SLM.parsing.xml_gui.xml_parse import XMLParser


class XMLFletParser(XMLParser):
    element_parser = {}

    def __init__(self, xml_str: str, flet_obj):
        super().__init__(xml_str)
        self.associated_object = flet_obj

    def add_control(self, control):
        from SLM.flet.flet_ext import ftUserControl
        if issubclass(self.associated_object.__class__, ftUserControl):
            self.associated_object.stack.controls.append(control)
        self.associated_object.controls.append(control)


update_parsers()
