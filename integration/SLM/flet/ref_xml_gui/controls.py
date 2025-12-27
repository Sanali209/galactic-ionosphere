from loguru import logger
import flet as ft

from SLM.parsing.xml_gui.xml_parse import ElementParser, AttributeParser, field_copy_attribute

element_dict = {}


def update_parsers():
    from SLM.flet.ref_xml_gui.xml_gui import XMLFletParser
    XMLFletParser.element_parser.update(element_dict)


class ftElementParser(ElementParser):
    def process_associated_object(self):
        try:
            self.parent_element_parser.add_control(self.associated_object)
        except AttributeError:
            logger.warning(f"Cannot add control to parent {self.parent_xml_parser.associated_object}")


class ftLayoutElementParser(ftElementParser):
    def add_control(self, control):
        self.associated_object.controls.append(control)


class ftRow(ftLayoutElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('wrap', bool), AttributeParser('expand', int)])

    def process_associated_object(self):
        row = ft.Row()
        self.associated_object = row
        super().process_associated_object()


element_dict["Row"] = ftRow


class ftColPanelVert(ftLayoutElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('wrap', bool), AttributeParser('expand', int)])

    def process_associated_object(self):
        from SLM.flet.flet_ext import ftCollapsiblePanelVertical
        row = ftCollapsiblePanelVertical()
        self.associated_object = row
        super().process_associated_object()

    def add_control(self, control):
        self.associated_object.set_content(control)


element_dict["ColPanelVert"] = ftColPanelVert


class ftColumn(ftLayoutElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('wrap', bool), AttributeParser('expand', int)])

    def process_associated_object(self):
        column = ft.Column()
        self.associated_object = column
        super().process_associated_object()


element_dict["Column"] = ftColumn


class ftCard(ftLayoutElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)

    def process_associated_object(self):
        self.associated_object = ft.Card()
        # add to parent
        super().process_associated_object()

    def add_control(self, control):
        self.associated_object.content = control


element_dict["Card"] = ftCard


class ftText(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('value', str)])

    def process_associated_object(self):
        text = ft.Text()
        self.associated_object = text
        super().process_associated_object()


element_dict["Text"] = ftText


class ftImage(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('src', str), AttributeParser('width', int),
                                AttributeParser('height', int)])

    def process_associated_object(self):
        self.associated_object = ft.Image()
        super().process_associated_object()


element_dict["Image"] = ftImage


class ftTextField(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([
            AttributeParser('label', str), AttributeParser('value', str),
            AttributeParser("multiline", bool), AttributeParser("min_lines", int),
            AttributeParser("max_lines", int)
        ])

    def process_associated_object(self):
        self.associated_object = ft.TextField()
        super().process_associated_object()


element_dict["TextField"] = ftTextField


class ftElevatedButton(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('text', str),
                                field_copy_attribute('on_click')])

    def process_associated_object(self):
        self.associated_object = ft.ElevatedButton()
        super().process_associated_object()


element_dict["Button"] = ftElevatedButton


class ftSwitch(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('label', str), AttributeParser('value', bool),
                                field_copy_attribute('on_change')])

    def process_associated_object(self):
        self.associated_object = ft.Switch()
        super().process_associated_object()

element_dict["Switch"] = ftSwitch

class ftDropdown(ftElementParser):
    """
    todo: implement posible sub nodes
    """

    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('label', str)])

    def process_associated_object(self):
        self.associated_object = ft.Dropdown()
        super().process_associated_object()


element_dict["DropDown"] = ftDropdown


class ftSlider(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('min', int), AttributeParser('max', int),
                                AttributeParser('division', int), AttributeParser('value', float),
                                AttributeParser('label', str)])

    def process_associated_object(self):
        self.associated_object = ft.Slider()
        super().process_associated_object()


element_dict["Slider"] = ftSlider


class ftGridView(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('expand', int), AttributeParser('runs_count', int),
                                AttributeParser('max_extent', int)])

    def process_associated_object(self):
        self.associated_object = ft.GridView()
        super().process_associated_object()


element_dict["Grid"] = ftGridView


class ftMenuBar(ftLayoutElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        attributes = [
            AttributeParser('expand', int),
        ]
        self.attributes.extend(attributes)

    def process_associated_object(self):
        self.associated_object = ft.MenuBar()
        super().process_associated_object()


element_dict["MenuBar"] = ftMenuBar


class ftSubmenuButton(ftLayoutElementParser):
    """

    """

    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([TextForContentAttribute('text')])

    def process_associated_object(self):
        self.associated_object = ft.SubmenuButton()
        super().process_associated_object()


element_dict["SubmenuButton"] = ftSubmenuButton


class ftMenuItemButton(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([TextForContentAttribute('text'),
                                AttributeParser('tooltip', str),
                                field_copy_attribute('on_click')])

    def process_associated_object(self):
        self.associated_object = ft.MenuItemButton()
        super().process_associated_object()


element_dict["MenuItemButton"] = ftMenuItemButton


class ftProgressBar(ftElementParser):
    def __init__(self, parent_xml_parser, parent_element_parser, xml_node):
        super().__init__(parent_xml_parser, parent_element_parser, xml_node)
        self.attributes.extend([AttributeParser('value', float)])
        self.attributes.extend([AttributeParser('width', int)])

    def process_associated_object(self):
        self.associated_object = ft.ProgressBar()
        super().process_associated_object()


element_dict["ProgressBar"] = ftProgressBar








class TextForContentAttribute(AttributeParser):
    """
    create ft.Text control with attribute value
    and place it in current object .content field
    """

    def __init__(self, name):
        super().__init__(name, str)

    def set_element_attribute(self, parse_element):
        if self.value is None:
            return
        try:
            parse_element.associated_object.content = ft.Text(self.value)
        except Exception:
            raise RuntimeError("cant set attribute")
