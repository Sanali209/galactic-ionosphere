import math

from typing import Callable, Any

import flet as ft
from flet_core import ScrollMode

from SLM.appGlue.DAL.treedata import TreeDataView
from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.appGlue.core import GlueApp, Allocator

from SLM.flet.ref_xml_gui.xml_gui import XMLFletParser


class ftCollapsiblePanelVertical(ft.Column):
    # todo: implement expand srink
    # todo: implement expand value storing
    def __init__(self, label=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = "vertical"

        self.label = "Panel"
        if label is not None:
            self.label = label

        self.header_canvas = ft.canvas.Canvas([
            ft.canvas.Text(-10, -10,
                           text=self.label)])

        self.visible_button = ft.ElevatedButton(text=" ", width=30)
        self.visible_button.on_click = self.on_collapse_click

        self.exp_but = ft.TextButton("+",width=30, on_click=self.on_exp)
        self.shrink_but = ft.TextButton("-",width=30, on_click=self.on_shrink)

        self.expanded = True

        self.header = ft.Card(content=ft.Row(controls=[
            self.visible_button, self.exp_but, self.shrink_but, self.header_canvas
        ]), color=ft.colors.BLUE)

        self.controls.append(self.header)

        self.content = None

        if self.content is not None:
            self.controls.append(self.content)

    def on_exp(self, ev):
        if self.expand is True:
            self.expand = 1
        self.expand = self.expand + 1

        self.update()

    def on_shrink(self, ev):
        if self.expand is True:
            self.expand = 1
        self.expand = self.expand - 1
        self.update()

    def on_collapse_click(self, event):
        if self.expanded:
            # on collapse
            self.expanded = False
            self.expand = False
            if self.mode == "vertical":
                self.header.content = ft.Column(controls=[
                    self.visible_button, self.header_canvas
                ])
                self.header_canvas.shapes.clear()
                self.header_canvas.shapes.append(ft.canvas.Text(0, 0,
                                                                text=self.label, rotate=math.pi / 2,
                                                                alignment=ft.alignment.bottom_left))
                self.header.height = 200
            else: # on mode horizontal
                pass
            if self.content is not None:
                self.controls.remove(self.content)
        else:
            # on expand
            self.expanded = True
            self.expand = True
            self.header.content = ft.Row(controls=[
                self.visible_button, self.exp_but, self.shrink_but, self.header_canvas
            ])
            self.header_canvas.shapes.clear()
            self.header_canvas.shapes.append(ft.canvas.Text(-10, -10,
                                                            text=self.label))
            self.header.height = 40
            if self.content is not None:
                self.controls.append(self.content)
        self.update()

    def set_content(self, content):
        if self.content is not None:
            self.controls.remove(self.content)
        self.content = content
        if self.content is not None:
            self.controls.append(self.content)


class Flet_view(ft.View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # requirements for parser
        self.layout_root = self
        self.xml_source = None
        self.app: FletGlueApp = GlueApp.current_app
        MessageSystem.Subscribe("app_window_resize", self, self.on_app_window_resize)

    def on_keyboard_event(self, event: ft.KeyboardEvent):
        pass

    def on_app_window_resize(self, event):
        if self.app is not None and self.page is not None:
            self.page.update()

    def close_view(self):
        self.on_view_close()
        self.app.view_pop()

    def on_view_close(self):
        MessageSystem.Unsubscribe("app_window_resize", self)

    def parse_xml(self):
        parser = XMLFletParser(self.xml_source, self)
        parser.parse()

    def update(self):
        if self.page is not None:
            self.page.update()


class FletGlueApp(GlueApp):
    def __init__(self, default_view=None):
        super().__init__()
        self.page: ft.Page = None
        self.delayed_views = []
        self.views = {}
        # todo refactor as property field whoth get walue of top view
        self.current_view = None
        self.file_open_dialog: ft.FilePicker = ft.FilePicker()

    def on_keyboard_event(self, event: ft.KeyboardEvent):
        self.current_view.on_keyboard_event(event)

    def init_app_bar(self):
        pass

    def on_resize(self, e):
        MessageSystem.SendMessage("app_window_resize", event=e)

    def set_theme(self,page):
        page.theme_mode = "light"
        theme = ft.Theme()
        theme.visual_density = ft.ThemeVisualDensity.COMPACT
        theme.text_theme = ft.TextTheme()
        theme.text_theme.body_large = ft.TextStyle(size=14,color=ft.colors.BLACK)
        theme.text_theme.body_medium = ft.TextStyle(size=12,color=ft.colors.BLACK)
        theme.text_theme.body_small = ft.TextStyle(size=10,color=ft.colors.BLACK)
        theme.text_theme.display_large = ft.TextStyle(size=14,color=ft.colors.BLACK)
        theme.text_theme.display_medium = ft.TextStyle(size=12,color=ft.colors.BLACK)
        theme.text_theme.display_small = ft.TextStyle(size=10,color=ft.colors.BLACK)
        theme.text_theme.headline_large = ft.TextStyle(size=16,color=ft.colors.BLACK)
        theme.text_theme.headline_medium = ft.TextStyle(size=14,color=ft.colors.BLACK)
        theme.text_theme.headline_small = ft.TextStyle(size=12,color=ft.colors.BLACK)
        theme.text_theme.label_large = ft.TextStyle(size=14,color=ft.colors.BLACK)
        theme.text_theme.label_medium = ft.TextStyle(size=12,color=ft.colors.BLACK)
        theme.text_theme.label_small = ft.TextStyle(size=10,color=ft.colors.BLACK)
        theme.text_theme.title_large = ft.TextStyle(size=14,color=ft.colors.BLACK)
        theme.text_theme.title_medium = ft.TextStyle(size=12,color=ft.colors.BLACK)
        theme.text_theme.title_small = ft.TextStyle(size=10,color=ft.colors.BLACK)

        page.theme = theme
        self.page.update()

    def main(self, page: ft.Page):
        self.page = page
        self.set_theme(page)
        self.page.on_close = self.on_close
        self.page.on_resize = self.on_resize
        self.page.on_keyboard_event = self.on_keyboard_event
        self.page.title = "FletGlueApp"
        # self.page.scroll = ScrollMode.AUTO
        self.page.overlay.append(self.file_open_dialog)
        for view in self.delayed_views:
            self.set_view(view)
        self.page.update()


        MessageSystem.SendMessage("app_window_resize", event=self.page)

    def route_change(self, route):
        pass

    def on_close(self, *args, **kwargs):
        print("closed")
        exit(0)
        pass

    def set_view(self, view):
        if self.page is None:
            self.delayed_views.append(view)
            return
        view.app = self
        self.page.views.append(view)
        self.current_view = view
        MessageSystem.SendMessage("app_window_resize", event=self.page)
        self.page.update()

    def view_pop(self):
        poped_view = self.page.views.pop()
        top_view = self.page.views[-1]
        self.page.update()
        if len(self.page.views) == 1:
            self.close_app()
        return poped_view

    def close_app(self):
        self.page.window_close()

    def run(self):
        super().run()
        ft.app(target=self.main)


class ProgressView(Flet_view):
    def __init__(self):
        super().__init__()
        self.route = "/progress"
        self.progress_max = 100
        self.current_progress = 0
        self.progress_float = 0.0
        self._progress_message = "progress"
        self.AppBar = ft.AppBar(title=ft.Text(self._progress_message))
        self.progress_bar = ft.ProgressBar(value=None)
        self.controls.append(self.AppBar)
        self.controls.append(self.progress_bar)

    @staticmethod
    def show_progress_view() -> 'ProgressView':
        app: FletGlueApp = Allocator.get_instance(GlueApp)
        view = ProgressView()
        app.set_view(view)
        return view

    @staticmethod
    def hide_progress_view():
        app: FletGlueApp = Allocator.get_instance(GlueApp)
        app.view_pop()

    def count_max(self, _list: list):
        self.progress_max = len(_list)
        self.current_progress = 0
        self.progress_float = 0.0
        self.progress_bar.value = None
        self.update()

    def update_title(self, title):
        self._progress_message = title
        self.AppBar.title = ft.Text(self._progress_message)
        self.update()

    def undetermined_progress(self):
        self.progress_bar.value = None
        self.update()

    def step(self):
        self.current_progress += 1
        self.progress_float = self.current_progress / self.progress_max
        self.progress_bar.value = self.progress_float
        self.update()


class flet_dialog_alert(ft.AlertDialog):
    """
    class for quick instantiate dialog
    todo: realize dialog modal
    """

    def __init__(self, title: str, content: str, modal=False):
        super().__init__(title=ft.Text(title), content=ft.Text(content))
        self.on_dismiss = None
        self.modal = modal

    def set_content(self, content):
        self.content = content

    def set_actions(self, action_list):
        self.actions = action_list

    def show(self):
        app: FletGlueApp = GlueApp.current_app
        app.page.dialog = self
        self.open = True
        app.page.update()

    def close_dlg(self):
        self.open = False
        app: FletGlueApp = GlueApp.current_app
        # app.page.dialog = None
        app.page.update()




class ft_TreeView_node(ft.UserControl):
    def __init__(self):
        super().__init__()

        self.expanded = False
        self._header = ft.Column()
        self._sub_items_container = ft.Column()
        self.data = None




class ftUserControl(ft.UserControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expand = True
        self.stack = ft.Stack(expand=True)
        self.xml_source = None

    def parse_xml(self):
        if self.xml_source is None:
            return
        parser = XMLFletParser(self.xml_source, self)
        parser.parse()
        self.update()

    def build(self):
        return self.stack

    def update(self):
        if self.page is not None:
            super().update()


class ftUserColumn(ft.Column):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xml_source = None
        # todo implement suspend _update
        self._suspend_update = False

    def parse_xml(self):
        if self.xml_source is None:
            return
        parser = XMLFletParser(self.xml_source, self)
        parser.parse()
        self.update()

    def update(self):
        if self.page is not None and not self._suspend_update:
            super().update()


class ftUserRow(ft.Column):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xml_source = None

    def parse_xml(self):
        if self.xml_source is None:
            return
        parser = XMLFletParser(self.xml_source, self)
        parser.parse()
        self.update()

    def update(self):
        if self.page is not None:
            super().update()
