import flet as ft

from SLM.flet.flet_ext import Flet_view, FletGlueApp, ftCollapsiblePanelVertical


class obj_editor_view(Flet_view):
    def __init__(self):
        super().__init__()
        row = ft.Row(expand=True)
        self.controls.append(row)
        panel1 = ftCollapsiblePanelVertical(expand=1)
        panel2 = ft.Column(expand=1, controls=[ft.Text("text")])
        panel3 = ft.Column(expand=1, controls=[ft.Text("text")])
        row.controls.extend([panel1, panel2, panel3])


if __name__ == "__main__":
    view = obj_editor_view()
    app = FletGlueApp()
    app.set_view(view)

    app.run()
