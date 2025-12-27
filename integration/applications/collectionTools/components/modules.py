from SLM.appGlue.DesignPaterns.SingletonAtr import singleton


class AppModule:

    def initialize(self):
        pass

    def handle_event(self, event):
        pass

    def unload(self):
        pass


@singleton
class ModuleManager:
    def __init__(self):
        self.modules = {}

    def load_module(self, module_type, module):
        self.modules[module_type] = module
        module.initialize()

    def unload_module(self, module_type):
        self.modules[module_type].unload()
        del self.modules[module_type]

    def get_module(self, module_type):
        return self.modules[module_type]


class SelectionManager(AppModule):
    # move to global component
    def __init__(self):
        super().__init__()
        self.selection = []
        self.on_selection_changed_callback = []
        self.multi_selection = False

    def register_on_selection_changed(self, callback):
        self.on_selection_changed_callback.append(callback)

    def dispatch_selection_changed(self):
        for callback in self.on_selection_changed_callback:
            callback(self.selection)

    def set_selection(self, selection):
        if selection is None:
            selection = []
        self.selection = selection
        self.dispatch_selection_changed()

    def get_selection(self):
        return self.selection

    def count_selection(self):
        return len(self.selection)


class FocusedModule(AppModule):
    def __init__(self):
        self.focused_module = None

    def set_focused_module(self, module_type):
        self.focused_module = module_type

    def get_focused_module(self):
        return ModuleManager().get_module(self.focused_module)
