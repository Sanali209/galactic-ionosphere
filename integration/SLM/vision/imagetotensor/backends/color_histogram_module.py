from SLM.modularity.modulemanager import Module, ModuleManager


@ModuleManager().register()
class ch_backend_module(Module):
    def __init__(self):
        super().__init__()
        self.import_path = "SLM.vision.imagetotensor.backends.color_histogram"
