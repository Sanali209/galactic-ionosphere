from SLM.modularity.modulemanager import Module, ModuleManager


@ModuleManager().register()
class resnet_backend_module(Module):
    def __init__(self):
        super().__init__("CNNResNetBackend")
        self.import_path = "SLM.vision.imagetotensor.backends.resnet"