from SLM.appGlue.core import Service


class ProgressManager(Service):
    def __init__(self):
        super().__init__()
        self.visualizers = []
        self.progress = 0
        self.max_progress = 100
        self.message = ""
        self.description = ""

    def add_visualizer(self, visualizer):
        self.visualizers.append(visualizer)

    def set_description(self, description):
        self.description = description
        self.update()

    def step(self, message=""):
        self.progress += 1
        self.message = message
        self.update()

    def reset(self):
        self.progress = 0
        self.message = ""
        self.description = ""
        self.update()

    def update(self):
        for visualizer in self.visualizers:
            visualizer.update_progress()




class ProgressVisualizer:
    # todo replace by callback
    def update_progress(self):
        pass
