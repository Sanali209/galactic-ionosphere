from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from applications.collectionTools.components.event_dispatcher import EventDispatcher


class SelectionManager:
    # move to global component
    def __init__(self):
        MessageSystem.Subscribe("on_collection_record_deleted", self, self.on_collection_record_deleted)
        self.selection = []
        self.on_selection_changed = []
        EventDispatcher().register_listener("on_select", self.set_selection)

    def on_collection_record_deleted(self, record):
        if record in self.selection:
            self.selection.remove(record)
            self.fire_selection_changed()

    def set_selection(self, selection, *args, **kwargs):
        mode = kwargs.get("mode", None)
        if selection is None:
            selection = []
        if mode is None:
            self.selection = selection
        elif mode == "single":
            self.selection = selection
        elif mode == "multi":
            # todo not good logic
            self.selection = selection + self.selection
            self.selection = list(set(self.selection))
        self.fire_selection_changed(*args, **kwargs)

    def register_on_selection_changed(self, callback):
        self.on_selection_changed.append(callback)

    def fire_selection_changed(self, *args, **kwargs):
        for callback in self.on_selection_changed:
            callback(self.selection, *args, **kwargs)

    def get_selection(self):
        return self.selection
