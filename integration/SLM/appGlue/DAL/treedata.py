from SLM.appGlue.DesignPaterns.specification import Specification


class TreeDataView:
    dict_child_list = {}

    @staticmethod
    def get_child_list(data_item):
        try:
            child_list = TreeDataView.dict_child_list[data_item]
        except:
            child_list = []
            TreeDataView.dict_child_list[data_item] = child_list
        return child_list

    def __init__(self):

        self.source_tree_ref = object()
        self.specification: Specification = None
        self._view_changed_call_backs = []
        self.get_child_list = TreeDataView.get_child_list
        self.suspend_view_changed = False

    def __iter__(self):
        for item in self.get_child_list(self.source_tree_ref):
            if self._is_satisfied(item):
                yield item

    def iter(self, data_item):
        for item in self.get_child_list(data_item):
            if self._is_satisfied(item):
                yield item

    def _is_satisfied(self, data_item):
        if self.specification is None: return True
        return self.specification.is_satisfied_by(data_item)

    def set_specification(self, spec: Specification):
        self.specification = spec
        self._fire_view_changed()

    def _fire_view_changed(self):
        if self.suspend_view_changed: return
        for callback in self._view_changed_call_backs:
            callback(self)

    def add_view_changed_callback(self, callback):
        self._view_changed_call_backs.append(callback)

    def add_to_tree(self, item, parent=None):
        if parent is None:
            parent = self.source_tree_ref
        child_list = self.get_child_list(parent)
        child_list.append(item)
        if self._is_satisfied(item):
            self._fire_view_changed()

    def remove(self, item, parent=None):
        if parent is None:
            parent = self.source_tree_ref
        child_list = self.get_child_list(parent)
        child_list.remove(item)
        if self._is_satisfied(item):
            self._fire_view_changed()

    def set_source(self, sorce_list):
        self.suspend_view_changed = True
        for item in sorce_list:
            self.add_to_tree(item)
        self.suspend_view_changed = False
        self._fire_view_changed()

