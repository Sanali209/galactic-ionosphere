class SelectionManager:
    def __init__(self):
        self.selectionUsers = []
        self.on_selection_changed = []
        self.last_selection = None

    def fire_selection_changed(self):
        for callback in self.on_selection_changed:
            callback(self)

    def register_user(self, user):
        self.selectionUsers.append(user)
        user.parent_manager = self

    def unregister_user(self, user):
        self.selectionUsers.remove(user)
        user.parent_manager = None

    def get_selection_data(self):
        sel_list = []
        for user in self.selectionUsers:
            if user.selected:
                sel_list.append(user.selection_data)
        return sel_list

    def get_selection_as(self, type=None):
        sel_list = []
        for user in self.selectionUsers:
            if user.selected:
                sel_list.append(user.selection_data.get_selection_as(type))
        return sel_list

    def clear_selection(self):
        for user in self.selectionUsers:
            user.set_selected(False)
        self.fire_selection_changed()


class SelectionManagerUser:
    def __init__(self):
        self.selected = False
        self.parent_manager = None
        self.selection_data = SelectionItemData()

    def set_selected(self, selected):
        self.selected = selected
        if self.parent_manager is not None:
            self.parent_manager.last_selection = self.selection_data
            self.parent_manager.fire_selection_changed()

    def __hash__(self):
        if self.selection_data.selection is None:
            return 0
        return self.selection_data.selection.__hash__()


class SelectionItemData:
    def __init__(self):
        self.selection = None
        self.selection_converters = {}

    def get_selection_as(self, type):
        if type in self.selection_converters:
            return self.selection_converters[type](self.selection)
        return self.selection
