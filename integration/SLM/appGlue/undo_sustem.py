class UndoItem:
    def __init__(self, undo, redo):
        self.undo = undo
        self.redo = redo

    def undo(self):
        self.undo()

    def redo(self):
        self.redo()


class UndoSystem:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def add_undo_item(self, undo, redo):
        undo_item = UndoItem(undo, redo)
        self.undo_stack.append(undo_item)
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) == 0:
            return
        undo_item = self.undo_stack.pop()
        undo_item.undo()
        self.redo_stack.append(undo_item)

    def redo(self):
        if len(self.redo_stack) == 0:
            return
        redo_item = self.redo_stack.pop()
        redo_item.redo()
        self.undo_stack.append(redo_item)

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def get_undo_stack(self):
        return self.undo_stack

    def get_redo_stack(self):
        return self.redo_stack
