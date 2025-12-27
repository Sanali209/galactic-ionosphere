class Blackboard:
    def __init__(self):
        self.data = {}

    def set_value(self, key, value):
        self.data[key] = value

    def get_value(self, key):
        return self.data.get(key)

    def clear(self):
        self.data.clear()


class Node:
    def run(self, blackboard):
        raise NotImplementedError("This method should be overridden by subclasses.")


class ActionNode(Node):
    def __init__(self, action):
        self.action = action

    def run(self, blackboard):
        return self.action(blackboard)


class CompositeNode(Node):
    def __init__(self):
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class Selector(CompositeNode):
    # this implementation Selector of behavior tree
    # will return success if any of the child nodes return success
    def run(self, blackboard):
        for child in self.children:
            result = child.run(blackboard)
            if result == "success":
                return "success"
        return "failure"


class Sequence(CompositeNode):
    def run(self, blackboard):
        for child in self.children:
            result = child.run(blackboard)
            if result == "failure":
                return "failure"
            elif result == "running":
                return "running"
        return "success"


class Sequence(CompositeNode):
    def __init__(self):
        super().__init__()
        self.cur_child = None

    def run(self, blackboard):
        self.cur_child = 0
        child = self.children[self.cur_child]
        result = child.run(blackboard)
        if result == "running":
            return "running"
        if result == "success":
            if self.cur_child + 1 < len(self.children):
                self.cur_child += 1
                return "running"
            else:
                self.cur_child = 0
                return "success"
            return "success"
        if result == "failure":
            self.cur_child = 0
            return "failure"

class Inverter(ActionNode):
    def __init__(self, node):
        self.node = node

    def run(self, blackboard):
        res = self.node.run(blackboard)
        if (res == "success"):
            return 'failure'
        else:
            return 'success'


class ForceSuccess(ActionNode):
    def __init__(self, node):
        self.node = node

    def run(self, blackboard):
        self.node.run(blackboard)
        return "success"


class ForceFailure(ActionNode):
    def __init__(self, node):
        self.node = node

    def run(self, blackboard):
        self.node.run(blackboard)
        return "failure"
